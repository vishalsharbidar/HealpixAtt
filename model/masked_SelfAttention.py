import torch
import torch.nn as nn
from torch_geometric.utils import to_dense_adj

from torch_geometric.nn import ChebConv, knn_graph
from typing import List, Tuple
from copy import deepcopy
from pathlib import Path


def MLP(channels: List[int], do_bn: bool = True) -> nn.Module:
    """ Multi-layer perceptron """
    n = len(channels)
    layers = []
    for i in range(1, n):
        layers.append(
            nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, bias=True))
        if i < (n-1):
            if do_bn:
                layers.append(nn.BatchNorm1d(channels[i]))
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)

def find_exp_mask(edges, head, scores):
    print(edges.shape, len(edges.shape))
    if len(edges.shape) < 3:
        m = to_dense_adj(edges).unsqueeze(0)
        mask = torch.cat(([m]*head), 1)
    else:
        stack = torch.tensor([]).to(edges)
        for i in edges:
            m = to_dense_adj(i).unsqueeze(0)
            mask = torch.cat(([m]*head), 1)
            stack = torch.cat((stack, mask), dim=0)
    
    masking_scores = torch.mul(scores, mask)
    exp_masking_scores = torch.exp(masking_scores) 
    return exp_masking_scores

def attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, edges: torch.Tensor) -> Tuple[torch.Tensor,torch.Tensor]:
    dim = query.shape[1]
    head = query.shape[2]
    scores = torch.einsum('bdhn,bdhm->bhnm', query, key) / dim**.5
    exp_masking_scores = find_exp_mask(edges, head, scores)
    exp_masking_scores_sum = exp_masking_scores.sum(dim=-1)
    prob = exp_masking_scores / exp_masking_scores_sum[:,:,:,None] 
    return torch.einsum('bhnm,bdhm->bdhn', prob, value), prob


class MultiHeadedAttention(nn.Module):
    """ Multi-head attention to increase model expressivitiy """
    def __init__(self, num_heads: int, d_model: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.dim = d_model // num_heads
        self.num_heads = num_heads
        self.merge = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.proj = nn.ModuleList([deepcopy(self.merge) for _ in range(3)])

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
        batch_dim = query.size(0)      
        query, key, value = [l(x).view(batch_dim, self.dim, self.num_heads, -1)
                             for l, x in zip(self.proj, (query, key, value))]        

        x, _ = attention(query, key, value, edges)
        return self.merge(x.contiguous().view(batch_dim, self.dim*self.num_heads, -1))


class AttentionalPropagation(nn.Module):
    def __init__(self, feature_dim: int, num_heads: int):
        super().__init__()
        self.attn = MultiHeadedAttention(num_heads, feature_dim)
        self.mlp = MLP([feature_dim*2, feature_dim*2, feature_dim])
        nn.init.constant_(self.mlp[-1].bias, 0.0)

    def forward(self, x: torch.Tensor, source: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
        message = self.attn(x, source, source, edges)
        return self.mlp(torch.cat([x, message], dim=1))


class SelfAttentionalGNN(nn.Module):
    def __init__(self, feature_dim: int, layer_names: List[str]) -> None:
        super().__init__()
        self.layers = nn.ModuleList([AttentionalPropagation(feature_dim, 4) for _ in range(len(layer_names))])
        self.names = layer_names

    def forward(self, desc0: torch.Tensor, edges0: torch.Tensor, desc1: torch.Tensor, edges1: torch.Tensor) -> Tuple[torch.Tensor,torch.Tensor]:
        for layer, name in zip(self.layers, self.names):
            src0, src1 = desc0, desc1
            delta0, delta1 = layer(desc0, src0, edges0), layer(desc1, src1, edges1)
            desc0, desc1 = (desc0 + delta0), (desc1 + delta1)
        return desc0, desc1
