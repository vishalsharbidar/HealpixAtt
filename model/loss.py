import numpy as np
import torch
import torch.nn.functional as F

def pairwise_cosine_dist(x1, x2):
    """
    Return pairwise half of cosine distance in range [0, 1].
    dist = (1 - cos(theta)) / 2
    """
    x1 = F.normalize(x1, dim=-1)
    x2 = F.normalize(x2, dim=-1)
    return 0.25 * torch.cdist(x1, x2).pow(2)


def criterion(y_true, y_pred, margin=None):
    gt_matches0, gt_matches1 = y_true['gt_matches0'], y_true['gt_matches1']
    if y_pred['context_descriptors0'] != None:
        gdesc0, gdesc1, scores = y_pred['context_descriptors0'], y_pred['context_descriptors1'], y_pred['scores']
        if margin is not None:
            dist = pairwise_cosine_dist(gdesc0.transpose(2, 1).contiguous(), gdesc1.transpose(2, 1).contiguous())
        else:
            dist = None
    else:
        dist, scores = y_pred['upscaled_scores'], y_pred['scores']
    # loss for keypoints with gt match
    batch_idx, idx_kpts0 = torch.where(gt_matches0 >= 0)
    idx_kpts1 = gt_matches0[batch_idx, idx_kpts0]
    idx_kpts1 = idx_kpts1.long()
    _, inv_idx, counts = torch.unique_consecutive(batch_idx, return_inverse=True, return_counts=True)
    mean_weights = (1 / counts)[inv_idx]

    #print(batch_idx, idx_kpts0, idx_kpts1.long())
    matched_loss = (-scores[batch_idx, idx_kpts0, idx_kpts1] * mean_weights).sum()
    matched_triplet_loss = matched_triplet_criterion(
        dist=dist, margin=margin,
        indexes=(batch_idx, idx_kpts0, idx_kpts1),
        mean_weights=mean_weights
    )

    # loss for unmatched keypoints in the image 0
    batch_idx, idx_kpts0 = torch.where(gt_matches0 == -1)
    _, inv_idx, counts = torch.unique_consecutive(batch_idx, return_inverse=True, return_counts=True)
    mean_weights = (1 / counts)[inv_idx]

    unmatched0_loss = (-scores[batch_idx, idx_kpts0, -1] * mean_weights).sum()
    unmatched0_margin_loss = unmatched_margin_criterion(
        dist=dist, margin=margin, indexes=(batch_idx, idx_kpts0),
        mean_weights=mean_weights, zero_to_one=True
    )

    # loss for unmatched keypoints in the image 1
    batch_idx, idx_kpts1 = torch.where(gt_matches1 == -1)
    _, inv_idx, counts = torch.unique_consecutive(batch_idx, return_inverse=True, return_counts=True)
    mean_weights = (1 / counts)[inv_idx]

    unmatched1_loss = (-scores[batch_idx, -1, idx_kpts1] * mean_weights).sum()
    unmatched1_margin_loss = unmatched_margin_criterion(
                            dist=dist, margin=margin, indexes=(batch_idx, idx_kpts1),
                            mean_weights=mean_weights, zero_to_one=False)

    loss = (matched_loss + 0.5 * (unmatched0_loss + unmatched1_loss)) / scores.size(0)
    metric_loss = (matched_triplet_loss + unmatched0_margin_loss + unmatched1_margin_loss) / scores.size(0)
    return loss
        


def matched_triplet_criterion(dist, margin, indexes, mean_weights):
    if margin is None:
        return torch.tensor(0, device=mean_weights.device)
    batch_idx, idx_kpts0, idx_kpts1 = indexes
    # distance between anchor and positive
    dist_ap = dist[batch_idx, idx_kpts0, idx_kpts1]

    dist_detached = dist.detach().clone()
    dist_detached[batch_idx, idx_kpts0, idx_kpts1] = np.inf
    idx_kpts0_closest_to_1 = torch.argmin(dist_detached, dim=1)
    idx_kpts1_closest_to_0 = torch.argmin(dist_detached, dim=2)
    idx_kpts1_neg = idx_kpts1_closest_to_0[batch_idx, idx_kpts0]
    idx_kpts0_neg = idx_kpts0_closest_to_1[batch_idx, idx_kpts1]

    dist_an0 = dist[batch_idx, idx_kpts0, idx_kpts1_neg]
    dist_an1 = dist[batch_idx, idx_kpts0_neg, idx_kpts1]

    loss0 = torch.maximum(dist_ap - dist_an0 + margin, torch.tensor(0, device=dist.device))
    loss1 = torch.maximum(dist_ap - dist_an1 + margin, torch.tensor(0, device=dist.device))
    return (loss0 * mean_weights).sum() + (loss1 * mean_weights).sum()


def unmatched_margin_criterion(dist, margin, indexes, mean_weights, zero_to_one=True):
    if margin is None:
        return torch.tensor(0, device=mean_weights.device)
    batch_idx, idx_kpts = indexes

    idx_kpts_closest = torch.argmin(dist, dim=2 if zero_to_one else 1)
    idx_kpts_neg = idx_kpts_closest[batch_idx, idx_kpts]

    # distance anchor-negative
    if zero_to_one:
        dist_an = dist[batch_idx, idx_kpts, idx_kpts_neg]
    else:
        dist_an = dist[batch_idx, idx_kpts_neg, idx_kpts]

    loss = torch.maximum(-dist_an + margin, torch.tensor(0, device=dist.device))
    return (loss * mean_weights).sum()