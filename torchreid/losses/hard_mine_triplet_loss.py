from __future__ import absolute_import
from __future__ import division

import torch
import torch.nn as nn


class TripletLoss(nn.Module):
    """Triplet loss with hard positive/negative mining.
    
    Reference:
    Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py.
    
    Args:
    - margin (float): margin for triplet.
    """
    def __init__(self, margin=0.3, soft=True):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.soft = soft
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        """
        Args:
        - inputs: feature matrix with shape (batch_size, feat_dim)
        - targets: ground truth labels with shape (num_classes)
        """
        n = inputs.size(0)
        
        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        
        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        
        # # Compute ranking hinge loss
        # y = torch.ones_like(dist_an)
        # loss = self.ranking_loss(dist_an, dist_ap, y)
        # return loss

        if self.soft:
            return torch.log(1 + torch.exp(dist_ap - dist_an)).mean()
        else:
            return self.ranking_loss(dist_an, dist_ap, torch.ones_like(dist_an))


class WeightedTripletLoss(object):
    """Related Weighted Triplet Loss theory can be found in paper
    'Attention Network Robustification for Person ReID'."""

    def __init__(self, margin=None):
        self.margin = margin

        if margin is not None:
            self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        else:
            self.ranking_loss = nn.SoftMarginLoss()

    def __call__(self, global_feat, labels, normalize_feature=False):
        if normalize_feature:
            global_feat = self.normalize(global_feat, axis=-1)

        dist_mat = self.euclidean_dist(global_feat, global_feat)
        dist_ap, dist_an, final_wp, final_wn = self.soft_example_mining(dist_mat, labels)
        y = final_wn.new().resize_as_(final_wn).fill_(1)
        if self.margin is not None:
            loss = self.ranking_loss(dist_an, dist_ap, y)
        else:
            loss = self.ranking_loss(final_wn - final_wp, y)
        # return loss, dist_ap, dist_an
        return loss

    def euclidean_dist(self, x, y):
        """
        Args:
          x: pytorch Variable, with shape [m, d]
          y: pytorch Variable, with shape [n, d]
        Returns:
          dist: pytorch Variable, with shape [m, n]
        """
        m, n = x.size(0), y.size(0)
        xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
        yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
        dist = xx + yy
        dist.addmm_(1, -2, x, y.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        return dist

    def normalize(self, x, axis=-1):
        """Normalizing to unit length along the specified dimension.
        Args:
          x: pytorch Variable
        Returns:
          x: pytorch Variable, same shape as input
        """
        x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
        return x

    def soft_example_mining(self, dist_mat, labels):
        eps = 1e-12
        assert len(dist_mat.size()) == 2
        assert dist_mat.size(0) == dist_mat.size(1)
        N = dist_mat.size(0)
        # shape [N, N]
        is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
        is_neg = labels.expand(N, N).ne(labels.expand(N, N).t())

        # shape [N]
        dist_ap = dist_mat[is_pos].contiguous().view(N, -1)
        dist_an = dist_mat[is_neg].contiguous().view(N, -1)

        exp_dist_ap = torch.exp(dist_ap)
        exp_dist_an = torch.exp(-dist_an)

        wp = exp_dist_ap / (exp_dist_ap.sum(1, keepdim=True) + eps)
        wn = exp_dist_an / (exp_dist_an.sum(1, keepdim=True) + eps)

        final_wp = (wp * dist_ap).sum(1)
        final_wn = (wn * dist_an).sum(1)

        return dist_ap, dist_an, final_wp, final_wn