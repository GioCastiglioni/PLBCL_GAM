import torch
from torch.nn import functional as F
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.nn import all_reduce as functional_all_reduce
from torch.distributed.nn import ReduceOp
import math
import copy
from multiprocessing import Value
from plbcl_gam.encoders.vit import Block
from torch.nn.init import trunc_normal_
from plbcl_gam.encoders.pos_embed import get_2d_sincos_pos_embed_with_scale

class BCLSegmentationLoss(nn.Module):
    def __init__(self, num_classes, tau=0.1, max_anchors=1024, max_context=2048, ignore_index=-1):
        super(BCLSegmentationLoss, self).__init__()
        self.num_classes = num_classes
        self.tau = tau
        self.max_anchors = max_anchors
        self.max_context = max_context
        self.ignore_index = ignore_index

    def _sample_pixels(self, features, targets, num_samples):
        b, d, h, w = features.shape
        feat_flat = features.permute(0, 2, 3, 1).reshape(-1, d)
        targ_flat = targets.reshape(-1)
        
        mask = targ_flat != self.ignore_index
        feat_valid = feat_flat[mask]
        targ_valid = targ_flat[mask]
        
        if feat_valid.size(0) == 0:
            return None, None

        if feat_valid.size(0) > num_samples:
            perm = torch.randperm(feat_valid.size(0), device=features.device)[:num_samples]
            return feat_valid[perm], targ_valid[perm]
        else:
            return feat_valid, targ_valid

    def forward(self, z1, z2, prototypes, targets1, targets2):
        device = z1.device
        
        anchors, anchors_labels = self._sample_pixels(z1, targets1, self.max_anchors)
        
        if anchors is None:
            return 0.0 * z1.sum()

        ctx1, ctx1_labels = self._sample_pixels(z1, targets1, self.max_context // 2)
        ctx2, ctx2_labels = self._sample_pixels(z2, targets2, self.max_context // 2)
        
        ctx_list = []
        lbl_list = []
        if ctx1 is not None: 
            ctx_list.append(ctx1); lbl_list.append(ctx1_labels)
        if ctx2 is not None: 
            ctx_list.append(ctx2); lbl_list.append(ctx2_labels)
            
        if not ctx_list:
            return torch.tensor(0.0, device=device, requires_grad=True)

        context_features = torch.cat(ctx_list, dim=0)
        context_labels = torch.cat(lbl_list, dim=0)

        anchors = anchors.float()
        context_features = context_features.float()
        prototypes = prototypes.float()

        anchors = F.normalize(anchors, dim=1)
        context_features = F.normalize(context_features, dim=1)
        prototypes = F.normalize(prototypes, dim=1)

        pool_features = torch.cat([context_features, prototypes], dim=0)
        
        proto_labels = torch.arange(self.num_classes, device=device)
        pool_labels = torch.cat([context_labels, proto_labels], dim=0)

        sim_matrix = torch.matmul(anchors, pool_features.T) / self.tau
        
        sim_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
        sim_matrix_shifted = sim_matrix - sim_max.detach()
        exp_sim = torch.exp(sim_matrix_shifted)

        pool_one_hot = F.one_hot(pool_labels, num_classes=self.num_classes).float()
        sum_exp_per_class = torch.matmul(exp_sim, pool_one_hot)
        cardinality = pool_one_hot.sum(dim=0).clamp(min=1.0)
        avg_exp_per_class = sum_exp_per_class / cardinality.view(1, -1)

        bcl_denominator = avg_exp_per_class.sum(dim=1, keepdim=True)
        
        log_prob_matrix = sim_matrix_shifted - torch.log(bcl_denominator + 1e-7)
        
        mask_positives = (anchors_labels.unsqueeze(1) == pool_labels.unsqueeze(0)).float()
        log_probs_pos = (log_prob_matrix * mask_positives).sum(dim=1)
        
        num_positives = mask_positives.sum(dim=1).clamp(min=1.0)
        loss_per_anchor = - (log_probs_pos / num_positives)
        
        return loss_per_anchor.mean()
    

class BalancedContrastiveLearning(nn.Module):
    def __init__(
            self,
            num_classes,
            distribution,
            ignore_index=-1,
            focal=False,
            gamma=2.0,
            lamb=2.0,
            mu=0.6,
            temperature=0.1,
            in_channels=64,
            hidden_d=512,
            out_d=128,
            max_anchors=4096,
            max_context=32768,
        ):
        super(BalancedContrastiveLearning, self).__init__()
        self.num_classes = num_classes
        self.distribution = distribution
        self.ignore_index = ignore_index
        self.lamb = lamb
        self.mu = mu
        self.temperature = temperature
        self.in_channels = in_channels
        self.hidden_d = hidden_d
        self.out_d = out_d

        self.LC = torch.nn.CrossEntropyLoss(ignore_index=self.ignore_index) if not focal else FocalLossSoftMax(gamma=gamma, ignore_index=self.ignore_index)
        self.BCL = BCLSegmentationLoss(
            self.num_classes, 
            tau=self.temperature,
            max_anchors=max_anchors, 
            max_context=max_context,
            ignore_index=self.ignore_index
        )

    def forward(self, logits, z2, z3, targets, targets2, targets3, prototypes):

        LC = self.LC(logits, targets)

        z2 = self.views_mlp(z2)
        z3 = self.views_mlp(z3)
        prototypes = self.prot_mlp(prototypes).flatten(start_dim=1)

        BCL = self.BCL(z2, z3, prototypes, targets2, targets3)
        
        return self.lamb*LC + self.mu*BCL
    
    def __str__(self):
        return 'BalancedContrastiveLearning'


class FocalLossSoftMax(nn.Module):
    def __init__(
        self,
        gamma: float = 2.0,
        ignore_index: int = -1,
    ):
        super(FocalLossSoftMax, self).__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, input: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # input: [B, C, H, W]
        # targets: [B, H, W]
        
        log_p = F.log_softmax(input, dim=1)
        ce_loss = F.nll_loss(log_p, targets, reduction='none', ignore_index=self.ignore_index)
        
        with torch.no_grad():
            p = torch.exp(log_p)
            # targets: [B, H, W] -> [B, 1, H, W]
            gathered_p = torch.gather(p, dim=1, index=targets.unsqueeze(1).clamp(min=0))
            gathered_p = gathered_p.squeeze(1)
        
        focal_weight = (1.0 - gathered_p) ** self.gamma
        loss = focal_weight * ce_loss
        
        if self.ignore_index >= 0:
            mask = targets != self.ignore_index
            return loss[mask].mean()
        
        return loss.mean()
    
    def __str__(self):
        return 'FocalLoss'