import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb


def consistency_loss(logits_w, logits_s, name='ce', T=1.0, p_cutoff=0.0, use_hard_labels=True):
    assert name in ['ce', 'L2']
    logits_w = logits_w.detach()
    if name == 'L2':
        assert logits_w.size() == logits_s.size()
        return F.mse_loss(logits_s, logits_w, reduction='mean')
    
    elif name == 'L2_mask':
        pass

    elif name == 'ce':
        # Logit input -> T 0.1 (Not Passed Agg Module) -> 68.57% 통과
        # W/O T: cutoff=0.5 이상-> 10% (7/70개)
        # Nonreliable sample 다수
        pseudo_label = torch.softmax(logits_w, dim=-1)
        max_probs, max_idx = torch.max(pseudo_label, dim=-1)
        mask = max_probs.ge(p_cutoff).float()
        
        if use_hard_labels:
            masked_loss = ce_loss(logits_s, max_idx, use_hard_labels, reduction='none') * mask
        else:
            pseudo_label = torch.softmax(logits_w/T, dim=-1)
            masked_loss = ce_loss(logits_s, pseudo_label, use_hard_labels) * mask
        return masked_loss.mean(), mask.mean()

    else:
        assert Exception('Not Implemented consistency_loss')

def log_loss(logits_p, prob_p, logits_q, prob_q, mask = None):
    eps = 1e-20 # P = Target
    if mask == None:
        mask = torch.ones(len(prob_p), device=prob_p.device)
    prob_p = prob_p if prob_p is not None else F.softmax(logits_p, dim=1)
    logq = F.log_softmax(logits_q, dim=1) if logits_q is not None else torch.log(prob_q + eps)
    return -torch.mean(torch.sum(prob_p.detach() * logq, dim=1)*mask.detach())

def consistency_loss_prob(probs_w, probs_s, name='ce', T=1.0, p_cutoff=0.0, use_hard_labels=True):
    assert name in ['ce', 'L2']
    probs_w = probs_w.detach()
    if name == 'L2':
        assert probs_w.size() == probs_s.size()
        return F.mse_loss(probs_s, probs_w, reduction='mean')

    elif name == 'L2_mask':
        pass

    elif name == 'ce':
        pseudo_label = probs_w
        max_probs, max_idx = torch.max(pseudo_label, dim=-1)
        mask = max_probs.ge(p_cutoff).float()
        if use_hard_labels:
            target_xl_2D = torch.zeros_like(probs_w, device=probs_w.device)
            target_xl_2D.scatter_(dim=1, index=max_idx.unsqueeze(1).repeat(1,1).reshape(-1,1), value=1.)
            masked_loss = log_loss(None, target_xl_2D, None, probs_s, mask)

        else:
            masked_loss = log_loss(None, probs_w, None, probs_s, mask)
        return masked_loss, mask.mean()

    else:
        assert Exception('Not Implemented consistency_loss')

def consistency_loss_wo_softmax(logits_w, logits_s, name='ce', T=1.0, p_cutoff=0.0, use_hard_labels=True):
    assert name in ['ce', 'L2'] # logits_w = Agg output Prob / logits_s = encoder output Logit
    logits_w = logits_w.detach()
    if name == 'L2':
        assert logits_w.size() == logits_s.size()
        return F.mse_loss(logits_s, logits_w, reduction='mean')
    
    elif name == 'L2_mask':
        pass

    elif name == 'ce':
        pseudo_label = torch.clamp(logits_w, min=0.0, max=1.0)  # Prob 
        max_probs, max_idx = torch.max(pseudo_label, dim=-1)
        mask = max_probs.ge(p_cutoff).float()
        nllloss = nn.NLLLoss(reduction='none')
        if use_hard_labels:
            masked_loss = ce_loss(logits_s, max_idx, use_hard_labels, reduction='none') * mask
            # masked_loss = nllloss(F.log_softmax(logits_s, dim=-1), max_idx) * mask
        else:
            pseudo_label = logits_w
            masked_loss = log_loss(None, logits_s, None, pseudo_label, mask=mask) * mask
        return masked_loss.mean(), mask.mean()

    else:
        assert Exception('Not Implemented consistency_loss')
def consistency_loss_inv_softmax(logits_w, logits_s, name='ce', T=1.0, p_cutoff=0.0, use_hard_labels=True):
    assert name in ['ce', 'L2'] # logits_w = logit / logits_s = agg output prob
    logits_w = logits_w.detach()
    if name == 'L2':
        assert logits_w.size() == logits_s.size()
        return F.mse_loss(logits_s, logits_w, reduction='mean')
    
    elif name == 'L2_mask':
        pass

    elif name == 'ce':
        pseudo_label = torch.softmax(logits_w, dim=-1)  # Prob 
        max_probs, max_idx = torch.max(pseudo_label, dim=-1)
        mask = max_probs.ge(p_cutoff).float()
        nllloss = nn.NLLLoss(reduction='none')
        if use_hard_labels:
            masked_loss = nllloss(torch.log(torch.clamp(logits_s, min=1e-20, max=1.0)), max_idx) * mask
            # masked_loss = nllloss(F.log_softmax(logits_s, dim=-1), max_idx) * mask
        else:
            pseudo_label = logits_w
            masked_loss = log_loss(None, logits_s, None, pseudo_label, mask=mask) * mask
        return masked_loss.mean(), mask.mean()

    else:
        assert Exception('Not Implemented consistency_loss')

def ce_loss(logits, targets, use_hard_labels=True, reduction='none'):
    """
    wrapper for cross entropy loss in pytorch.
    
    Args
        logits: logit values, shape=[Batch size, # of classes]
        targets: integer or vector, shape=[Batch size] or [Batch size, # of classes]
        use_hard_labels: If True, targets have [Batch size] shape with int values. If False, the target is vector (default True)
    """
    if use_hard_labels:
        return F.cross_entropy(logits, targets, reduction=reduction)
    else:
        assert logits.shape == targets.shape
        log_pred = F.log_softmax(logits, dim=-1)
        nll_loss = torch.sum(-targets*log_pred, dim=1)
        return nll_loss


class Losses:
    @classmethod
    def sharpen(cls, p, T=0.1):
        sharp_p = p**(1./T)
        sharp_p /= torch.sum(sharp_p, dim=1, keepdim=True)
        return sharp_p

    @classmethod
    def me_max(cls, probs):
        avg_probs = torch.mean(cls.sharpen(probs), dim=0)
        return torch.sum(torch.log(avg_probs**(-avg_probs)))


class FixMatchLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, *args, **kwargs):
        return consistency_loss_wo_softmax(*args, **kwargs)


class BidirectionalLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.CosineSimilarity(dim=1)
        
    def forward(self, logits_x_ulb_1, logits_x_ulb_2, T, p_cutoff, use_hard_labels,
        anchor_feat=None, positive_feat=None, anchor_pred_feat=None, positive_pred_feat=None,
        simsiam=False, me_max=True, sharpen=True):
        prob_x_ulb_1 = F.softmax(logits_x_ulb_1, dim=1)
        prob_x_ulb_2 = F.softmax(logits_x_ulb_2, dim=1)
        #compare by max_value
        max_probs_x_ulb_1, _ = torch.max(prob_x_ulb_1, dim=-1)
        max_probs_x_ulb_2, _ = torch.max(prob_x_ulb_2, dim=-1)
        logits_x_ulb_gt = torch.empty_like(logits_x_ulb_1)
        logits_x_ulb_pd = torch.empty_like(logits_x_ulb_1)

        gt_idx_bool = torch.ge(max_probs_x_ulb_1, max_probs_x_ulb_2)

        logits_x_ulb_gt[gt_idx_bool] = logits_x_ulb_1[gt_idx_bool]
        logits_x_ulb_gt[~gt_idx_bool] = logits_x_ulb_2[~gt_idx_bool]
        logits_x_ulb_pd[gt_idx_bool] = logits_x_ulb_2[gt_idx_bool]
        logits_x_ulb_pd[~gt_idx_bool] = logits_x_ulb_1[~gt_idx_bool]

        logits_loss = consistency_loss(logits_x_ulb_gt,
                                        logits_x_ulb_pd,
                                        'ce', T, p_cutoff,
                                        use_hard_labels=use_hard_labels)
        if simsiam:
            feat_gt = torch.empty_like(anchor_feat)
            feat_pd = torch.empty_like(positive_feat)

            feat_gt[gt_idx_bool] = anchor_feat[gt_idx_bool]
            feat_gt[~gt_idx_bool] = positive_feat[~gt_idx_bool]
            feat_pd[gt_idx_bool] = positive_pred_feat[gt_idx_bool]
            feat_pd[~gt_idx_bool] = anchor_pred_feat[~gt_idx_bool]

            feat_gt = feat_gt.detach()

            feat_loss = -self.criterion(feat_gt, feat_pd).mean() * 0.5
            return logits_loss[0] + feat_loss, logits_loss[1]

        return logits_loss


class BidirectionalLoss_all(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.CosineSimilarity(dim=1)

    def forward(self, logits_x_ulb_1, logits_x_ulb_2, logits_x_ulb_1_agg, logits_x_ulb_2_agg, T, p_cutoff, use_hard_labels,
                anchor_feat=None, positive_feat=None, anchor_pred_feat=None, positive_pred_feat=None,
                simsiam=False, me_max=True, sharpen=True):
        prob_x_ulb_1 = F.softmax(logits_x_ulb_1, dim=1)
        prob_x_ulb_2 = F.softmax(logits_x_ulb_2, dim=1)
        prob_x_ulb_1_agg = F.softmax(logits_x_ulb_1_agg, dim=1)
        prob_x_ulb_2_agg = F.softmax(logits_x_ulb_2_agg, dim=1)

        # compare by max_value
        max_probs_x_ulb_1, _ = torch.max(prob_x_ulb_1, dim=-1)
        max_probs_x_ulb_2, _ = torch.max(prob_x_ulb_2, dim=-1)
        max_prob_x_ulb_1_agg, _ = torch.max(prob_x_ulb_1_agg, dim=-1)
        max_prob_x_ulb_2_agg, _ = torch.max(prob_x_ulb_2_agg, dim=-1)
        max_all = torch.cat([max_probs_x_ulb_1.unsqueeze(0), max_probs_x_ulb_2.unsqueeze(0)], dim=0)
        max_all = torch.cat([max_all, max_prob_x_ulb_1_agg.unsqueeze(0)], dim=0)
        max_all = torch.cat([max_all, max_prob_x_ulb_2_agg.unsqueeze(0)], dim=0)
        max_prob_max_all, max_index_max_all = torch.max(max_all, dim=0)
        gt_idx_bool = torch.zeros_like(max_all, device=max_all.device)
        gt_idx_bool.scatter_(dim=0, index=max_index_max_all.unsqueeze(0), value=1)
        logits_x_ulb_gt = torch.empty_like(logits_x_ulb_1)
        gt_idx_bool = gt_idx_bool.long()
        logits_x_ulb_gt[gt_idx_bool[0]] = logits_x_ulb_1[gt_idx_bool[0]]
        logits_x_ulb_gt[gt_idx_bool[1]] = logits_x_ulb_2[gt_idx_bool[1]]
        logits_x_ulb_gt[gt_idx_bool[2]] = logits_x_ulb_1_agg[gt_idx_bool[2]]
        logits_x_ulb_gt[gt_idx_bool[3]] = logits_x_ulb_2_agg[gt_idx_bool[3]]
        pdb.set_trace()
        logits_loss_1, mask1 = consistency_loss(logits_x_ulb_gt,
                                       logits_x_ulb_1,
                                       'ce', T, p_cutoff,
                                       use_hard_labels=use_hard_labels)
        logits_loss_2, mask2 = consistency_loss(logits_x_ulb_gt,
                                       logits_x_ulb_2,
                                       'ce', T, p_cutoff,
                                       use_hard_labels=use_hard_labels)
        logits_loss_3, mask3  = consistency_loss(logits_x_ulb_gt,
                                       logits_x_ulb_1_agg,
                                       'ce', T, p_cutoff,
                                       use_hard_labels=use_hard_labels)
        logits_loss_4, mask4 = consistency_loss(logits_x_ulb_gt,
                                       logits_x_ulb_2_agg,
                                       'ce', T, p_cutoff,
                                       use_hard_labels=use_hard_labels)

        return [logits_loss_1, logits_loss_2, logits_loss_3 ,logits_loss_4], [mask1, mask2, mask3, mask4]


class BidirectionalLoss_prob(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.CosineSimilarity(dim=1)
    #logits_x_ulb can be logit and prob -> always prob
    def forward(self, probs_x_ulb_1, probs_x_ulb_2, T, p_cutoff, use_hard_labels,
                anchor_feat=None, positive_feat=None, anchor_pred_feat=None, positive_pred_feat=None,
                simsiam=False, me_max=True, sharpen=True):

        # compare by max_value
        max_probs_x_ulb_1, _ = torch.max(probs_x_ulb_1, dim=-1)
        max_probs_x_ulb_2, _ = torch.max(probs_x_ulb_2, dim=-1)
        probs_x_ulb_gt = torch.empty_like(probs_x_ulb_1)
        probs_x_ulb_pd = torch.empty_like(probs_x_ulb_2)

        gt_idx_bool = torch.ge(max_probs_x_ulb_1, max_probs_x_ulb_2)

        probs_x_ulb_gt[gt_idx_bool] = probs_x_ulb_1[gt_idx_bool]
        probs_x_ulb_gt[~gt_idx_bool] = probs_x_ulb_2[~gt_idx_bool]
        probs_x_ulb_pd[gt_idx_bool] = probs_x_ulb_2[gt_idx_bool]
        probs_x_ulb_pd[~gt_idx_bool] = probs_x_ulb_1[~gt_idx_bool]

        logits_loss = consistency_loss_wo_softmax(probs_x_ulb_gt,
                                       probs_x_ulb_pd,
                                       'ce', T, p_cutoff,
                                       use_hard_labels=use_hard_labels)

        # if me_max:
        #     me_max_loss = Losses.me_max(torch.cat((logits_x_ulb_1, logits_x_ulb_2)))
        #     logits_loss = \
        #         (logits_loss[0] + me_max_loss, logits_loss[1])

        if simsiam:
            feat_gt = torch.empty_like(anchor_feat)
            feat_pd = torch.empty_like(positive_feat)

            feat_gt[gt_idx_bool] = anchor_feat[gt_idx_bool]
            feat_gt[~gt_idx_bool] = positive_feat[~gt_idx_bool]
            feat_pd[gt_idx_bool] = positive_pred_feat[gt_idx_bool]
            feat_pd[~gt_idx_bool] = anchor_pred_feat[~gt_idx_bool]

            feat_gt = feat_gt.detach()

            feat_loss = -self.criterion(feat_gt, feat_pd).mean() * 0.5
            return logits_loss[0] + feat_loss, logits_loss[1]

        return logits_loss




class PAWSLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, anchor_feat, positive_feat, support_feat, support_label):
        pass