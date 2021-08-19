import pdb
import time
from typing import Sequence

from torch._C import set_anomaly_enabled
from utils import ForkedPdb
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from utils import ForkedPdb
# future work : add taus argument, label smoothing require

from models.transformer import Block

def visualize(atten_mat=None, agg_out=None, logits_x_ulb=None, label_gt=None):
    now = time.localtime()
    nowtime = f"{now.tm_mon}_{now.tm_mday}__{now.tm_hour}_{now.tm_min}_{now.tm_sec}"
    plt.title('Logit Maps', fontsize=8)
    plt.subplot(2,2,1)
    plt.xticks(fontsize=6, rotation=60)
    plt.yticks(fontsize=6)
    plt.imshow(F.softmax(logits_x_ulb, dim=1).detach().cpu().numpy(), cmap="inferno", aspect="auto")
    plt.colorbar(shrink=0.6)
    # Plot Attention Map 1
    plt.subplot(2,2,2)
    plt.title('Attention Maps', fontsize=8)
    plt.xticks(fontsize=6, rotation=60)
    plt.yticks(fontsize=6)
    plt.imshow(atten_mat.detach().cpu().numpy(), cmap="inferno", aspect="auto")
    plt.colorbar(shrink=0.6)
    # Plot Before Agg Predictions 2.
    plt.subplot(2,2,3)
    plt.title('Probs Before Agg Softmax', fontsize=8)
    plt.xticks(fontsize=6, rotation=60)
    plt.yticks(fontsize=6)
    class_val = label_gt.detach().cpu().numpy()
    plt.imshow(class_val, cmap="inferno", aspect="auto")
    plt.colorbar(shrink=0.6)
    # Plot After Agg Predictions 3.
    plt.subplot(2,2,4)
    plt.title('Probs After Agg Softmax', fontsize=8)
    plt.xticks(fontsize=6, rotation=60)
    plt.yticks(fontsize=6)
    af_class_val = agg_out.detach().cpu().numpy()
    plt.imshow(af_class_val, cmap="inferno", aspect="auto")
    plt.colorbar(shrink=0.6)
    plt.tight_layout()
    plt.savefig(f"./SelfNonParametric_Proto_{nowtime}.png")

class PAWSNonParametric(nn.Module):
    '''at least one sample required per each class'''
    def __init__(self):
        super().__init__()
        self.softmax = nn.Softmax(dim=1)

    def forward(
        self,
        anchor_feat,
        positive_feat,
        lb_feat,
        lb_one_hot,
        logits_x_lb,
        logits_x_ulb_1,
        logits_x_ulb_2,
        args):
        tau = args.aggregator_tau


        lb_one_hot_1, lb_one_hot_2 = lb_one_hot.chunk(2)
        lb_feat_1, lb_feat_2 = lb_feat.chunk(2)

        support_k_1 = F.normalize(lb_feat_1)
        support_k_2 = F.normalize(lb_feat_2)

        anchor_q = F.normalize(anchor_feat)
        positive_q = F.normalize(positive_feat)

        logits_x_ulb_1 = self.softmax(anchor_q @ support_k_1.T / tau) @ lb_one_hot_1
        logits_x_ulb_2 = self.softmax(positive_q @ support_k_2.T / tau) @ lb_one_hot_2

        return (anchor_feat,
            positive_feat,
            lb_feat,
            lb_one_hot,
            logits_x_lb,
            logits_x_ulb_1,
            logits_x_ulb_2
            )

class Queue(nn.Module):
    def __init__(self, feature_dim, K):
        super().__init__()
        # create the queue
        self.K = K
        self.register_buffer("queue", torch.randn(feature_dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

    @torch.no_grad()
    def enqueue(self, keys):
        batch_size = keys.shape[0]

        self.queue[:, batch_size:] = self.queue[:, :self.K-batch_size]
        self.queue[:, :batch_size] = keys.T
    
    def forward(self):
        return self.queue.T

class QueuePAWSTransformer(nn.Module):
    def __init__(self, feature_dim=128, num_label=10, num_heads=4, K=4048, thres=0.95):
        super().__init__()
        self.thres = thres
        self.block = Block(feature_dim, v_dim=num_label, output_dim=num_label, num_heads=num_heads)

        self.feat_queue1 = Queue(feature_dim, K)
        self.feat_queue2 = Queue(feature_dim, K)
        self.label_queue1 = Queue(num_label, K)
        self.label_queue2 = Queue(num_label, K)
    
    def forward(
        self,
        anchor_feat,
        positive_feat,
        lb_feat,
        lb_one_hot,
        logits_x_lb,
        logits_x_ulb_1,
        logits_x_ulb_2,
        args):
        
        lb_feat_1, lb_feat_2 = lb_feat.chunk(2)
        lb_one_hot_1, lb_one_hot_2 = lb_one_hot.chunk(2)


        self.feat_queue1.enqueue(lb_feat_1)
        self.label_queue1.enqueue(lb_one_hot_1)

        ulb_1_max = logits_x_ulb_1.max(dim=1)[0] > self.thres
        self.feat_queue1.enqueue(anchor_feat[ulb_1_max])
        self.label_queue1.enqueue(logits_x_ulb_1[ulb_1_max])


        self.feat_queue2.enqueue(lb_feat_2)
        self.label_queue2.enqueue(lb_one_hot_2)

        ulb_2_max = logits_x_ulb_2.max(dim=1)[0] > self.thres
        self.feat_queue2.enqueue(positive_feat[ulb_2_max])
        self.label_queue2.enqueue(logits_x_ulb_2[ulb_2_max])

        logits_x_ulb_1 = self.block.forward_cross(
            anchor_feat[None, ...],
            self.feat_queue1()[None, ...],
            self.label_queue1()[None, ...],
        ).squeeze(0)

        logits_x_ulb_2 = self.block.forward_cross(
            positive_feat[None, ...],
            self.feat_queue2()[None, ...],
            self.label_queue2()[None, ...],
        ).squeeze(0)

        return (anchor_feat,
            positive_feat,
            lb_feat,
            lb_one_hot,
            logits_x_lb,
            logits_x_ulb_1,
            logits_x_ulb_2
            )
    

class PAWSTransformer(nn.Module):
    def __init__(self, feature_dim=128, num_label=10, num_heads=4):
        super().__init__()
        self.block = Block(feature_dim, v_dim=num_label, output_dim=num_label, num_heads=num_heads)

    def forward(
        self,
        anchor_feat,
        positive_feat,
        lb_feat,
        lb_one_hot,
        logits_x_lb,
        logits_x_ulb_1,
        logits_x_ulb_2,
        args):

        lb_feat_1, lb_feat_2 = lb_feat.chunk(2)
        lb_one_hot_1, lb_one_hot_2 = lb_one_hot.chunk(2)

        logits_x_ulb_1 = self.block.forward_cross(
            anchor_feat[None, ...],
            lb_feat_1[None, ...],
            lb_one_hot_1[None, ...],
        ).squeeze(0)

        logits_x_ulb_2 = self.block.forward_cross(
            positive_feat[None, ...],
            lb_feat_2[None, ...],
            lb_one_hot_2[None, ...],
        ).squeeze(0)
        
        return (anchor_feat,
            positive_feat,
            lb_feat,
            lb_one_hot,
            logits_x_lb,
            logits_x_ulb_1,
            logits_x_ulb_2
            )


class SelfNonParametric(nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax = nn.Softmax(dim=1)
    # Weak Aug만 Aggregation 통과
    def forward(
        self,
        anchor_feat,
        positive_feat,
        lb_feat,
        lb_one_hot,
        logits_x_lb,
        logits_x_ulb_1,
        logits_x_ulb_2,
        args):
        #define required parameters
        tau = args.aggregator_tau
        one_hot_class_label = args.one_hot_class_label
        softmax_class_label = args.softmax_class_label
        T = 1.0
        normed_lb_feat = F.normalize(lb_feat)
        anchor_q = F.normalize(anchor_feat)
        anchor_k = F.normalize(anchor_feat)
        positive_q = F.normalize(positive_feat)
        positive_k = F.normalize(positive_feat)

        #concat_part
        #experiment1. normalize -> concat
        # normalized_anchor_lb_ulb = torch.cat([normed_lb_feat, anchor_q])
        normalized_anchor_lb_ulb = F.normalize(torch.cat([lb_feat, anchor_feat, positive_feat]))

        logits_x_ulb = torch.cat([logits_x_ulb_1, logits_x_ulb_2])
        if softmax_class_label:
            #unlabeled_sharpening
            prob_xulb = torch.softmax(logits_x_ulb / T, dim=-1)
            prob_xlb = self.softmax(logits_x_lb)
            if one_hot_class_label:
                class_val = torch.cat([lb_one_hot, prob_xulb])
            else:
                class_val = torch.cat([prob_xlb, prob_xulb])
        else:
            class_val = torch.cat([logits_x_lb, logits_x_ulb])
        #experiment2. concat -> normalize
        # anchor_lb_ulb = torch.cat([lb_feat, anchor_feat])
        # anchor_lb_ulb = torch.cat([anchor_lb_ulb, positive_feat])
        # normalized_anchor_lb_ulb_2 = F.normalize(anchor_lb_ulb)
        logits_x_lb_ulb = self.softmax(normalized_anchor_lb_ulb @ normalized_anchor_lb_ulb.T / tau) @ class_val

        num_lb = logits_x_lb.shape[0]
        logits_x_lb = logits_x_lb_ulb[:num_lb]
        logits_x_ulb_1, logits_x_ulb_2 = logits_x_lb_ulb[num_lb:].chunk(2)
        # logits_x_ulb_1 = self.softmax(anchor_q @ anchor_k.T / tau) @ self.softmax(logits_x_ulb_1)
        # logits_x_ulb_2 = self.softmax(positive_q @ positive_k.T / tau) @ self.softmax(logits_x_ulb_2)


        return (anchor_feat,
            positive_feat,
            lb_feat,
            lb_one_hot,
            logits_x_lb,
            logits_x_ulb_1,
            logits_x_ulb_2
            )
#Now I'm working at
class SelfNonParametric_Prototype(nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax = nn.Softmax(dim=1)
    # Weak Aug만 Aggregation 통과
    # anchor = weak, positive = simclr
    def forward(
        self,
        weak_feat,    # Weak
        hard_feat,      # Hard
        lb_feat,
        lb_one_hot,
        logits_x_lb,
        logits_x_ulb_1, # Weak
        logits_x_ulb_2, # Hard
        y_lb,
        y_ulb,
        args):
        with torch.no_grad():
            #define required parameters
            tau = args.aggregator_tau
            # weak_feat = F.normalize(weak_feat, dim=1)
            # hard_feat = F.normalize(hard_feat, dim=1)
            # lb_feat = F.normalize(lb_feat, dim=1)
            # feat_all = torch.cat([lb_feat, weak_feat, hard_feat])

            #(to_do)Choice: feat_all can be weak_feat, weak_feat + strong_feat
            feat_all = weak_feat.detach()

            prototype_lb = torch.zeros((10, 128), device=f"cuda:{args.gpu}", dtype=lb_feat.dtype)
            prototype_ulb = torch.zeros((10, 128), device=f"cuda:{args.gpu}", dtype=lb_feat.dtype)
            
            # Class Num
            lb_class_num = torch.sum(lb_one_hot, dim=0).detach()
            ulb_class_num = torch.zeros(args.num_classes, dtype=torch.float32, device=f"cuda:{args.gpu}")

            # Labeled Prototype

            #index_add#
            '''
            dim (int) – dimension along which to index

            index (IntTensor or LongTensor) – indices of tensor to select from
            
            tensor (Tensor) – the tensor containing values to add
            '''
            prototype_lb = prototype_lb.index_add_(0, y_lb, lb_feat[0:len(y_lb)]).detach()
            #(sharpening will be needed?)

            # Thresholding UnLabeled
            max_val, max_idx_ulb = torch.max(self.softmax(logits_x_ulb_1), dim=1)
            mask = (max_val>args.p_cutoff)
            masked_weak_feat, masked_unlabeled_class_index = weak_feat[mask], max_idx_ulb[mask]
            
            # Unlabeled Prototype
            prototype_ulb = prototype_ulb.index_add_(0, masked_unlabeled_class_index, masked_weak_feat[0:len(masked_unlabeled_class_index)]).detach()
            ulb_class_num = ulb_class_num.index_add_(0, masked_unlabeled_class_index, torch.ones((len(masked_unlabeled_class_index)), device=f"cuda:{args.gpu}"))
            
            prototype = prototype_lb + prototype_ulb
            class_num = lb_class_num + ulb_class_num
            # # Normalize Prototype
            # for i in range(10):
            #     if not class_num[i] == 0:
            #         prototype[i] = torch.div(prototype[i], class_num[i])
            #(to_do) check prototype_check_0 == prototype
            prototype_check_0 = F.normalize(torch.div(prototype[0], class_num[0]))
            prototype = torch.div(prototype.T, class_num).T
            pdb.set_trace()
            prototype = F.normalize(prototype, dim=1)
            feat_all = F.normalize(feat_all, dim=1)

            atten_mat = self.softmax(feat_all @ prototype.T / tau)

            label_gt = torch.eye(10, dtype=torch.float32, device=f"cuda:{args.gpu}")
            # label_gt_smooth = torch.zeros_like(label_gt)
            # label_gt_smooth += 0.005
            # label_gt_smooth[range(10), range(10)] = label_gt.diagonal() - 0.05 + 0.005
            agg_out = atten_mat @ label_gt
            if args.vis:
                visualize(atten_mat=atten_mat, agg_out=agg_out, logits_x_ulb=logits_x_ulb_1, label_gt=label_gt)

                visualize(feat_all @ prototype.T / tau, #atten_mat
                          self.softmax(feat_all @ prototype.T / tau), #agg_out
                          self.softmax(logits_x_ulb_1), # logits_x_ulb_1
                          agg_out  # agg_out
                          )
            #fast_debugging
            # 0 < agg_out < 1 (약간 over)
            # probs_x_ulb_1_agg = self.softmax(agg_out)
            probs_x_ulb_1_agg = agg_out
            max_val_agg, max_idx_ulb_agg = torch.max(probs_x_ulb_1_agg, dim=1)        
            diff_list = []
            pairs = zip(max_idx_ulb, max_idx_ulb_agg)
            for i, pair in enumerate(pairs):
                if pair[0] != pair[1]:
                    diff_list.append([y_ulb[i],pair[0], max_val[i], pair[1],  max_val_agg[i]])
            # diff_list 보고 달라진 sample을 가장 잘 살릴수 있는 tau 구하기
        
            visualize(atten_mat=atten_mat, agg_out=agg_out, logits_x_ulb=logits_x_ulb_1, label_gt=label_gt)

            # logits_x_lb = agg_out[:len(lb_feat)]
            # logits_x_ulb_1, logits_x_ulb_2 = agg_out[len(lb_feat):].chunk(2)
            logits_x_ulb_1 = agg_out

            
            return (weak_feat,
                hard_feat,
                lb_feat,
                lb_one_hot,
                logits_x_lb,
                logits_x_ulb_1,
                logits_x_ulb_2
                )

class SelfNonParametric_Mod1(nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax = nn.Softmax(dim=1)
    # Weak Aug만 Aggregation 통과
    # anchor = weak, positive = simclr
    def forward(
        self,
        anchor_feat,
        positive_feat,
        lb_feat,
        lb_one_hot,
        logits_x_lb,
        logits_x_ulb_1, # Weak
        logits_x_ulb_2, # SimCLR
        y_lb,
        args):
        #define required parameters
        tau = args.aggregator_tau
        one_hot_class_label = args.one_hot_class_label
        softmax_class_label = args.softmax_class_label
        T = 1.0

        lb_feat_1 , lb_feat_2 = lb_feat.chunk(2)
        lb_one_hot_1, lb_one_hot_2 = lb_one_hot.chunk(2)
        logits_x_lb_1, logits_x_lb_2 = logits_x_lb.chunk(2)
        y_lb_1, y_lb_2 = y_lb.chunk(2)
        # 애매한 class prob 0으로
        # 애매한 Feature Sim도 0으로
        # Positive - Negative로 분리
        # Labeled - Unlabled 개수 맞추기
        # Weak aug Labeled P cutoff 안넘으면 0으로
        # Weak에 agg, Strong엔 x -> Weak에서 Simclr로 Fixmatchloss
        # High Sharpening - Target - Source Temp 차이
        logits_x_ulb = logits_x_ulb_1   # Weak Aug
        if softmax_class_label:
            #unlabeled_sharpening
            prob_xulb_1 = torch.softmax(logits_x_ulb / T, dim=-1)
            prob_xlb = self.softmax(logits_x_lb)
            prob_xulb_2 = self.softmax(logits_x_ulb_2 / T)
            if one_hot_class_label:
                class_val = torch.cat([lb_one_hot, prob_xulb_1])
            else:
                class_val = torch.cat([prob_xlb, prob_xulb_1])
        else:
            class_val = torch.cat([logits_x_lb, logits_x_ulb])

        # Normalizing Attention Map by Class Number
        p_cutoff = 0.95
        # Cutoff Prob
        prob_x_ulb_1 = F.softmax(logits_x_ulb_1 / 0.5, dim=1)
        prob_x_ulb_2 = F.softmax(logits_x_ulb_2 / 0.5, dim=1)
        # Compare by max_value
        max_probs_x_ulb_1, _ = torch.max(prob_x_ulb_1, dim=-1)
        max_probs_x_ulb_2, _ = torch.max(prob_x_ulb_2, dim=-1)
        logits_x_ulb_gt = torch.empty_like(logits_x_ulb_1)
        gt_idx_bool = torch.ge(max_probs_x_ulb_1, max_probs_x_ulb_2)
        logits_x_ulb_gt[gt_idx_bool] = logits_x_ulb_1[gt_idx_bool]
        logits_x_ulb_gt[~gt_idx_bool] = logits_x_ulb_2[~gt_idx_bool]
        mask = self.softmax(logits_x_ulb_gt).ge(p_cutoff).float()
        logits_x_ulb_gt = logits_x_ulb_gt * mask
        max_val, max_idx_ulb = torch.max(logits_x_ulb_gt, dim=1)
        zero_idx = torch.nonzero(max_val==0)
        max_idx_ulb[zero_idx] = -1
        class_num_ulb = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        for i in max_idx_ulb:
            if not i == -1:
                class_num_ulb[i] += 1
        class_num_ulb = torch.tensor(class_num_ulb, dtype=torch.float32, device=f"cuda:{args.gpu}")
        class_num_lb = lb_one_hot.sum(dim=0)
        class_num = class_num_ulb + class_num_lb

        ### Thresholding 대신 Prototype
        # Thresholding Logit (Probs)
        # Thresholding Attention Map
        self_atten = self.softmax(Q @ K.T / tau) 
        self_atten_threshold = self_atten[torch.ge(self_atten, 0.6)]
        self_atten_negative = self_atten[~torch.ge(self_atten, 0.6)]
        logits_x_lb_ulb = self_atten_threshold @ class_val


        num_lb = logits_x_lb.shape[0]
        logits_x_lb = logits_x_lb_ulb[:num_lb]
        logits_x_ulb_1 = logits_x_lb_ulb[num_lb:]
        
        # # Plot Attention Map 1
        # plt.subplot(1,3,1)
        # plt.title('Attention Maps', fontsize=8)
        # plt.xticks(fontsize=6, rotation=60)
        # plt.yticks(fontsize=6)
        # plt.imshow(self_atten.detach().cpu().numpy(), cmap="viridis", aspect="auto")
        # # Plot Before Agg Predictions 2.
        # plt.subplot(1,3,2)
        # plt.title('Probs Before Agg Softmax', fontsize=8)
        # plt.xticks(fontsize=6, rotation=60)
        # plt.yticks(fontsize=6)
        # class_val = torch.cat([class_val, prob_xulb_2]).detach().cpu().numpy()
        # plt.imshow(class_val, cmap="viridis", aspect="auto")
        # # Plot After Agg Predictions 3.
        # plt.subplot(1,3,3)
        # plt.title('Probs After Agg Softmax', fontsize=8)
        # plt.xticks(fontsize=6, rotation=60)
        # plt.yticks(fontsize=6)
        # af_class_val = torch.cat([logits_x_lb_ulb, prob_xulb_2]).detach().cpu().numpy()
        # plt.imshow(af_class_val, cmap="viridis", aspect="auto")
        # plt.tight_layout()
        # plt.savefig("./SelfNonParametric_Mod1.png")
        
        return (anchor_feat,
            positive_feat,
            lb_feat,
            lb_one_hot,
            logits_x_lb,
            logits_x_ulb_1,
            logits_x_ulb_2
            )


class SelfNonParametric_Vis(nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax = nn.Softmax(dim=1)

    def forward(
        self,
        anchor_feat,
        positive_feat,
        lb_feat,
        lb_one_hot,
        logits_x_lb,
        logits_x_ulb_1,
        logits_x_ulb_2,
        labels_lb, labels_ulb,
        args):
        #define required parameters
        tau = args.aggregator_tau
        one_hot_class_label = args.one_hot_class_label
        softmax_class_label = args.softmax_class_label
        T = 1.0
        normed_lb_feat = F.normalize(lb_feat)
        anchor_q = F.normalize(anchor_feat)
        anchor_k = F.normalize(anchor_feat)
        positive_q = F.normalize(positive_feat)
        positive_k = F.normalize(positive_feat)

        #concat_part
        #experiment1. normalize -> concat
        normalized_anchor_lb_ulb = torch.cat([normed_lb_feat, anchor_q, positive_q])

        
        logits_x_ulb = torch.cat([logits_x_ulb_1, logits_x_ulb_2])

        logits_before_agg = torch.cat([logits_x_lb, logits_x_ulb])
        max_logits_before_agg, pred_before_agg = logits_before_agg.topk(k=1, dim=1, largest=True, sorted=True)
        logits_after_agg = self.softmax(normalized_anchor_lb_ulb @ normalized_anchor_lb_ulb.T / tau) @ logits_before_agg
        max_logits_after_agg, pred_after_agg = logits_after_agg.topk(k=1, dim=1, largest=True, sorted=True)

        changed_pred_idx = torch.nonzero(pred_before_agg.squeeze() != pred_after_agg.squeeze(), as_tuple=False)    # 다르면 True
        diff_agg = torch.abs(self.softmax(max_logits_before_agg[changed_pred_idx]) - self.softmax(max_logits_after_agg[changed_pred_idx]))
        max_diff_idx = changed_pred_idx[diff_agg.argmax()]
        min_diff_idx = changed_pred_idx[diff_agg.argmin()]
        pred_max_diff_bf_agg = pred_before_agg[max_diff_idx]
        pred_max_diff_af_agg = pred_after_agg[max_diff_idx]
        pred_min_diff_bf_agg = pred_before_agg[min_diff_idx]
        pred_min_diff_af_agg = pred_after_agg[min_diff_idx]

        num_lb = logits_x_lb.shape[0]
        logits_x_lb = logits_after_agg[:num_lb]
        logits_x_ulb_1, logits_x_ulb_2 = logits_after_agg[num_lb:].chunk(2)
        # logits_x_ulb_1 = self.softmax(anchor_q @ anchor_k.T / tau) @ self.softmax(logits_x_ulb_1)
        # logits_x_ulb_2 = self.softmax(positive_q @ positive_k.T / tau) @ self.softmax(logits_x_ulb_2)

        return (anchor_feat,
            positive_feat,
            lb_feat,
            lb_one_hot,
            logits_x_lb,
            logits_x_ulb_1,
            logits_x_ulb_2,
            max_diff_idx, min_diff_idx, pred_max_diff_bf_agg, pred_max_diff_af_agg, pred_min_diff_bf_agg, pred_min_diff_af_agg
            )


class SelfTransformer(nn.Module):
    def __init__(self, feature_dim=128, num_label=10, num_heads=4):
        super().__init__()
        self.block = Block(feature_dim, v_dim=num_label, output_dim=num_label, num_heads=num_heads)

    def forward(
        self,
        anchor_feat,
        positive_feat,
        lb_feat,
        lb_one_hot,
        logits_x_lb,
        logits_x_ulb_1,
        logits_x_ulb_2,
        args):
        
        lb_feat_1, lb_feat_2 = lb_feat.chunk(2)
        lb_one_hot_1, lb_one_hot_2 = lb_one_hot.chunk(2)

        logits_x_ulb_1 = self.block(
            torch.cat((lb_feat_1, anchor_feat))[None, ...],
            torch.cat((lb_one_hot_1, logits_x_ulb_1))[None, ...]
        ).squeeze(0)

        logits_x_ulb_2 = self.block(
            torch.cat((lb_feat_2, positive_feat))[None, ...],
            torch.cat((lb_one_hot_2, logits_x_ulb_2))[None, ...]
        ).squeeze(0)

        return (anchor_feat,
            positive_feat,
            lb_feat,
            lb_one_hot,
            logits_x_lb,
            logits_x_ulb_1,
            logits_x_ulb_2
            )

class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        anchor_feat,
        positive_feat,
        lb_feat,
        lb_one_hot,
        logits_x_lb,
        logits_x_ulb_1,
        logits_x_ulb_2,
        args):
        
        return (anchor_feat,
            positive_feat,
            lb_feat,
            lb_one_hot,
            logits_x_lb,
            logits_x_ulb_1,
            logits_x_ulb_2
            )

class SelfNonParametric_Distil(nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax = nn.Softmax(dim=1)
    # Weak Aug만 Aggregation 통과
    # anchor = weak, positive = simclr
    def forward(
        self,
        anchor_feat,
        positive_feat,
        lb_feat,
        lb_one_hot,
        logits_x_lb,
        logits_x_ulb_1,
        logits_x_ulb_2,
        args):
        tau = args.aggregator_tau
        Q = F.normalize(logits_x_ulb_1, dim=1)
        K = F.normalize(logits_x_ulb_2, dim=2)

        atten = self.softmax(Q @ K.T / tau)
        one_hot_class_label = args.one_hot_class_label
        softmax_class_label = args.softmax_class_label
        T = 1.0
        normalized_anchor_lb = F.normalize(torch.cat([lb_feat, anchor_feat]))

        logits_x_ulb = logits_x_ulb_1   # Weak Aug
        if softmax_class_label:
            #unlabeled_sharpening
            prob_xulb_1 = torch.softmax(logits_x_ulb / T, dim=-1)
            prob_xlb = self.softmax(logits_x_lb)
            prob_xulb_2 = self.softmax(logits_x_ulb_2 / T)
            if one_hot_class_label:
                class_val = torch.cat([lb_one_hot, prob_xulb_1])
            else:
                class_val = torch.cat([prob_xlb, prob_xulb_1])
        else:
            class_val = torch.cat([logits_x_lb, logits_x_ulb])

        self_atten = self.softmax(normalized_anchor_lb @ normalized_anchor_lb.T / tau) 
        logits_x_lb_ulb = self_atten @ class_val
        num_lb = logits_x_lb.shape[0]
        logits_x_lb = logits_x_lb_ulb[:num_lb]
        logits_x_ulb_1 = logits_x_lb_ulb[num_lb:]
        
        # # Plot Attention Map 1
        # plt.subplot(1,3,1)
        # plt.title('Attention Maps', fontsize=8)
        # plt.xticks(fontsize=6, rotation=60)
        # plt.yticks(fontsize=6)
        # plt.imshow(self_atten.detach().cpu().numpy(), cmap="plasma", aspect="auto")
        # # Plot Before Agg Predictions 2.
        # plt.subplot(1,3,2)
        # plt.title('Probs Before Agg Softmax', fontsize=8)
        # plt.xticks(fontsize=6, rotation=60)
        # plt.yticks(fontsize=6)
        # class_val = torch.cat([class_val, prob_xulb_2]).detach().cpu().numpy()
        # plt.imshow(class_val, cmap="plasma", aspect="auto")
        # # Plot After Agg Predictions 3.
        # plt.subplot(1,3,3)
        # plt.title('Probs After Agg Softmax', fontsize=8)
        # plt.xticks(fontsize=6, rotation=60)
        # plt.yticks(fontsize=6)
        # af_class_val = torch.cat([logits_x_lb_ulb, prob_xulb_2]).detach().cpu().numpy()
        # plt.imshow(af_class_val, cmap="plasma", aspect="auto")
        # plt.tight_layout()
        # plt.savefig("./SelfNonParametric_Mod1.png")
        
        return (anchor_feat,
            positive_feat,
            lb_feat,
            lb_one_hot,
            logits_x_lb,
            logits_x_ulb_1,
            logits_x_ulb_2
            )