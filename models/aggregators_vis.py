import pdb
from typing import Sequence

from matplotlib import cm
from utils import ForkedPdb
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from utils import ForkedPdb
# future work : add taus argument, label smoothing require

from models.transformer import Block


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

        class_num_lb = lb_one_hot.sum(dim=0)
        logits_x_ulb_gt = logits_x_ulb_gt * mask
        max_val, max_idx_ulb = torch.max(logits_x_ulb_gt, dim=1)
        zero_idx = torch.nonzero(max_val==0)
        max_idx_ulb[zero_idx] = -1
        class_num_ulb = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        for i in max_idx_ulb:
            if not i == -1:
                class_num_ulb[i] += 1
        class_num_ulb = torch.tensor(class_num_ulb, dtype=torch.float32, device=f"cuda:{args.gpu}") * 2
        class_num = class_num_ulb + class_num_lb
        
        logits_x_ulb = torch.cat([logits_x_ulb_gt, logits_x_ulb_gt])

        prob_xulb = torch.softmax(logits_x_ulb / 0.5, dim=-1)

        logits_bf_agg_onehot = torch.cat([lb_one_hot, prob_xulb])
        #experiment2. concat -> normalize
        # anchor_lb_ulb = torch.cat([lb_feat, anchor_feat])
        # anchor_lb_ulb = torch.cat([anchor_lb_ulb, positive_feat])
        # normalized_anchor_lb_ulb_2 = F.normalize(anchor_lb_ulb)

        # TODO: Visualize Attention Map
        # _, logits_pred = class_val.topk(k=1, dim=1, largest=True, sorted=True)
        normalized_anchor_lb_ulb = F.normalize(torch.cat([lb_feat, anchor_feat, positive_feat]))
        self_atten = self.softmax(normalized_anchor_lb_ulb @ F.normalize(normalized_anchor_lb_ulb, dim=1).T / 0.5) @ logits_bf_agg_onehot
        logits_af_agg_onehot = torch.empty_like(logits_bf_agg_onehot)
        for i in range(10):
            if not class_num[i] == 0:
                logits_af_agg_onehot[:,i] = torch.div(self_atten[:,i], class_num[i])
            else:
                logits_af_agg_onehot[:,i] = torch.div(self_atten[:,i], class_num.sum())
        # atten_mat = self.softmax(normalized_anchor_lb_ulb @ normalized_anchor_lb_ulb.T).detach().cpu().numpy()
        # atten_mat = atten_mat / atten_mat.max()
        # pred_mat = (logits_pred == logits_pred.T).detach().cpu().numpy()
        # agg_pred_mat = (agg_logits_pred == agg_logits_pred.T).detach().cpu().numpy()

        # diff_agg_mat = (logits_pred != agg_logits_pred).detach().cpu().numpy()

        # num_lb = logits_x_lb.shape[0]
        # num_ulb = logits_x_ulb_1.shape[0]
        # x1 = [0, 64]
        # y1 = [num_lb, num_lb]
        # x2 = [num_lb, num_lb]
        # y2 = [0, 64]
        # x3 = [0, 64]
        # y3 = [num_lb + num_ulb - 1, num_lb + num_ulb - 1]
        # x4 = [num_lb + num_ulb - 1, num_lb + num_ulb - 1]
        # y4 = [0, 64]

        # plt.subplot(2,2,1)
        # plt.plot(x1, y1, color="red", linewidth=1)
        # plt.plot(x2, y2, color="red", linewidth=1)
        # plt.plot(x3, y3, color="red", linewidth=1)
        # plt.plot(x4, y4, color="red", linewidth=1)
        # plt.imshow(atten_mat, cmap="gray")
        # plt.subplot(2,2,2)
        # plt.plot(x1, y1, color="red", linewidth=1)
        # plt.plot(x2, y2, color="red", linewidth=1)
        # plt.plot(x3, y3, color="red", linewidth=1)
        # plt.plot(x4, y4, color="red", linewidth=1)
        # plt.imshow(pred_mat, cmap="gray")
        # plt.subplot(2,2,3)
        # plt.imshow(diff_agg_mat, cmap="gray", aspect="auto")
        # plt.subplot(2,2,4)
        # plt.imshow(agg_pred_mat, cmap="gray")
        # plt.savefig("./map.png")

        # Visualize Logit Map
        prob_bf_agg_onehot = self.softmax(logits_bf_agg_onehot).detach().cpu().numpy()
        plt.subplot(1,2,1)
        plt.title('Logits before Agg Onehot', fontsize=8)
        plt.xticks(fontsize=6, rotation=60)
        plt.yticks(fontsize=6)
        plt.imshow(prob_bf_agg_onehot, cmap="viridis", aspect="auto")

        # af_agg_pseudo_label_idx = logits_af_agg_onehot.argmax(dim=1)
        # prob_af_agg_onehot = torch.eye(10).cuda(args.gpu)[af_agg_pseudo_label_idx.unsqueeze(0).repeat(1,1).squeeze(0)].detach().cpu().numpy()
        prob_af_agg_onehot = self.softmax(logits_af_agg_onehot / 0.001).detach().cpu().numpy()
        plt.subplot(1,2,2)
        plt.title('Logits after Agg Onehot', fontsize=8)
        plt.xticks(fontsize=6, rotation=60)
        plt.yticks(fontsize=6)
        plt.imshow(prob_af_agg_onehot, cmap="viridis", aspect="auto")

        # prob_bf_agg_softmax = self.softmax(logits_bf_agg_softmax / 0.1).detach().cpu().numpy()
        # plt.subplot(2,2,3)
        # plt.title('Logits before Agg Softmax', fontsize=8)
        # plt.xticks(fontsize=6, rotation=60)
        # plt.yticks(fontsize=6)
        # plt.imshow(prob_bf_agg_softmax, cmap="viridis", aspect="auto")

        # prob_af_agg_softmax = self.softmax(logits_af_agg_softmax / 0.1).detach().cpu().numpy()
        # plt.subplot(2,2,4)
        # plt.title('Logits after Agg Softmax', fontsize=8)
        # plt.xticks(fontsize=6, rotation=60)
        # plt.yticks(fontsize=6)
        # plt.imshow(prob_af_agg_softmax, cmap="viridis", aspect="auto")
        # plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
        # cax = plt.axes([0.85, 0.1, 0.075, 0.8])
        # plt.colorbar(cax=cax)
        plt.savefig("./logit_prob_map_train_co_atten.png")

        # num_lb = logits_x_lb.shape[0]
        # logits_x_lb = logits_x_lb_ulb[:num_lb]
        # logits_x_ulb_1, logits_x_ulb_2 = logits_x_lb_ulb[num_lb:].chunk(2)

        
        return (anchor_feat,
            positive_feat,
            lb_feat,
            lb_one_hot,
            logits_x_lb,
            logits_x_ulb_1,
            logits_x_ulb_2
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

class SelfNonParametricEval(nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, feats, logits, labels, args):
        #define required parameters
        tau = args.aggregator_tau
        T = 1.0
        normalized_feat = F.normalize(feats)
        # Visualize Attention Map
        _, logits_pred = logits.topk(k=1, dim=1, largest=True, sorted=True)
        atten_mat = self.softmax(normalized_feat @ normalized_feat.T)
        logits_agg = atten_mat @ self.softmax(logits)
        _, agg_logits_pred = logits_agg.topk(k=1, dim=1, largest=True, sorted=True)

        label_mat = (labels.unsqueeze(dim=1) == labels.unsqueeze(dim=0)).detach().cpu().numpy()
        atten_mat = atten_mat.detach().cpu().numpy()
        pred_mat = (logits_pred == logits_pred.T).detach().cpu().numpy().astype("float32")
        agg_pred_mat = (agg_logits_pred == agg_logits_pred.T).detach().cpu().numpy().astype("float32")

        diff_label_agg_mat = (label_mat != agg_pred_mat).astype("float32")
        diff_label_pred_mat = (label_mat != pred_mat).astype("float32")

        plt.subplot(2,3,1)
        plt.title('Affinity Map', fontsize=8)
        plt.xticks(fontsize =6)
        plt.imshow(atten_mat, cmap="viridis")
        plt.subplot(2,3,2)
        plt.title('Prediction Map', fontsize=8)
        plt.xticks(fontsize =6)
        plt.imshow(pred_mat, cmap="viridis")
        plt.subplot(2,3,6)
        plt.title('Diff Map Between\nLabel Map and Agg Pred Map', fontsize=8)
        plt.xticks(fontsize =6)
        plt.imshow(diff_label_agg_mat, cmap="viridis")
        plt.subplot(2,3,3)
        plt.title('Aggregated Prediction Map', fontsize=8)
        plt.xticks(fontsize =6)
        plt.imshow(agg_pred_mat, cmap="viridis")
        plt.subplot(2,3,4)
        plt.title('Label Map', fontsize=8)
        plt.xticks(fontsize =6)
        plt.imshow(label_mat, cmap="viridis")
        plt.subplot(2,3,5)
        plt.title('Diff Map Between\nLabel Map and Pred Map', fontsize=8)
        plt.xticks(fontsize =6)
        plt.imshow(diff_label_pred_mat, cmap="viridis")
        plt.savefig("./Affinity Map.png")
        # prob_bf_agg = self.softmax(logits / 0.1).detach().cpu().numpy()
        # plt.subplot(2,1,1)
        # plt.title('Logits before Agg', fontsize=8)
        # plt.xticks(fontsize=6, rotation=60)
        # plt.yticks(fontsize=6)
        # plt.imshow(prob_bf_agg, cmap="viridis", aspect=0.2)

        # prob_af_agg = self.softmax(logits_agg / 0.1).detach().cpu().numpy()
        # plt.subplot(2,1,2)
        # plt.title('Logits after Agg', fontsize=8)
        # plt.xticks(fontsize=6, rotation=60)
        # plt.yticks(fontsize=6)
        # plt.imshow(prob_af_agg, cmap="viridis", aspect=0.2)

        # plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
        # cax = plt.axes([0.85, 0.1, 0.075, 0.8])
        # plt.colorbar(cax=cax)
        # plt.savefig("./logit_prob_map_end.png")

        return (feats, logits, logits_agg, labels)

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
        # max_diff_idx = changed_pred_idx[diff_agg.argmax() if not diff_agg.size==0 else ]
        # min_diff_idx = changed_pred_idx[diff_agg.argmin()]
        # pred_max_diff_bf_agg = pred_before_agg[max_diff_idx]
        # pred_max_diff_af_agg = pred_after_agg[max_diff_idx]
        # pred_min_diff_bf_agg = pred_before_agg[min_diff_idx]
        # pred_min_diff_af_agg = pred_after_agg[min_diff_idx]

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
            logits_x_ulb_2
            )