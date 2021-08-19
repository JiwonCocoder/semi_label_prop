import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.models as models
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
import matplotlib.pyplot as plt
import os
import contextlib

from .fixmatch_utils import Get_Scalar

from utils import ForkedPdb

import models.aggregators
import loss_module 


# default `log_dir` is "runs" - we'll be more specific here
writer = SummaryWriter()

def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))

class FixMatch:
    def __init__(self, net_builder, num_classes, ema_m, T, p_cutoff, lambda_u,\
                 hard_label=True, t_fn=None, p_fn=None, it=0, num_eval_iter=1000, tb_log=None, 
                 logger=None, loss=None, aggregator_module=None):
        """
        class Fixmatch contains setter of data_loader, optimizer, and model update methods.
        Args:
            net_builder: backbone network class (see net_builder in utils.py)
            num_classes: # of label classes 
            ema_m: momentum of exponential moving average for eval_model
            T: Temperature scaling parameter for output sharpening (only when hard_label = False)
            p_cutoff: confidence cutoff parameters for loss masking
            lambda_u: ratio of unsupervised loss to supervised loss
            hard_label: If True, consistency regularization use a hard pseudo label.
            it: initial iteration count
            num_eval_iter: freqeuncy of iteration (after 500,000 iters)
            tb_log: tensorboard writer (see train_utils.py)
            logger: logger (see utils.py)
        """
        
        super(FixMatch, self).__init__()

        # momentum update param
        self.loader = {}
        self.num_classes = num_classes
        self.ema_m = ema_m

        self.aggregator = getattr(models.aggregators, aggregator_module, None)()
        if self.aggregator is None:
            raise NotImplementedError()

        self.loss = getattr(loss_module, loss, None)()
        if self.loss is None:
            raise NotImplementedError()

        # create the encoders
        # network is builded only by num_classes,
        # other configs are covered in main.py
        
        self.train_model = net_builder(num_classes=num_classes) 
        self.eval_model = net_builder(num_classes=num_classes)
        self.num_eval_iter = num_eval_iter
        self.t_fn = Get_Scalar(T) #temperature params function
        self.p_fn = Get_Scalar(p_cutoff) #confidence cutoff function
        self.lambda_u = lambda_u
        self.tb_log = tb_log
        self.use_hard_label = hard_label
        
        self.optimizer = None
        self.scheduler = None
        
        self.it = 0
        
        self.logger = logger
        self.print_fn = print if logger is None else logger.info
        
        for param_q, param_k in zip(self.train_model.parameters(), self.eval_model.parameters()):
            param_k.data.copy_(param_q.detach().data)  # initialize
            param_k.requires_grad = False  # not update by gradient for eval_net
            
        self.eval_model.eval()
            
            
    @torch.no_grad()
    def _eval_model_update(self):
        """
        Momentum update of evaluation model (exponential moving average)
        """
        for param_train, param_eval in zip(self.train_model.parameters(), self.eval_model.parameters()):
            param_eval.copy_(param_eval * self.ema_m + param_train.detach() * (1-self.ema_m))
        
        for buffer_train, buffer_eval in zip(self.train_model.buffers(), self.eval_model.buffers()):
            buffer_eval.copy_(buffer_train)            
    
     
    def set_data_loader(self, loader_dict):
        self.loader_dict = loader_dict
        self.print_fn(f'[!] data loader keys: {self.loader_dict.keys()}')    
            
    
    def set_optimizer(self, optimizer, scheduler=None):
        self.optimizer = optimizer
        self.scheduler = scheduler

    def pretrain(self, args, logger=None):
        args.num_pretrain_iter = 0
        """
        Train function of FixMatch.
        From data_loader, it inference training data, computes losses, and update the networks.
        """
        ngpus_per_node = torch.cuda.device_count()

        # lb: labeled, ulb: unlabeled
        self.train_model.train()

        # for gpu profiling
        start_batch = torch.cuda.Event(enable_timing=True)
        end_batch = torch.cuda.Event(enable_timing=True)
        start_run = torch.cuda.Event(enable_timing=True)
        end_run = torch.cuda.Event(enable_timing=True)

        start_batch.record()
        best_eval_acc, best_it = 0.0, 0

        scaler = GradScaler()
        amp_cm = autocast if args.amp else contextlib.nullcontext
        for (x_lb_1, _, y_lb) in self.loader_dict['train_lb']:

            # prevent the training iterations exceed args.num_train_iter
            if self.it > args.num_pretrain_iter:
                break

            end_batch.record()
            torch.cuda.synchronize()
            start_run.record()

            # inference and calculate sup/unsup losses
            with amp_cm():
                logits_x_lb, feat = self.train_model(x_lb_1.cuda(args.gpu), ood_test=True, simsiam=args.simsiam)

                # hyper-params for update
                T = self.t_fn(self.it)
                p_cutoff = self.p_fn(self.it)

                sup_loss = loss_module.ce_loss(logits_x_lb, y_lb.cuda(args.gpu), reduction='mean')

                total_loss = sup_loss

            # parameter updates
            if args.amp:
                scaler.scale(total_loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
            else:
                total_loss.backward()
                self.optimizer.step()

            self.scheduler.step()
            self.train_model.zero_grad()

            with torch.no_grad():
                self._eval_model_update()

            end_run.record()
            torch.cuda.synchronize()

            # tensorboard_dict update
            tb_dict = {}

            tb_dict['train/sup_loss'] = sup_loss.detach()
            tb_dict['train/unsup_loss'] = torch.tensor(0.0, device=args.gpu)
            tb_dict['train/total_loss'] = torch.tensor(0.0, device=args.gpu)
            tb_dict['train/mask_ratio'] = torch.tensor(0.0, device=args.gpu)
            tb_dict['lr'] = self.optimizer.param_groups[0]['lr']
            tb_dict['train/prefecth_time'] = start_batch.elapsed_time(end_batch) / 1000.
            tb_dict['train/run_time'] = start_run.elapsed_time(end_run) / 1000.

            if self.it % self.num_eval_iter == 0:
                eval_dict = self.evaluate(args=args)
                tb_dict.update(eval_dict)

                save_path = os.path.join(args.save_dir, args.save_name)

                if tb_dict['eval/top-1-acc'] > best_eval_acc:
                    best_eval_acc = tb_dict['eval/top-1-acc']
                    best_it = self.it

                self.print_fn(
                    f"{self.it} iteration, USE_EMA: {hasattr(self, 'eval_model')}, {tb_dict}, BEST_EVAL_ACC: {best_eval_acc}, at {best_it} iters")

            if not args.multiprocessing_distributed or \
                    (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):

                if self.it == best_it:
                    self.save_model('model_best.pth', save_path)

                if not self.tb_log is None:
                    self.tb_log.update(tb_dict, self.it)

            self.it += 1
            del tb_dict
            start_batch.record()
            if self.it > 2 ** 19:
                self.num_eval_iter = 1000

        eval_dict = self.evaluate(args=args)
        eval_dict.update({'eval/best_acc': best_eval_acc, 'eval/best_it': best_it})
        return eval_dict

    def train(self, args, logger=None):
        """
        Train function of FixMatch.
        From data_loader, it inference training data, computes losses, and update the networks.
        """
        ngpus_per_node = torch.cuda.device_count()

        #lb: labeled, ulb: unlabeled
        self.train_model.train()

        # for gpu profiling
        start_batch = torch.cuda.Event(enable_timing=True)
        end_batch = torch.cuda.Event(enable_timing=True)
        start_run = torch.cuda.Event(enable_timing=True)
        end_run = torch.cuda.Event(enable_timing=True)
        
        start_batch.record()
        best_eval_acc, best_it = 0.0, 0
        
        scaler = GradScaler()
        amp_cm = autocast if args.amp else contextlib.nullcontext

        for (x_lb_1,x_lb_2, y_lb), (x_ulb_1, x_ulb_2, y_ulb) in zip(self.loader_dict['train_lb'], self.loader_dict['train_ulb']):

            # prevent the training iterations exceed args.num_train_iter
            if self.it > args.num_train_iter:
                break
            
            end_batch.record()
            torch.cuda.synchronize()
            start_run.record()
            num_lb = x_lb_1.shape[0] + x_lb_2.shape[0]
            num_ulb = x_ulb_1.shape[0] + x_ulb_2.shape[0]
            assert x_lb_1.shape[0] == x_lb_2.shape[0]
            assert x_ulb_1.shape[0] == x_ulb_2.shape[0]

            
            x_lb_1, x_lb_2, x_ulb_1, x_ulb_2 = x_lb_1.cuda(args.gpu), x_lb_2.cuda(args.gpu), x_ulb_1.cuda(args.gpu), x_ulb_2.cuda(args.gpu)
            y_lb = y_lb.cuda(args.gpu)
            y_lb_total = y_lb.unsqueeze(0).repeat(1,2).squeeze(0)

            #(bl *2) + (bul*2) = 8 + 56 = 64
            inputs = torch.cat((x_lb_1, x_lb_2, x_ulb_1, x_ulb_2))
            
            # inference and calculate sup/unsup losses
            with amp_cm():
                logits, feat = self.train_model(inputs, ood_test=True, simsiam=args.simsiam)

                logits_x_lb = logits[:num_lb]
                logits_x_ulb_1, logits_x_ulb_2 = logits[num_lb:].chunk(2)

                # if args.simsiam:
                #     pred_feat = feat[1]
                #     feat = feat[0]
                #(else)
                #we can use feat = feat
                lb_feat = feat[:num_lb]
                lb_one_hot = torch.eye(self.num_classes).cuda(args.gpu)[y_lb_total]
                anchor_feat, positive_feat = feat[num_lb:].chunk(2)

                # if args.simsiam:
                #     anchor_pred_feat, positive_pred_feat = pred_feat[num_lb:].chunk(2)
                # else:
                #     anchor_pred_feat, positive_pred_feat = None, None
                del logits

                # hyper-params for update
                tb_dict = {}
                T = self.t_fn(self.it)
                p_cutoff = self.p_fn(self.it)
                tb_dict['train/T'] = T.detach()
                tb_dict['train/p_cutoff'] = p_cutoff.detach()
                sup_loss = loss_module.ce_loss(logits_x_lb, y_lb_total, reduction='mean')
                # print("args.softmax_class_label:", args.softmax_class_label)
                # print("args.one_hot_class_label", args.one_hot_class_label)
                # print("args.hard_label:", args.hard_label)
                # print("args.simsiam:", args.simsiam)
                anchor_feat_agg, positive_feat_agg, lb_feat_agg, lb_one_hot_agg, logits_x_lb_agg, logits_x_ulb_1_agg, logits_x_ulb_2_agg =\
                    self.aggregator(
                        anchor_feat,
                        positive_feat,
                        lb_feat,
                        lb_one_hot,
                        logits_x_lb,
                        logits_x_ulb_1,
                        logits_x_ulb_2,
                        y_lb,
                        y_ulb,
                        args
                    )
                
                if isinstance(self.loss, loss_module.BidirectionalLoss):
                    unsup_loss, mask = self.loss(logits_x_ulb_1_agg, logits_x_ulb_2_agg, T, p_cutoff, use_hard_labels=args.hard_label)
                elif isinstance(self.loss, loss_module.PAWSLoss):
                    unsup_loss, mask = self.loss(anchor_feat, positive_feat, lb_feat, lb_one_hot)
                elif isinstance(self.loss, loss_module.FixMatchLoss):
                    # 1. Pows Loss만 사용 (Sup x, thres x, fixloss x)
                    # 2. 
                    unsup_loss_1, mask_1 = loss_module.consistency_loss(logits_x_ulb_1, 
                        logits_x_ulb_2, 
                        'ce', T, args.p_cutoff_fix,
                        use_hard_labels=args.hard_label
                    )
                    unsup_loss_2, mask_2 = loss_module.consistency_loss_wo_softmax(logits_x_ulb_1_agg,    # POWS Loss
                       logits_x_ulb_2, 
                        'ce', T, args.p_cutoff_paws,
                        use_hard_labels=args.hard_label
                    )

                    unsup_loss = unsup_loss_1 + unsup_loss_2
                    tb_dict['train/unsup_fix_loss'] = unsup_loss_1.detach() 
                    tb_dict['train/unsup_pows_loss'] = unsup_loss_2.detach() 
                    tb_dict['train/mask_fixloss_ratio'] = 1.0 - mask_1.detach() 
                    tb_dict['train/mask_powsloss_ratio'] = 1.0 - mask_2.detach() 
                    mask = (mask_1 + mask_2) / 2

                elif isinstance(self.loss, loss_module.BidirectionalLoss_prob):
                    unsup_loss, mask = self.loss(logits_x_ulb_1_agg, logits_x_ulb_2_agg, T, p_cutoff, use_hard_labels=args.hard_label)
                else:
                    raise NotImplementedError()


                total_loss = sup_loss + self.lambda_u * unsup_loss
            
            # parameter updates
            if args.amp:
                scaler.scale(total_loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
            else:
                total_loss.backward()
                self.optimizer.step()
                
            self.scheduler.step()
            self.train_model.zero_grad()
            
            with torch.no_grad():
                self._eval_model_update()
            
            end_run.record()
            torch.cuda.synchronize()
            
            #tensorboard_dict update
            tb_dict['train/sup_loss'] = sup_loss.detach()
            tb_dict['train/unsup_loss'] = unsup_loss.detach() 
            tb_dict['train/total_loss'] = total_loss.detach() 
            tb_dict['train/mask_ratio'] = 1.0 - mask.detach() 
            tb_dict['lr'] = self.optimizer.param_groups[0]['lr']
            tb_dict['lr_aggregator'] = self.optimizer.param_groups[2]['lr']
            tb_dict['train/prefecth_time'] = start_batch.elapsed_time(end_batch)/1000.
            tb_dict['train/run_time'] = start_run.elapsed_time(end_run)/1000.
            
            
            if self.it % self.num_eval_iter == 0:
                eval_dict = self.evaluate(args=args)
                tb_dict.update(eval_dict)
                
                save_path = os.path.join(args.save_dir, args.save_name)
                
                if tb_dict['eval/top-1-acc'] > best_eval_acc:
                    best_eval_acc = tb_dict['eval/top-1-acc']
                    best_it = self.it
                
                self.print_fn(f"{self.it} iteration\tUSE_EMA: {hasattr(self, 'eval_model')}\n{tb_dict}\nBEST_EVAL_ACC: {best_eval_acc}, at {best_it} iters")
                for k, v in tb_dict.items():
                    print(f"{k}:\t{v}")
                print(f"{self.it} iteration\tUSE_EMA: {hasattr(self, 'eval_model')}\tBEST_EVAL_ACC: {best_eval_acc}, at {best_it} iters")
            if not args.multiprocessing_distributed or \
                    (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
                
                if self.it == best_it:
                    self.save_model('model_best.pth', save_path)
                
                if not self.tb_log is None:
                    self.tb_log.update(tb_dict, self.it)
                
            self.it +=1
            del tb_dict
            start_batch.record()
            if self.it > 2**19:
                self.num_eval_iter = 1000
        
        eval_dict = self.evaluate(args=args)
        eval_dict.update({'eval/best_acc': best_eval_acc, 'eval/best_it': best_it})
        return eval_dict
            
            
    @torch.no_grad()
    def evaluate(self, eval_loader=None, args=None):
        use_ema = hasattr(self, 'eval_model')
        
        eval_model = self.eval_model if use_ema else self.train_model
        eval_model.eval()
        if eval_loader is None:
            eval_loader = self.loader_dict['eval']
        
        total_loss = 0.0
        total_acc = 0.0
        total_num = 0.0
        for x, y in eval_loader:
            x, y = x.cuda(args.gpu), y.cuda(args.gpu)
            num_batch = x.shape[0]
            total_num += num_batch
            logits, _ = eval_model(x)
            loss = F.cross_entropy(logits, y, reduction='mean')
            acc = torch.sum(torch.max(logits, dim=-1)[1] == y)
            
            total_loss += loss.detach()*num_batch
            total_acc += acc.detach()
        
        if not use_ema:
            eval_model.train()
            
        return {'eval/loss': total_loss/total_num, 'eval/top-1-acc': total_acc/total_num}
    
    
    def save_model(self, save_name, save_path):
        save_filename = os.path.join(save_path, save_name)
        train_model = self.train_model.module if hasattr(self.train_model, 'module') else self.train_model
        eval_model = self.eval_model.module if hasattr(self.eval_model, 'module') else self.eval_model
        torch.save({'train_model': train_model.state_dict(),
                    'eval_model': eval_model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'scheduler': self.scheduler.state_dict(),
                    'it': self.it}, save_filename)
        
        self.print_fn(f"model saved: {save_filename}")
    
    
    def load_model(self, load_path, gpu):
        checkpoint = torch.load(load_path)
        
        train_model = self.train_model.module if hasattr(self.train_model, 'module') else self.train_model
        eval_model = self.eval_model.module if hasattr(self.eval_model, 'module') else self.eval_model
        
        for key in checkpoint.keys():
            if hasattr(self, key) and getattr(self, key) is not None:
                if 'train_model' in key:
                    train_model.load_state_dict(checkpoint[key])
                    print("loaded_train_model's fc_param[0]")
                    print(list(train_model.fc.named_parameters())[0])
                    train_model.fc = nn.Linear(train_model.fc.in_features, 10).cuda(gpu)
                    print("init_train_model's fc_param[0]")
                    print(list(train_model.fc.named_parameters())[0])
                    
                elif 'eval_model' in key:
                    eval_model.load_state_dict(checkpoint[key])
                    print("loaded_eval_model's fc_param[0]")
                    eval_model.fc = nn.Linear(eval_model.fc.in_features, 10).cuda(gpu)
                    print("init_eval_model's fc_param[0]")
                    print(list(eval_model.fc.named_parameters())[0])
                elif key == 'it':
                    self.it = checkpoint[key]
                    self.print_fn(f"{key} {self.it} is LOADED")
                    if self.it > 100000:
                        self.it = 100
                elif key == 'scheduler':
                    self.scheduler.load_state_dict(checkpoint[key])
                elif key == 'optimizer':
                    self.optimizer.load_state_dict(checkpoint[key]) 
                else:
                    getattr(self, key).load_state_dict(checkpoint[key])
                self.print_fn(f"Check Point Loading: {key} is LOADED")
            else:
                self.print_fn(f"Check Point Loading: {key} is **NOT** LOADED")

if __name__ == "__main__":
    pass