#import needed library
# weak-weak
# weak-strong
# simclr-simclr
# x
# class logit: 
import os
import logging
import random
import warnings
import json
import yaml

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp

from utils import net_builder, get_logger, count_parameters
from train_utils import TBLog, get_SGD, get_cosine_schedule_with_warmup
from models.fixmatch.fixmatch import FixMatch
from datasets.ssl_dataset import SSL_Dataset
from datasets.data_utils import get_data_loader

from utils import ForkedPdb
import time
import models.aggregators_vis
import loss_module

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

        self.aggregator = getattr(models.aggregators_vis, aggregator_module, None)()
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

    def set_data_loader(self, loader_dict):
        self.loader_dict = loader_dict
        self.print_fn(f'[!] data loader keys: {self.loader_dict.keys()}')    

    def load_model(self, load_path):
        train_model = self.train_model.module if hasattr(self.train_model, 'module') else self.train_model
        eval_model = self.eval_model.module if hasattr(self.eval_model, 'module') else self.eval_model
        checkpoint = torch.load(load_path)

        for key in checkpoint.keys():
            if hasattr(self, key) and getattr(self, key) is not None:
                if 'train_model' in key:
                    train_model.load_state_dict(checkpoint[key])
                elif 'eval_model' in key:
                    eval_model.load_state_dict(checkpoint[key])
                elif key == 'it':
                    self.it = checkpoint[key]
                elif key == 'scheduler':
                    self.scheduler.load_state_dict(checkpoint[key])
                elif key == 'optimizer':
                    self.optimizer.load_state_dict(checkpoint[key]) 
                else:
                    getattr(self, key).load_state_dict(checkpoint[key])
                self.print_fn(f"Check Point Loading: {key} is LOADED")
            else:
                self.print_fn(f"Check Point Loading: {key} is **NOT** LOADED")
    @torch.no_grad()
    def train(self, args, logger=None):
        #lb: labeled, ulb: unlabeled
        self.train_model.eval()
        best_eval_acc, best_it = 0.0, 0

        for (x_lb_1,x_lb_2, y_lb), (x_ulb_1, x_ulb_2, y_ulb) in zip(self.loader_dict['train_lb'], self.loader_dict['train_ulb']):

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
            logits, feat = self.train_model(inputs, ood_test=True, simsiam=args.simsiam)

            logits_x_lb = logits[:num_lb]
            logits_x_ulb_1, logits_x_ulb_2 = logits[num_lb:].chunk(2)

            if args.simsiam:
                pred_feat = feat[1]
                feat = feat[0]
            #(else)
            #we can use feat = feat
            lb_feat = feat[:num_lb]
            lb_one_hot = torch.eye(self.num_classes).cuda(args.gpu)[y_lb_total]
            anchor_feat, positive_feat = feat[num_lb:].chunk(2)

            if args.simsiam:
                anchor_pred_feat, positive_pred_feat = pred_feat[num_lb:].chunk(2)
            else:
                anchor_pred_feat, positive_pred_feat = None, None
            del logits

            sup_loss = loss_module.ce_loss(logits_x_lb, y_lb_total, reduction='mean')
            anchor_feat_agg, positive_feat_agg, lb_feat_agg, lb_one_hot, logits_x_lb_agg, logits_x_ulb_1_agg, logits_x_ulb_2_agg =\
                self.aggregator(
                    anchor_feat,
                    positive_feat,
                    lb_feat,
                    lb_one_hot,
                    logits_x_lb,
                    logits_x_ulb_1,
                    logits_x_ulb_2,
                    y_lb, y_ulb,
                    args
                )

            tb_dict = {}
            
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
        acc_dict = {}
        for x, y in eval_loader:
            x, y = x.cuda(args.gpu), y.cuda(args.gpu)
            num_batch = x.shape[0]
            total_num += num_batch
            logits, feats = eval_model(x)
            self.aggregator(feats, logits, y, args)
            loss = F.cross_entropy(logits, y, reduction='mean')
            acc = torch.sum(torch.max(logits, dim=-1)[1] == y)
            total_loss += loss.detach()*num_batch
            total_acc += acc.detach()
        
        if not use_ema:
            eval_model.train()
            
        return {'eval/loss': total_loss/total_num, 'eval/top-1-acc': total_acc/total_num}
    
    def set_data_loader(self, loader_dict):
        self.loader_dict = loader_dict
        self.print_fn(f'[!] data loader keys: {self.loader_dict.keys()}')    

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--save_dir', type=str, default='./saved_models')
    parser.add_argument('--save_name', type=str, default=time.strftime('%Y_%m_%d_%H_%M'))
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--load_path', type=str, default=None)
    parser.add_argument('--overwrite', action='store_true')
    
    '''
    Training Configuration of FixMatch
    '''
    
    parser.add_argument('--epoch', type=int, default=1)
    parser.add_argument('--num_train_iter', type=int, default=2**20, 
                        help='total number of training iterations')
    parser.add_argument('--num_eval_iter', type=int, default=100,
                        help='evaluation frequency')
    parser.add_argument('--num_labels', type=int, default=4000)
    #(erase)fast_debugging
    parser.add_argument('--batch_size', type=int, default=4,
                        help='total number of batch size of labeled data')
    parser.add_argument('--uratio', type=int, default=7,
                        help='the ratio of unlabeled data to labeld data in each mini-batch')
    parser.add_argument('--eval_batch_size', type=int, default=32,
                        help='batch size of evaluation data loader (it does not affect the accuracy)')
    
    parser.add_argument('--hard_label', type=bool, default=True)
    parser.add_argument('--T', type=float, default=0.5)
    parser.add_argument('--p_cutoff', type=float, default=0.95)
    parser.add_argument('--ema_m', type=float, default=0.999, help='ema momentum for eval_model')
    parser.add_argument('--ulb_loss_ratio', type=float, default=1.0)
    parser.add_argument('--loss', 
        choices=['PAWSLoss', 'BidirectionalLoss', 'FixMatchLoss', 'BidirectionalLoss_all'],
        default='FixMatchLoss')
    parser.add_argument('--aggregator_module',
        choices=['PAWSNonParametric', 'QueuePAWSTransformer', 'PAWSTransformer', 'SelfNonParametric', 'SelfNonParametricEval', 'SelfNonParametric_Vis', 'SelfTransformer', 'Identity'],
        default='Identity')
    parser.add_argument('--aggregator-tau', type=float, default=0.1)
    parser.add_argument('--simsiam', action='store_true')
    parser.add_argument('--strong_aug', action='store_true')
    parser.add_argument('--simclr_aug', action='store_true')
    parser.add_argument('--one_hot_class_label', action='store_true')
    parser.add_argument('--softmax_class_label', action='store_true')

    '''
    Optimizer configurations
    '''
    parser.add_argument('--lr', type=float, default=0.03)
    parser.add_argument('--lr_aggregator', type=float, default=0.0003)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--amp', action='store_true', help='use mixed precision training or not')

    '''
    Backbone Net Configurations
    '''
    parser.add_argument('--net', type=str, default='WideResNet')
    parser.add_argument('--net_from_name', type=bool, default=False)
    parser.add_argument('--depth', type=int, default=28)
    parser.add_argument('--widen_factor', type=int, default=2)
    parser.add_argument('--leaky_slope', type=float, default=0.1)
    parser.add_argument('--dropout', type=float, default=0.0)
    
    '''
    Data Configurations
    '''
    
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--train_sampler', type=str, default='RandomSampler')
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=1)
    
    '''
    multi-GPUs & Distrbitued Training
    '''
    
    ## args for distributed training (from https://github.com/pytorch/examples/blob/master/imagenet/main.py)
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=1, type=int,
                        help='**node rank** for distributed training')
    parser.add_argument('--dist-url', default='tcp://127.0.0.1:10001', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--seed', default=0, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--gpu', default=0, type=int,
                        help='GPU id to use.')
    parser.add_argument('--multiprocessing-distributed', action='store_true',
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs. This is the '
                             'fastest way to use PyTorch for either single node or '
                             'multi node data parallel training')
    args = parser.parse_args()
    args.save_name = args.save_name + '_{}_{}_{}'.format(args.aggregator_module, args.loss, args.num_labels)
    if args.simclr_aug:
        args.save_name = args.save_name + '_simclrAug'
    if args.simsiam:
        args.save_name = args.save_name + '_simsiam'
    if args.strong_aug:
        args.save_name = args.save_name + '_strongAug'

    save_path = os.path.join(args.save_dir, args.save_name)
    if os.path.exists(save_path) and not args.overwrite:
        raise Exception('already existing model: {}'.format(save_path))
    if args.resume:
        if args.load_path is None:
            raise Exception('Resume of training requires --load_path in the args')
        if os.path.abspath(save_path) == os.path.abspath(args.load_path) and not args.overwrite:
            raise Exception('Saving & Loading pathes are same. \
                            If you want over-write, give --overwrite in the argument.')
        
    if args.seed is not None:
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')
        
    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')
    
    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])
    
    #distributed: true if manually selected or if world_size > 1
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed 
    ngpus_per_node = torch.cuda.device_count() # number of gpus of each node
    
    #divide the batch_size according to the number of nodes
    args.batch_size = int(args.batch_size / args.world_size)
    args.bn_momentum = 1 - args.ema_m


    _net_builder = net_builder(args.net, 
                               args.net_from_name,
                               {'depth': args.depth, 
                                'widen_factor': args.widen_factor,
                                'leaky_slope': args.leaky_slope,
                                'bn_momentum': args.bn_momentum,
                                'dropRate': args.dropout})
    
    model = FixMatch(_net_builder,
                     args.num_classes,
                     args.ema_m,
                     args.T,
                     args.p_cutoff,
                     args.ulb_loss_ratio,
                     args.hard_label,
                     num_eval_iter=args.num_eval_iter,
                     tb_log=None,
                     logger=None,
                     loss=args.loss,
                     aggregator_module=args.aggregator_module)
    torch.cuda.set_device(args.gpu)
    model.train_model = model.train_model.cuda(args.gpu)
    model.eval_model = model.eval_model.cuda(args.gpu)
    model.aggregator = model.aggregator.cuda(args.gpu)
    # Construct Dataset & DataLoader
    train_dset = SSL_Dataset(name=args.dataset, train=True, 
                             num_classes=args.num_classes, data_dir=args.data_dir,
                             simclr_aug=args.simclr_aug, strong_aug=args.strong_aug)
    lb_dset, ulb_dset = train_dset.get_ssl_dset(args.num_labels)
    
    _eval_dset = SSL_Dataset(name=args.dataset, train=False, simclr_aug=True,
                             num_classes=args.num_classes, data_dir=args.data_dir)
    eval_dset = _eval_dset.get_dset()
    
    loader_dict = {}
    dset_dict = {'train_lb': lb_dset, 'train_ulb': ulb_dset, 'eval': eval_dset}
    # labeled weak
    # ulb simclr
    loader_dict['train_lb'] = get_data_loader(dset_dict['train_lb'],
                                              args.batch_size,
                                              data_sampler = args.train_sampler,
                                              num_iters=args.num_train_iter,
                                              num_workers=args.num_workers, 
                                              distributed=args.distributed)
    
    loader_dict['train_ulb'] = get_data_loader(dset_dict['train_ulb'],
                                               args.batch_size,
                                               data_sampler = args.train_sampler,
                                               num_iters=args.num_train_iter,
                                               num_workers=4*args.num_workers,
                                               distributed=args.distributed)
    
    loader_dict['eval'] = get_data_loader(dset_dict['eval'],
                                          args.eval_batch_size, 
                                          num_workers=args.num_workers)
    
    ## set DataLoader on FixMatch
    model.set_data_loader(loader_dict)
    model.load_model(args.load_path)
    trainer = model.train
    trainer(args)
    # print(model.evaluate(args=args))
