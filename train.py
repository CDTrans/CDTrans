import os
from torch.backends import cudnn
from utils.logger import setup_logger
from datasets import make_dataloader
from model import make_model
from solver import make_optimizer, create_scheduler
from loss import make_loss
from processor import do_train_pretrain, do_train_uda
import random
import torch
import numpy as np
import os
import argparse
# from timm.scheduler import create_scheduler

from config import cfg
from timm.data import Mixup
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        "--config_file", default="", help="path to config file", type=str
    )

    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
    # parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
    #                     help='LR scheduler (default: "cosine"')
    # parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
    #                     help='learning rate (default: 5e-4)')
    # parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
    #                     help='learning rate noise on/off epoch percentages')
    # parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
    #                     help='learning rate noise limit percent (default: 0.67)')
    # parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
    #                     help='learning rate noise std-dev (default: 1.0)')
    # parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
    #                     help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    # parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
    #                     help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    # parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
    #                     help='LR decay rate (default: 0.1)')
    # parser.add_argument('--epochs', default=120, type=int)
    parser.add_argument("--local_rank", default=0, type=int)
    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    set_seed(cfg.SOLVER.SEED)
    if cfg.MODEL.DIST_TRAIN:
        torch.cuda.set_device(args.local_rank)
    else:
        pass

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("reid_baseline", output_dir, if_train=True)
    logger.info("Saving model in the path :{}".format(cfg.OUTPUT_DIR))
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    if cfg.MODEL.DIST_TRAIN:
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID

    if cfg.MODEL.UDA_STAGE == 'UDA':
        train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num, train_loader1, train_loader2, img_num1, img_num2, s_dataset, t_dataset = make_dataloader(cfg)
    else:
        train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)
    
    model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num = view_num)
    loss_func, center_criterion = make_loss(cfg, num_classes=num_classes)
    optimizer, optimizer_center = make_optimizer(cfg, model, center_criterion)
    scheduler = create_scheduler(cfg, optimizer)
    
    if cfg.MODEL.UDA_STAGE == 'UDA':
        do_train_uda(
        cfg,
        model,
        center_criterion,
        train_loader,
        train_loader1,
        train_loader2,
        img_num1,
        img_num2,
        val_loader,
        s_dataset, t_dataset,
        optimizer,
        optimizer_center,
        scheduler,  
        loss_func,
        num_query, args.local_rank
    )
    else:
        print('pretrain train')
        do_train_pretrain(
            cfg,
            model,
            center_criterion,
            train_loader,
            val_loader,
            optimizer,
            optimizer_center,
            scheduler,  
            loss_func,
            num_query, args.local_rank
        )

    
    
