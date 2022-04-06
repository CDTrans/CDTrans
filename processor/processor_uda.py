import logging
from typing import OrderedDict
import numpy as np
import os
import time
import torch
import torch.nn as nn
import cv2
from utils.meter import AverageMeter
from utils.metrics import R1_mAP, R1_mAP_eval, R1_mAP_Pseudo, R1_mAP_query_mining, R1_mAP_save_feature, R1_mAP_draw_figure, Class_accuracy_eval
from utils.reranking import re_ranking, re_ranking_numpy
from torch.nn.parallel import DistributedDataParallel
from torch.cuda import amp
import torch.distributed as dist
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import accuracy
from itertools import cycle
import os.path as osp
import torch.nn.functional as F
import random, pdb, math, copy
from scipy.spatial.distance import cdist
from sklearn.metrics import confusion_matrix
import torchvision.transforms as T
from datasets.make_dataloader import train_collate_fn, source_target_train_collate_fn
from datasets.sampler import RandomIdentitySampler
from datasets.bases import ImageDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment
from collections  import defaultdict

def obtain_label(logger, val_loader, model, distance='cosine', threshold=0):
    device = "cuda"
    start_test = True
    print('obtain label')
    model.eval()
    for n_iter, (img, vid, _, _, _) in enumerate(tqdm(val_loader)):
        with torch.no_grad():
            img = img.to(device)
            labels = torch.tensor(vid)
            probs = model(**dict(x = img, x2 = img, return_feat_prob=True))
            outputs, _, feas = probs[1]
            
            if start_test:
                all_fea = feas.float().cpu()
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

    all_output = nn.Softmax(dim=1)(all_output)
    _, predict = torch.max(all_output, 1)

    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    if distance == 'cosine':
        all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
        all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()
    ### all_fea: extractor feature [bs,N]

    all_fea = all_fea.float().cpu().numpy()
    K = all_output.size(1)
    aff = all_output.float().cpu().numpy()
    ### aff: softmax output [bs,c]

    initc = aff.transpose().dot(all_fea)
    initc = initc / (1e-8 + aff.sum(axis=0)[:, None])
    cls_count = np.eye(K)[predict].sum(axis=0)
    labelset = np.where(cls_count > threshold)
    labelset = labelset[0]

    dd = cdist(all_fea, initc[labelset], distance)
    pred_label = dd.argmin(axis=1)
    pred_label = labelset[pred_label]
    acc = np.sum(pred_label == all_label.float().numpy()) / len(all_fea)
    log_str = 'Fisrt Accuracy = {:.2f}% -> {:.2f}%'.format(accuracy * 100, acc * 100)

    logger.info(log_str)
    
    for round in range(1):
        aff = np.eye(K)[pred_label]
        initc = aff.transpose().dot(all_fea)
        initc = initc / (1e-8 + aff.sum(axis=0)[:, None])
        dd = cdist(all_fea, initc[labelset], distance)
        pred_label = dd.argmin(axis=1)
        pred_label = labelset[pred_label]

    acc = np.sum(pred_label == all_label.float().numpy()) / len(all_fea)
    log_str = 'Second Accuracy = {:.2f}% -> {:.2f}%'.format(accuracy * 100, acc * 100)
    logger.info(log_str)

    return pred_label.astype('int')

def update_feat(cfg, epoch, model, train_loader1,train_loader2, device,feat_memory1,feat_memory2, label_memory1,label_memory2):
    model.eval()
    for n_iter, (img, vid, _, _, idx) in enumerate(tqdm(train_loader1)):
        with torch.no_grad():
            img = img.to(device)
            feats = model(img, img)
            feat = feats[1]/(torch.norm(feats[1],2,1,True)+1e-8)
            feat_memory1[idx] = feat.detach().cpu()
            label_memory1[idx] = vid

    for n_iter, (img, vid, _, _, idx) in enumerate(tqdm(train_loader2)):
        with torch.no_grad():
            img = img.to(device)
            feats = model(img, img)
            feat = feats[1]/(torch.norm(feats[1],2,1,True)+1e-8)
            feat_memory2[idx] = feat.detach().cpu()
            label_memory2[idx] = vid


    return feat_memory1, feat_memory2, label_memory1, label_memory2
    
# def filter_knnidx(tensor_x, indexs, threshold):
#     return torch.where(tensor_x >=threshold, indexs, (-torch.ones(indexs.size())).long() )

def compute_knn_idx(logger, model, train_loader1, train_loader2, feat_memory1, feat_memory2, label_memory1, label_memory2, img_num1, img_num2, target_sample_num=2, topk=1, reliable_threshold=0.0):
    #assert((torch.sum(feat_memory2,axis=1)!=0).all())
    simmat = torch.matmul(feat_memory1,feat_memory2.T)
    _, knnidx = torch.max(simmat,1)

    if topk == 1:
        target_knnsim, target_knnidx = torch.max(simmat, 0)
    else:
        target_knnsim, target_knnidx = torch.topk(simmat,dim=0,k=topk)
        target_knnsim, target_knnidx = target_knnsim[topk-1, :], target_knnidx[topk-1, :]

    _, knnidx_topk = torch.topk(simmat,k=target_sample_num,dim=1)
    del simmat
    count_target_usage(logger, knnidx, label_memory1, label_memory2, img_num1, img_num2)

    target_label = obtain_label(logger, train_loader2, model)
    target_label = torch.from_numpy(target_label).cuda()

    return target_label, knnidx, knnidx_topk, target_knnidx

def count_target_usage(logger, idxs, label_memory1, label_memory2, img_num1, img_num2, source_idxs=None):
    unique_knnidx = torch.unique(idxs)
    logger.info('target number usage: {}'.format(len(unique_knnidx)/img_num2))
    if source_idxs is not None:
        source_unique_knnidx = torch.unique(source_idxs)
        logger.info('source number usage: {}'.format(len(source_unique_knnidx)/img_num1))
    else:
        logger.info('source number usage: 100%')

    per_class_num = torch.bincount(label_memory2)
    per_class_select_num = torch.bincount(label_memory2[unique_knnidx])
    logger.info('target each class usage: {} '.format(per_class_select_num/per_class_num[:len(per_class_select_num)]))
    if len(per_class_num) != len(per_class_select_num):
        logger.info('target last {} class usage is 0%'.format(len(per_class_num) - len(per_class_select_num)))
    
    target_labels = label_memory2[idxs]
    if source_idxs is not None: # sample should filter
        source_labels = label_memory1[source_idxs]
    else:
        source_labels = label_memory1

    logger.info('match right rate: {}'.format((target_labels==source_labels).int().sum()/len(target_labels)))
    matrix = confusion_matrix(target_labels, source_labels)
    acc = matrix.diagonal()/matrix.sum(axis=1) * 100
    aa = [str(np.round(i, 2)) for i in acc]
    logger.info('each target class match right rate: {}'.format(aa))


def generate_new_dataset(cfg, logger, label_memory2, s_dataset, t_dataset, knnidx, target_knnidx, target_pseudo_label, label_memory1, img_num1, img_num2, with_pseudo_label_filter=True):
    # generate new dataset
    train_set = []
    new_target_knnidx = []
    new_targetidx = []
    
    source_dataset = s_dataset.train
    target_dataset = t_dataset.train
    # combine_target_sample:
    for idx, data in enumerate(tqdm(target_dataset)):
        t_img_path, _, _, _,t_idx = data
        curidx = target_knnidx[t_idx]
        if curidx<0: continue
        source_data = source_dataset[curidx]
        s_img_path, label, camid, trackid, _  = source_data
        mask = label == target_pseudo_label[t_idx]
        if (with_pseudo_label_filter and mask) or not with_pseudo_label_filter:
            new_targetidx.append(t_idx)
            new_target_knnidx.append(curidx)
            train_set.append(((s_img_path, t_img_path), (label, target_pseudo_label[t_idx].item()), camid, trackid, (curidx, t_idx)))
    logger.info('target match accuracy') 
    count_target_usage(logger, torch.tensor(new_targetidx), label_memory1, label_memory2, img_num1, img_num2, source_idxs=torch.tensor(new_target_knnidx))
        
    # combine_source_sample:
    new_source_knnidx = []
    new_source_idx = []
    for idx, data in enumerate(tqdm(source_dataset)):
        s_img_path, label, camid, trackid,s_idx = data
        curidx = knnidx[s_idx]
        if curidx<0:continue
        target_data = target_dataset[curidx]
        t_img_path, _, _, _, _  = target_data
        mask = target_pseudo_label[curidx] == label
        if (with_pseudo_label_filter and mask) or not with_pseudo_label_filter:
            new_source_idx.append(s_idx)
            new_source_knnidx.append(curidx)
            train_set.append(((s_img_path, t_img_path), (label,target_pseudo_label[curidx].item()), camid, trackid, (s_idx, curidx.item())))
        
    logger.info('source match accuracy') 
    count_target_usage(logger, torch.tensor(new_source_knnidx), label_memory1, label_memory2, img_num1, img_num2, source_idxs=torch.tensor(new_source_idx))
    
    new_target_knnidx = new_target_knnidx + new_source_idx
    new_targetidx = new_targetidx + new_source_knnidx 

    count_target_usage(logger, torch.tensor(new_targetidx), label_memory1, label_memory2, img_num1, img_num2, source_idxs=torch.tensor(new_target_knnidx))
    train_transforms = T.Compose([
            T.Resize((256, 256)),
            T.RandomCrop((224, 224)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    new_dataset = ImageDataset(train_set, train_transforms)

    num_workers = cfg.DATALOADER.NUM_WORKERS
    train_loader = DataLoader(
            new_dataset, batch_size=cfg.SOLVER.IMS_PER_BATCH,
            shuffle=True, drop_last = True, 
            num_workers=num_workers, collate_fn=source_target_train_collate_fn,
            pin_memory=True, persistent_workers=True,
            )

    return train_loader

def do_train_uda(cfg,
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
             loss_fn,
             num_query, local_rank):
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD

    device = "cuda"
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("reid_baseline.train")
    logger.info('start training')
    _LOCAL_PROCESS_GROUP = None
    if device:
        model.to(local_rank)
        if cfg.MODEL.DIST_TRAIN:
            print('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)
        elif torch.cuda.device_count() > 1:
            model = nn.DataParallel(model).to(device)
        else:
            model.to(device)

    loss1_meter = AverageMeter()
    loss2_meter = AverageMeter()
    loss13_meter = AverageMeter()
    loss3_meter = AverageMeter()
    acc_meter = AverageMeter()
    acc_2_meter = AverageMeter()
    acc_2_pse_meter = AverageMeter()

    if cfg.MODEL.TASK_TYPE == 'classify_DA':
        evaluator = Class_accuracy_eval(logger=logger, dataset=cfg.DATASETS.NAMES)
    else:
        evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    
    scaler = amp.GradScaler()
    label_memory1 = torch.zeros((img_num1),dtype=torch.long)
    label_memory2 = torch.zeros((img_num2),dtype=torch.long)
    if '384' in cfg.MODEL.Transformer_TYPE or 'small' in cfg.MODEL.Transformer_TYPE:
        feat_memory1 = torch.zeros((img_num1,384),dtype=torch.float32)
        feat_memory2 = torch.zeros((img_num2,384),dtype=torch.float32)
    else:
        feat_memory1 = torch.zeros((img_num1,768),dtype=torch.float32)
        feat_memory2 = torch.zeros((img_num2,768),dtype=torch.float32)
    update_epoch = 10  #10
    
    best_model_mAP = 0
    min_mean_ent = 1e5        
            
    # train
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        loss1_meter.reset()
        loss2_meter.reset()
        loss13_meter.reset()
        loss3_meter.reset()
        acc_meter.reset()
        acc_2_meter.reset()
        acc_2_pse_meter.reset()
        evaluator.reset()
        scheduler.step(epoch)

        if (epoch-1) % update_epoch == 0:
            feat_memory1, feat_memory2, label_memory1, label_memory2 = update_feat(cfg, epoch, model, train_loader1,train_loader2, device,feat_memory1,feat_memory2, label_memory1,label_memory2)
            dynamic_top = 1
            print('source and target topk==',dynamic_top)
            target_label, knnidx, knnidx_topk, target_knnidx = compute_knn_idx(logger, model, train_loader1, train_loader2, feat_memory1, feat_memory2, label_memory1, label_memory2, img_num1, img_num2, topk=dynamic_top, reliable_threshold=0.0)
            del train_loader
            
            train_loader = generate_new_dataset(cfg, logger, label_memory2, s_dataset, t_dataset, knnidx, target_knnidx, target_label, label_memory1, img_num1, img_num2, with_pseudo_label_filter = cfg.SOLVER.WITH_PSEUDO_LABEL_FILTER)
                
        model.train()

        for n_iter, (imgs, vid, target_cam, target_view, file_name, idx) in enumerate(train_loader):
            img = imgs[0].to(device)
            t_img = imgs[1].to(device) #target img
            target = vid[0].to(device)
            t_pseudo_target = vid[1].to(device)
            s_idx,t_idx = idx
            label_knn = label_memory2[t_idx].cuda()

            optimizer.zero_grad()
            optimizer_center.zero_grad()
  
            target_cam = target_cam.to(device)
            target_view = target_view.to(device)
            with amp.autocast(enabled=True):
                def distill_loss(teacher_output, student_out):
                    teacher_out = F.softmax(teacher_output, dim=-1)    
                    loss = torch.sum( -teacher_out * F.log_softmax(student_out, dim=-1), dim=-1)
                    return loss.mean()
                    
                (self_score1, self_feat1, _), (score2, feat2, _), (score_fusion, _, _), aux_data  = model(img, t_img, target, cam_label=target_cam, view_label=target_view ) # output: source , target , source_target_fusion
                
                loss1 = loss_fn(self_score1, self_feat1, target, target_cam)
                loss2 = loss_fn(score2, feat2, t_pseudo_target, target_cam)
                loss3 = distill_loss(score_fusion, score2)
                loss = loss2 + loss3 + loss1

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            scaler.step(optimizer)
            scaler.update()

            if 'center' in cfg.MODEL.METRIC_LOSS_TYPE:
                for param in center_criterion.parameters():
                    param.grad.data *= (1. / cfg.SOLVER.CENTER_LOSS_WEIGHT)
                scaler.step(optimizer_center)
                scaler.update()
            if isinstance(self_score1, list):
                acc = (self_score1[0].max(1)[1] == target).float().mean()
            else:
                acc = (self_score1.max(1)[1] == target).float().mean()

            if isinstance(score2, list):
                acc2 = (score2[0].max(1)[1] == label_knn).float().mean()
            else:
                acc2 = (score2.max(1)[1] == label_knn).float().mean()

            if isinstance(score2, list):
                acc2_pse = (score2[0].max(1)[1] == t_pseudo_target).float().mean()
            else:
                acc2_pse = (score2.max(1)[1] == t_pseudo_target).float().mean()
                
            loss1_meter.update(loss1.item(), img.shape[0])
            loss2_meter.update(loss2.item(), img.shape[0])
            loss3_meter.update(loss3.item(), img.shape[0])
            acc_meter.update(acc, 1)
            acc_2_meter.update(acc2, 1)
            acc_2_pse_meter.update(acc2_pse, 1)

            torch.cuda.synchronize()
            if (n_iter + 1) % log_period == 0:
                logger.info("Epoch[{}] Iteration[{}/{}] Loss1: {:.3f}, Loss2: {:.3f}, Loss3: {:.3f},  Acc: {:.3f}, Acc2: {:.3f}, Acc2_pse: {:.3f}, Base Lr: {:.2e}"
                            .format(epoch, (n_iter + 1), len(train_loader),
                                    loss1_meter.avg, loss2_meter.avg, loss3_meter.avg, acc_meter.avg, acc_2_meter.avg, acc_2_pse_meter.avg, scheduler._get_lr(epoch)[0]))

        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)
        if cfg.MODEL.DIST_TRAIN:
            if dist.get_rank() == 0:
                logger.info("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                    .format(epoch, time_per_batch, train_loader.batch_size / time_per_batch))
        else:
            logger.info("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                    .format(epoch, time_per_batch, train_loader.batch_size / time_per_batch))

        if epoch % checkpoint_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    torch.save(model.state_dict(),
                               os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))
            else:
                torch.save(model.state_dict(),
                           os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))

        if epoch % eval_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    model.eval()
                    for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(val_loader):
                        with torch.no_grad():
                            img = img.to(device)
                            camids = camids.to(device)
                            target_view = target_view.to(device)
                            feat = model(img, cam_label=camids, view_label=target_view)
                            evaluator.update((feat, vid, camid))
                    cmc, mAP, _, _, _, _, _ = evaluator.compute()
                    logger.info("Validation Results - Epoch: {}".format(epoch))
                    logger.info("mAP: {:.1%}".format(mAP))
                    for r in [1, 5, 10]:
                        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                    torch.cuda.empty_cache()
            elif cfg.MODEL.TASK_TYPE == 'classify_DA':
                model.eval()
                # evaluate the imgknn and img 
                for n_iter, (img, vid, camid, camids, target_view, idex) in enumerate(val_loader):
                    with torch.no_grad():
                        img = img.to(device)
                        camids = camids.to(device)
                        target_view = target_view.to(device)
                        output_probs = model(img, img, cam_label=camids, view_label=target_view, return_logits=True, cls_embed_specific=False)
                        # try output prob and output_prob2
                        evaluator.update((output_probs[1], vid))
                accuracy, mean_ent = evaluator.compute()
                if accuracy > best_model_mAP:
                # if mean_ent <= min_mean_ent:
                    min_mean_ent = mean_ent
                    best_model_mAP = accuracy
                    torch.save(model.state_dict(),
                           os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_best_model.pth'))
                logger.info("Classify Domain Adapatation Validation Results - Epoch: {}".format(epoch))
                logger.info("Accuracy: {:.1%}, best Accuracy: {:.1%}, min Mean_entropy: {:.1}".format(accuracy, best_model_mAP, min_mean_ent))
                
                torch.cuda.empty_cache()
            else:
                model.eval()
                for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(val_loader):
                    with torch.no_grad():
                        img = img.to(device)
                        camids = camids.to(device)
                        target_view = target_view.to(device)
                        feat, feat2 = model(img, img, cam_label=camids, view_label=target_view)
                        evaluator.update((feat2, vid, camid))
                cmc, mAP, _, _, _, _, _ = evaluator.compute()
                if mAP > best_model_mAP:
                    best_model_mAP = mAP
                    torch.save(model.state_dict(),
                           os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_best_model.pth'))
                logger.info("Validation Results - Epoch: {}".format(epoch))
                logger.info("mAP: {:.1%}".format(mAP))
                for r in [1, 5, 10]:
                    logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                torch.cuda.empty_cache()
                
    # inference
    print('best model preformance is {}'.format(best_model_mAP))
    if torch.cuda.device_count() > 1:
        model.module.load_param_finetune(os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_best_model.pth'))
    else:
        model.load_param_finetune(os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_best_model.pth'))
    model.eval()
    evaluator.reset()

    for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(val_loader):
        with torch.no_grad():
            img = img.to(device)
            camids = camids.to(device)
            target_view = target_view.to(device)
            if cfg.MODEL.TASK_TYPE == 'classify_DA':
                feats = model(img, img, cam_label=camids, view_label=target_view, return_logits=True)
                evaluator.update((feats[1], vid))
            else:
                feat = model(img, img, cam_label=camids, view_label=target_view, return_logits=True)
                evaluator.update((feat, vid, camid))
    if cfg.MODEL.TASK_TYPE == 'classify_DA':
        accuracy, _ = evaluator.compute()  
        logger.info("Classify Domain Adapatation Validation Results - Best model")
        logger.info("Accuracy: {:.1%}".format(accuracy))
    else:
        cmc, mAP, _, _, _, _, _ = evaluator.compute()
        logger.info("Best Model Validation Results ")
        logger.info("mAP: {:.1%}".format(mAP))
        for r in [1, 5, 10]:
            logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))


def do_inference_uda(cfg,
                 model,
                 val_loader,
                 num_query):
    device = "cuda"
    logger = logging.getLogger("reid_baseline.test")
    logger.info("Enter inferencing")
    if cfg.MODEL.TASK_TYPE == 'classify_DA':
        evaluator = Class_accuracy_eval(dataset=cfg.DATASETS.NAMES, logger= logger)
    elif cfg.TEST.EVAL:
        evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    else:

        evaluator = R1_mAP_draw_figure(cfg, num_query, max_rank=50, feat_norm=True,
                       reranking=cfg.TEST.RE_RANKING)
        # evaluator = R1_mAP_save_feature(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM,
        #                reranking=cfg.TEST.RE_RANKING)
    evaluator.reset()

    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    model.eval()
    img_path_list = []
    for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(val_loader):
            with torch.no_grad():
                img = img.to(device)
                camids = camids.to(device)
                target_view = target_view.to(device)
                target = torch.tensor(vid).to(device)
                
                if cfg.MODEL.TASK_TYPE == 'classify_DA':
                    probs = model(img, img, cam_label=camids, view_label=target_view, return_logits=True)
                    evaluator.update((probs[1], vid))
                else:
                    feat1, feat2 = model(img, img, cam_label=camids, view_label=target_view, return_logits=False)
                    evaluator.update((feat2, vid, camid))

    if cfg.TEST.EVAL:
        if cfg.MODEL.TASK_TYPE == 'classify_DA':
            accuracy, mean_ent = evaluator.compute()  
            logger.info("Classify Domain Adapatation Validation Results - In the source trained model")
            logger.info("Accuracy: {:.1%}".format(accuracy))
            return 
        else:
            cmc, mAP, _, _, _, _, _ = evaluator.compute()
            logger.info("Validation Results ")
            logger.info("mAP: {:.1%}".format(mAP))
            for r in [1, 5, 10]:
                logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
            return cmc[0], cmc[4]
    else:
        print('yes begin saving feature')
        feats, distmats, pids, camids, viewids, img_name_path = evaluator.compute()

        torch.save(feats, os.path.join(cfg.OUTPUT_DIR, 'features.pth'))
        np.save(os.path.join(cfg.OUTPUT_DIR, 'distmat.npy'), distmats)
        np.save(os.path.join(cfg.OUTPUT_DIR, 'label.npy'), pids)
        np.save(os.path.join(cfg.OUTPUT_DIR, 'camera_label.npy'), camids)
        np.save(os.path.join(cfg.OUTPUT_DIR, 'image_name.npy'), img_name_path)
        np.save(os.path.join(cfg.OUTPUT_DIR, 'view_label.npy'), viewids)
        print('over')
