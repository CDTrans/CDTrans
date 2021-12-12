import torch
import torch.nn as nn
from .backbones.resnet import ResNet, BasicBlock, Bottleneck
from loss.arcface import ArcFace
from .backbones.resnet_ibn_a import resnet50_ibn_a,resnet101_ibn_a
from .backbones.se_resnet_ibn_a import se_resnet101_ibn_a
from .backbones.vit_pytorch import vit_base_patch16_224_TransReID, vit_small_patch16_224_TransReID
from .backbones.vit_pytorch_uda import uda_vit_base_patch16_224_TransReID, uda_vit_small_patch16_224_TransReID
import torch.nn.functional as F
from loss.metric_learning import Arcface, Cosface, AMSoftmax, CircleLoss

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class Backbone(nn.Module):
    def __init__(self, num_classes, cfg):
        super(Backbone, self).__init__()
        last_stride = cfg.MODEL.LAST_STRIDE
        model_path = cfg.MODEL.PRETRAIN_PATH
        model_name = cfg.MODEL.NAME
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.task_type = cfg.MODEL.TASK_TYPE

        if model_name == 'resnet50':
            self.in_planes = 2048
            self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck, frozen_stages=cfg.MODEL.FROZEN,
                               layers=[3, 4, 6, 3])
            print('using resnet50 as a backbone')
        elif model_name == 'resnet101':
            self.in_planes = 2048
            self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck, frozen_stages=cfg.MODEL.FROZEN,
                               layers=[3, 4, 23, 3])
            print('using resnet101 as a backbone')
        elif model_name == 'resnet50_ibn_a':
            self.in_planes = 2048
            self.base = resnet50_ibn_a(last_stride)
            print('using resnet50_ibn_a as a backbone')
        elif model_name == 'resnet101_ibn_a':
            self.in_planes = 2048
            self.base = resnet101_ibn_a(last_stride, frozen_stages=cfg.MODEL.FROZEN)
            print('using resnet101_ibn_a as a backbone')
        elif model_name == 'se_resnet101_ibn_a':
            self.in_planes = 2048
            self.base = se_resnet101_ibn_a(last_stride,frozen_stages=cfg.MODEL.FROZEN)
            print('using se_resnet101_ibn_a as a backbone')
        else:
            print('unsupported backbone! but got {}'.format(model_name))

        
        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......from {}'.format(model_path))
        elif pretrain_choice == 'un_pretrain':
            self.base.load_un_param(model_path)
            print('Loading un_pretrain model......from {}'.format(model_path))

        self.gap = nn.AdaptiveAvgPool2d(1)

        self.num_classes = num_classes

        if self.cos_layer:
            print('using cosine layer')
            self.arcface = ArcFace(self.in_planes, self.num_classes, s=30.0, m=0.50)
        else:
            self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

        self.bottleneck_2 = nn.LayerNorm(self.in_planes)
        
    def forward(self, x, label=None, cam_label=None, view_label=None, return_logits=False):  # label is unused if self.cos_layer == 'no'
        
        x = self.base(x, cam_label=cam_label)
        global_feat = nn.functional.avg_pool2d(x, x.shape[2:4])
        global_feat = global_feat.view(global_feat.shape[0], -1)  # flatten to (bs, 2048)

        if self.neck == 'no':
            feat = global_feat
        elif self.neck == 'bnneck':
            feat = self.bottleneck(global_feat)

        if return_logits:
            cls_score = self.classifier(feat)
            return cls_score
        
        if self.training:
            if self.cos_layer:
                cls_score = self.arcface(feat, label)
            else:
                cls_score = self.classifier(feat)

            return cls_score, global_feat  # global feature for triplet loss
        elif self.task_type == 'classify_DA': # test for classify domain adapatation
            if self.cos_layer:
                cls_score = self.arcface(feat, label)
            else:
                cls_score = self.classifier(feat)
            return cls_score
        
        else:

            if self.neck_feat == 'after':
                # print("Test with feature after BN")
                return feat
            else:
                # print("Test with feature before BN")
                return global_feat

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        if 'state_dict' in param_dict:
            param_dict = param_dict['state_dict']
        for i in param_dict:
            # if 'classifier' in i or 'arcface' in i:
            #     continue
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model from revise {}'.format(trained_path))

    def load_un_param(self, trained_path):
        param_dict = torch.load(trained_path)
        if 'state_dict' in param_dict:
            param_dict = param_dict['state_dict']
        for i in self.state_dict():
            if 'classifier' in i or 'arcface' in i:
                continue
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))


class build_transformer(nn.Module):
    def __init__(self, num_classes, camera_num, view_num, cfg, factory):
        super(build_transformer, self).__init__()
        last_stride = cfg.MODEL.LAST_STRIDE
        model_path = cfg.MODEL.PRETRAIN_PATH
        model_name = cfg.MODEL.NAME
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.task_type = cfg.MODEL.TASK_TYPE
        if '384' in cfg.MODEL.Transformer_TYPE or 'small' in cfg.MODEL.Transformer_TYPE:
            self.in_planes = 384 
        else:
            self.in_planes = 768
        self.bottleneck_dim = 256
        print('using Transformer_type: {} as a backbone'.format(cfg.MODEL.Transformer_TYPE))
        if cfg.MODEL.TASK_TYPE == 'classify_DA':
            self.base = factory[cfg.MODEL.Transformer_TYPE](img_size=cfg.INPUT.SIZE_CROP, aie_xishu=cfg.MODEL.AIE_COE,local_feature=cfg.MODEL.LOCAL_F, stride_size=cfg.MODEL.STRIDE_SIZE, drop_path_rate=cfg.MODEL.DROP_PATH)
        else:
            self.base = factory[cfg.MODEL.Transformer_TYPE](img_size=cfg.INPUT.SIZE_TRAIN, aie_xishu=cfg.MODEL.AIE_COE,local_feature=cfg.MODEL.LOCAL_F, stride_size=cfg.MODEL.STRIDE_SIZE, drop_path_rate=cfg.MODEL.DROP_PATH)

        self.gap = nn.AdaptiveAvgPool2d(1)

        self.num_classes = num_classes
        self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE
        if self.ID_LOSS_TYPE == 'arcface':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = Arcface(self.in_planes, self.num_classes,
                                    s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'cosface':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = Cosface(self.in_planes, self.num_classes,
                                    s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'amsoftmax':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = AMSoftmax(self.in_planes, self.num_classes,
                                        s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'circle':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE, cfg.SOLVER.COSINE_SCALE, cfg.SOLVER.COSINE_MARGIN))
            self.classifier = CircleLoss(self.in_planes, self.num_classes,
                                        s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        else:
            self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

        self._load_parameter(pretrain_choice, model_path)

    def _load_parameter(self, pretrain_choice, model_path):
        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......from {}'.format(model_path))
        elif pretrain_choice == 'un_pretrain':
            self.base.load_un_param(model_path)
            print('Loading trans_tune model......from {}'.format(model_path))
        elif pretrain_choice == 'pretrain':
            self.load_param_finetune(model_path)
            print('Loading pretrained model......from {}'.format(model_path))

    def forward(self, x, label=None, cam_label= None, view_label=None, return_logits=False):  # label is unused if self.cos_layer == 'no'
        global_feat = self.base(x, cam_label=cam_label, view_label=view_label)
        feat = self.bottleneck(global_feat)
        if return_logits:
            if self.cos_layer:
                cls_score = self.arcface(feat, label)
            else:
                cls_score = self.classifier(feat)
            return cls_score
        elif self.training:
            if self.ID_LOSS_TYPE in ('arcface', 'cosface', 'amsoftmax', 'circle'):
                cls_score = self.classifier(feat, label)
            else:
                cls_score = self.classifier(feat)

            return cls_score, global_feat  # global feature for triplet loss
        else:
            if self.neck_feat == 'after':
                # print("Test with feature after BN")
                return feat
            else:
                # print("Test with feature before BN")
                return global_feat

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            if 'classifier' in i or 'arcface' in i or 'bottleneck' in i or 'gap' in i:
                continue
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))


    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            if 'module.' in i: new_i = i.replace('module.','') 
            else: new_i = i 
            if new_i not in self.state_dict().keys():
                print('model parameter: {} not match'.format(new_i))
                continue
            self.state_dict()[new_i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))

class build_uda_transformer(nn.Module):
    def __init__(self, num_classes, camera_num, view_num, cfg, factory):
        super(build_uda_transformer, self).__init__()
        last_stride = cfg.MODEL.LAST_STRIDE
        model_path = cfg.MODEL.PRETRAIN_PATH
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.task_type = cfg.MODEL.TASK_TYPE
        self.in_planes = 384 if 'small' in cfg.MODEL.Transformer_TYPE else 768
        print('using Transformer_type: {} as a backbone'.format(cfg.MODEL.Transformer_TYPE))
        if cfg.MODEL.TASK_TYPE == 'classify_DA':
            self.base = factory[cfg.MODEL.Transformer_TYPE](img_size=cfg.INPUT.SIZE_CROP, aie_xishu=cfg.MODEL.AIE_COE,local_feature=cfg.MODEL.LOCAL_F, stride_size=cfg.MODEL.STRIDE_SIZE, drop_path_rate=cfg.MODEL.DROP_PATH, block_pattern=cfg.MODEL.BLOCK_PATTERN)
        else:
            self.base = factory[cfg.MODEL.Transformer_TYPE](img_size=cfg.INPUT.SIZE_TRAIN, aie_xishu=cfg.MODEL.AIE_COE,local_feature=cfg.MODEL.LOCAL_F, stride_size=cfg.MODEL.STRIDE_SIZE, drop_path_rate=cfg.MODEL.DROP_PATH, use_cross=cfg.MODEL.USE_CROSS, use_attn=cfg.MODEL.USE_ATTN,  block_pattern=cfg.MODEL.BLOCK_PATTERN)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.num_classes = num_classes
        self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE
        if self.ID_LOSS_TYPE == 'arcface':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = Arcface(self.in_planes, self.num_classes,
                                      s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'cosface':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = Cosface(self.in_planes, self.num_classes,
                                      s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'amsoftmax':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = AMSoftmax(self.in_planes, self.num_classes,
                                        s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'circle':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE, cfg.SOLVER.COSINE_SCALE, cfg.SOLVER.COSINE_MARGIN))
            self.classifier = CircleLoss(self.in_planes, self.num_classes,
                                        s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        else:
            self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......from {}'.format(model_path))
        elif pretrain_choice == 'un_pretrain':
            self.base.load_un_param(model_path)
            print('Loading trans_tune model......from {}'.format(model_path))
        elif pretrain_choice == 'pretrain':
            if model_path == '':
                print('make model without initialization')
            else:
                self.load_param_finetune(model_path)
                print('Loading pretrained model......from {}'.format(model_path))

    def forward(self, x, x2, label=None, cam_label= None, view_label=None, domain_norm=False, return_logits=False, return_feat_prob=False, cls_embed_specific=False):  # label is unused if self.cos_layer == 'no'
        inference_flag = not self.training
        global_feat, global_feat2, global_feat3, cross_attn = self.base(x, x2, cam_label=cam_label, view_label=view_label, domain_norm=domain_norm, cls_embed_specific=cls_embed_specific, inference_target_only=inference_flag)
        if self.neck == '':
            feat = global_feat
            feat2 = global_feat2
            feat3 = global_feat3
        else:
            feat = self.bottleneck(global_feat) if self.training else None
            feat3 = self.bottleneck(global_feat3) if self.training else None
            feat2 = self.bottleneck(global_feat2)

        if return_logits:
            if self.ID_LOSS_TYPE in ('arcface', 'cosface', 'amsoftmax', 'circle'):
                cls_score = self.classifier(feat, label)
                cls_score2 = self.classifier(feat2, label)
                cls_score3 = self.classifier(feat3, label) if global_feat3 is not None else None
            else:
                cls_score = self.classifier(feat)   if global_feat is not None else None
                cls_score2 = self.classifier(feat2)
                cls_score3 = self.classifier(feat3) if global_feat3 is not None else None

            return cls_score, cls_score2, cls_score3

        if self.training or return_feat_prob:
            if self.ID_LOSS_TYPE in ('arcface', 'cosface', 'amsoftmax', 'circle'):
                cls_score = self.classifier(feat, label) if self.training else None
                cls_score2 = self.classifier(feat2, label)
                cls_score3 = self.classifier(feat3, label) if self.training else None
            else:
                cls_score = self.classifier(feat) if self.training else None
                cls_score2 = self.classifier(feat2)
                cls_score3 = self.classifier(feat3) if self.training else None

            return (cls_score, global_feat, feat), (cls_score2, global_feat2, feat2), (cls_score3, global_feat3, feat3), cross_attn   # source , target , source_target_fusion
            
        else:
            if self.neck_feat == 'after' and self.neck != '':
                # print("Test with feature after BN")
                return feat, feat2, feat3
                
            else:
                # print("Test with feature before BN")
                return global_feat, global_feat2, global_feat3  # source , target , source_target_fusion
                

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            if 'classifier' in i or 'arcface' in i or 'bottleneck' in i or 'gap' in i:
                continue
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))


    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:  
            if 'module.' in i: new_i = i.replace('module.','') 
            else: new_i = i 
            if new_i not in self.state_dict().keys():
                print('model parameter: {} not match'.format(new_i))
                continue
            self.state_dict()[new_i].copy_(param_dict[i])

        print('Loading pretrained model for finetuning from {}'.format(model_path))


__factory_hh = {
    'vit_base_patch16_224_TransReID': vit_base_patch16_224_TransReID,
    'vit_small_patch16_224_TransReID': vit_small_patch16_224_TransReID, 
    'uda_vit_small_patch16_224_TransReID': uda_vit_small_patch16_224_TransReID, 
    'uda_vit_base_patch16_224_TransReID': uda_vit_base_patch16_224_TransReID,
    # 'resnet101': resnet101,
}

def make_model(cfg, num_class, camera_num, view_num):
    if cfg.MODEL.NAME == 'transformer':
        if cfg.MODEL.BLOCK_PATTERN == '3_branches':
            model = build_uda_transformer(num_class, camera_num, view_num, cfg, __factory_hh)
            print('===========building uda transformer===========')
        else:
            model = build_transformer(num_class, camera_num, view_num, cfg, __factory_hh)
            print('===========building transformer===========')
    else:
        print('===========ResNet===========')
        model = Backbone(num_class, cfg)
    return model
