# from transformer_normal_DA_v0.datasets.office import Office
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from .bases import ImageDataset
from .preprocessing import RandomErasing
# from timm.data.random_erasing import RandomErasing
from .sampler import RandomIdentitySampler
from .sampler_ddp import RandomIdentitySampler_DDP
import torch.distributed as dist
from .ourapi import OURAPI
from .office_home import OfficeHome
from .visda import VisDA
from .domainnet import DomainNet
from .office import Office


__factory = {
    'OURAPI': OURAPI,
    'OfficeHome': OfficeHome,
    'VisDA': VisDA,
    'DomainNet': DomainNet,
    'Office': Office,

}

def train_collate_fn(batch):
    """
    # collate_fn这个函数的输入就是一个list，list的长度是一个batch size，list中的每个元素都是__getitem__得到的结果
    """
    imgs, pids, camids, viewids , _ , idx= zip(*batch)
    # print('train collate fn' , imgs)
    pids = torch.tensor(pids, dtype=torch.int64)
    viewids = torch.tensor(viewids, dtype=torch.int64)
    camids = torch.tensor(camids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids, camids, viewids, idx

def val_collate_fn(batch):##### revised by luo
    imgs, pids, camids, viewids, img_paths, idx = zip(*batch)
    viewids = torch.tensor(viewids, dtype=torch.int64)
    camids_batch = torch.tensor(camids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids, camids, camids_batch, viewids, img_paths

def source_target_train_collate_fn(batch):
    """
    # collate_fn这个函数的输入就是一个list，list的长度是一个batch size，list中的每个元素都是__getitem__得到的结果
    """
    b_data = zip(*batch)
    # print('b_data is {}'.format(b_data))
    # if len(b_data) == 8:
    s_imgs, t_imgs, s_pids, t_pids, camids, viewids , s_file_name, t_file_name , s_idx, t_idx = b_data
    # print('make dataloader collate_fn {}'.format(pids))
    # print(pids)
    s_pid = torch.tensor(s_pids, dtype=torch.int64)
    t_pid = torch.tensor(t_pids, dtype=torch.int64)
    pids = (s_pid, t_pid)

    file_name = (s_file_name, t_file_name)
    
    viewids = torch.tensor(viewids, dtype=torch.int64)
    camids = torch.tensor(camids, dtype=torch.int64)
    s_idx = torch.tensor(s_idx, dtype=torch.int64)
    t_idx = torch.tensor(t_idx, dtype=torch.int64)
    idx = (s_idx, t_idx)
    img1 = torch.stack(s_imgs, dim=0)
    img2 = torch.stack(t_imgs, dim=0)
    return (img1, img2), pids, camids, viewids, file_name, idx


from .autoaugment import AutoAugment

def make_dataloader(cfg):
    
    train_transforms = T.Compose([
        T.Resize((256, 256)),
        T.RandomCrop((224, 224)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    val_transforms = T.Compose([
        T.Resize((256, 256)),
        T.CenterCrop((224, 224)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    num_workers = cfg.DATALOADER.NUM_WORKERS
    dataset = __factory[cfg.DATASETS.NAMES](root_train=cfg.DATASETS.ROOT_TRAIN_DIR,root_val=cfg.DATASETS.ROOT_TEST_DIR, plus_num_id=cfg.DATASETS.PLUS_NUM_ID)
    train_set = ImageDataset(dataset.train, train_transforms)
    train_set1 = ImageDataset(dataset.train, val_transforms)
    
    train_set_normal = ImageDataset(dataset.train, val_transforms)
    img_num1 = len(dataset.train)
    

    if cfg.MODEL.UDA_STAGE == 'UDA':
        dataset2 = __factory[cfg.DATASETS.NAMES2](root_train=cfg.DATASETS.ROOT_TRAIN_DIR2,root_val=cfg.DATASETS.ROOT_TEST_DIR, plus_num_id=cfg.DATASETS.PLUS_NUM_ID)
        train_set2 = ImageDataset(dataset2.train, val_transforms)
        img_num2 = len(dataset2.train)
    
    num_classes = max(dataset.num_train_pids, dataset.num_test_pids)
    cam_num = dataset.num_train_cams
    view_num = dataset.num_train_vids
    
    if cfg.MODEL.DIST_TRAIN:
        print('DIST_TRAIN START')
        mini_batch_size = cfg.SOLVER.IMS_PER_BATCH // dist.get_world_size()
        data_sampler = RandomIdentitySampler_DDP(dataset.train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE)
        batch_sampler = torch.utils.data.sampler.BatchSampler(data_sampler, mini_batch_size, True)
        train_loader = torch.utils.data.DataLoader(
            train_set,
            num_workers=num_workers,
            batch_sampler=batch_sampler,
            collate_fn=train_collate_fn,
            pin_memory=True,
        )
    elif cfg.MODEL.UDA_STAGE == 'UDA':
        train_loader = DataLoader(
                train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH,
                sampler=RandomIdentitySampler(dataset.train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE),
                num_workers=num_workers, collate_fn=train_collate_fn
            )
        train_loader1 = DataLoader(
            train_set1, batch_size=cfg.TEST.IMS_PER_BATCH,
            shuffle=False,
            num_workers=num_workers, collate_fn=train_collate_fn,
            persistent_workers=True, pin_memory=True,

        )
        train_loader2 = DataLoader(
            train_set2, batch_size=cfg.TEST.IMS_PER_BATCH,
            shuffle=False,
            num_workers=num_workers, collate_fn=train_collate_fn,
            pin_memory=True, persistent_workers=True,
        )
    else:
        print('use shuffle sampler strategy')
        train_loader = DataLoader(
        train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH, shuffle=True, num_workers=num_workers,
        collate_fn=train_collate_fn
    )
        
    if cfg.DATASETS.QUERY_MINING:
        if cfg.MODEL.TASK_TYPE == 'classify_DA':
            val_set = ImageDataset(dataset.test, val_transforms)
        else:
            val_set = ImageDataset(dataset.query + dataset.query, val_transforms)
    else:
        if cfg.MODEL.TASK_TYPE == 'classify_DA':
            val_set = ImageDataset(dataset.test, val_transforms)
        else:
            val_set = ImageDataset(dataset.query + dataset.gallery, val_transforms)
            
    val_loader = DataLoader(
        val_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_fn
    )
    train_loader_normal = DataLoader(
        train_set_normal, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_fn
    )
    if cfg.MODEL.TASK_TYPE == 'classify_DA':
        if cfg.MODEL.UDA_STAGE == 'UDA':
            return train_loader, train_loader_normal, val_loader, None, num_classes, cam_num, view_num, train_loader1, train_loader2, img_num1, img_num2, dataset,dataset2
        else:
            return train_loader, train_loader_normal, val_loader, None, num_classes, cam_num, view_num
    else:
        return train_loader, train_loader_normal, val_loader, len(dataset.query), num_classes, cam_num, view_num


def make_dataloader_Pseudo(cfg):
    train_transforms = T.Compose([
            T.Resize(cfg.INPUT.SIZE_TRAIN),
            T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
            T.Pad(cfg.INPUT.PADDING),
            T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
            T.ToTensor(),
            T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
            RandomErasing(probability=cfg.INPUT.RE_PROB, mean=cfg.INPUT.PIXEL_MEAN)
        ])
    val_transforms = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TEST),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    ])
    print('using size :{} for training'.format(cfg.INPUT.SIZE_TRAIN))

    num_workers = cfg.DATALOADER.NUM_WORKERS

    dataset = __factory[cfg.DATASETS.NAMES](root=cfg.DATASETS.ROOT_DIR,plus_num_id=cfg.DATASETS.PLUS_NUM_ID)
    num_classes = dataset.num_train_pids

    train_set = ImageDataset(dataset.train, train_transforms)

    if 'triplet' in cfg.DATALOADER.SAMPLER:
        train_loader = DataLoader(
            train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH,
            sampler=RandomIdentitySampler(dataset.train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE),
            num_workers=num_workers, collate_fn=train_collate_fn
        )
    elif cfg.DATALOADER.SAMPLER == 'softmax':
        print('using softmax sampler')
        train_loader = DataLoader(
            train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH, shuffle=True, num_workers=num_workers,
            collate_fn=train_collate_fn
        )
    else:
        print('unsupported sampler! expected softmax or triplet but got {}'.format(cfg.SAMPLER))

    val_set = ImageDataset(dataset.query + dataset.gallery, val_transforms)
    val_loader = DataLoader(
        val_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_fn
    )
    return train_loader, val_loader, len(dataset.query), num_classes, dataset, train_set, train_transforms
