import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader

from .bases import ImageDataset
from timm.data.random_erasing import RandomErasing
from .sampler import RandomIdentitySampler
from .market1501 import Market1501
from .msmt17 import MSMT17
from .dev_market import DevMarket
from .dev_msmt import DevMSMT
from .sampler_ddp import RandomIdentitySampler_DDP
import torch.distributed as dist
from .veri import VeRi

__factory = {
    'market1501': Market1501,
    'msmt17': MSMT17,
    'veri': VeRi,
    'dev_market': DevMarket,
    'dev_msmt': DevMSMT,
}

def train_collate_fn(batch):
    imgs, pids, camids, viewids , _ = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    viewids = torch.tensor(viewids, dtype=torch.int64)
    camids = torch.tensor(camids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids, camids, viewids,

def val_collate_fn(batch):
    imgs, pids, camids, viewids, img_paths = zip(*batch)
    viewids = torch.tensor(viewids, dtype=torch.int64)
    camids_batch = torch.tensor(camids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids, camids, camids_batch, viewids, img_paths

def make_dataloader(cfg):
    train_transforms = T.Compose([
            T.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=3),
            T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
            T.Pad(cfg.INPUT.PADDING),
            T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
            T.ToTensor(),
            T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
            RandomErasing(probability=cfg.INPUT.RE_PROB, mode='pixel', max_count=1, device='cpu'),
        ])

    val_transforms = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TEST),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    ])

    num_workers = cfg.DATALOADER.NUM_WORKERS
    if cfg.DATA_COMBINE == True:
        # dataset1 = DevMarket(root=cfg.DATASETS.ROOT_DIR)
        # dataset2 = DevMSMT(root=cfg.DATASETS.ROOT_DIR)
        """
        for test
        
        """
        dataset1 = Market1501(root=cfg.DATASETS.ROOT_DIR)
        dataset2 = MSMT17(root=cfg.DATASETS.ROOT_DIR)
        pid_offset = dataset1.num_train_pids
        cam_offset = dataset2.num_train_cams
        
        train_data = dataset1.train + [(img_path, pid + pid_offset, camid + cam_offset, viewid) for img_path, pid, camid, viewid in dataset2.train]
        
        if cfg.DATA_COMBINE_EVAL == 0:
            query_data = dataset1.query + dataset2.query
            gallery_data = dataset1.gallery + dataset2.gallery
        elif cfg.DATA_COMBINE_EVAL == 1:
            query_data = dataset1.query
            gallery_data = dataset1.gallery 
        elif cfg.DATA_COMBINE_EVAL == 2:
            query_data = dataset2.query
            gallery_data = dataset2.gallery
        
        # ImageDataset으로 변환
        train_set = ImageDataset(train_data, train_transforms)
        train_set_normal = ImageDataset(train_data, val_transforms)
        val_set = ImageDataset(query_data + gallery_data, val_transforms)
        num_classes = dataset1.num_train_pids + dataset2.num_train_pids
        cam_num = dataset1.num_train_cams + dataset2.num_train_cams
        view_num = max(dataset1.num_train_vids, dataset2.num_train_vids)

    else:
        dataset = __factory[cfg.DATASETS.NAMES](root=cfg.DATASETS.ROOT_DIR)
        train_set = ImageDataset(dataset.train, train_transforms)
        train_set_normal = ImageDataset(dataset.train, val_transforms)
        val_set = ImageDataset(dataset.query + dataset.gallery, val_transforms)
        num_classes = dataset.num_train_pids
        cam_num = dataset.num_train_cams
        view_num = dataset.num_train_vids

    if 'triplet' in cfg.DATALOADER.SAMPLER:
        if cfg.MODEL.DIST_TRAIN:
            print('DIST_TRAIN START')
            mini_batch_size = cfg.SOLVER.STAGE2.IMS_PER_BATCH // dist.get_world_size()
            data_sampler = RandomIdentitySampler_DDP(train_data, cfg.SOLVER.STAGE2.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE)
            batch_sampler = torch.utils.data.sampler.BatchSampler(data_sampler, mini_batch_size, True)
            train_loader_stage2 = torch.utils.data.DataLoader(
                train_set,
                num_workers=num_workers,
                batch_sampler=batch_sampler,
                collate_fn=train_collate_fn,
                pin_memory=True,
            )
        else:
            train_loader_stage2 = DataLoader(
                train_set, batch_size=cfg.SOLVER.STAGE2.IMS_PER_BATCH,
                sampler=RandomIdentitySampler(train_data, cfg.SOLVER.STAGE2.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE),
                num_workers=num_workers, collate_fn=train_collate_fn
            )
    elif cfg.DATALOADER.SAMPLER == 'softmax':
        print('using softmax sampler')
        train_loader_stage2 = DataLoader(
            train_set, batch_size=cfg.SOLVER.STAGE2.IMS_PER_BATCH, shuffle=True, num_workers=num_workers,
            collate_fn=train_collate_fn
        )
    else:
        print('unsupported sampler! expected softmax or triplet but got {}'.format(cfg.SAMPLER))

    val_loader = DataLoader(
        val_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_fn
    )
    train_loader_stage1 = DataLoader(
        train_set_normal, batch_size=cfg.SOLVER.STAGE1.IMS_PER_BATCH, shuffle=True, num_workers=num_workers,
        collate_fn=train_collate_fn
    )
    
    return train_loader_stage2, train_loader_stage1, val_loader, len(query_data), num_classes, cam_num, view_num

