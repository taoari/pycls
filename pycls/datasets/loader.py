#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Data loader."""

import os

import torch
from pycls.core.config import cfg
from pycls.datasets.cifar10 import Cifar10
from pycls.datasets.imagenet import ImageNet
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler


# Supported datasets
_DATASETS = {"cifar10": Cifar10, "imagenet": ImageNet}

# Default data directory (/path/pycls/pycls/datasets/data)
_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# Relative data paths to default data directory
_PATHS = {"cifar10": "cifar10", "imagenet": "imagenet"}


def get_dataloader(dataset_name, split, batch_size, shuffle, drop_last):
    data_root = cfg.DATA_LOADER.DATA_ROOT if cfg.DATA_LOADER.DATA_ROOT else os.path.join(_DATA_DIR, _PATHS[dataset_name])
    data_aug = cfg.DATA_LOADER.DATA_AUG
    crop_size = cfg.TRAIN.IM_SIZE
    workers = cfg.DATA_LOADER.NUM_WORKERS
    pin_memory = cfg.DATA_LOADER.PIN_MEMORY
    backend = cfg.DATA_LOADER.BACKEND
    _world_size = 1 # NOTE: pycls only support for one node
    distributed = cfg.NUM_GPUS > 1

    is_train = split == 'train'
    if backend.startswith('dali'):
        from taowei.torch2.utils.classif import get_dataloader_dali
        dali_dataset = get_dataloader_dali(dataset=dataset_name, data_root=data_root,
            batch_size=batch_size, crop_size=crop_size, workers=workers, world_size=max(1, _world_size),
            backend=backend)
        loader = dali_dataset.get_train_loader() if is_train else dali_dataset.get_val_loader()
        loader.sampler = None # TODO: is it OK for dali distributed sampling without a sampler?
    else:
        from taowei.torch2.utils.classif import get_transform, get_dataset
        transform = get_transform(data_aug, is_train=is_train, crop_size=crop_size,
            backend=backend)
        dataset = get_dataset(dataset_name, data_root, split,
            transform=transform, num_examples=0, backend=backend)
        # Create a sampler for multi-process training
        sampler = DistributedSampler(dataset) if distributed else None
        # Create a loader
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(False if sampler else shuffle),
            sampler=sampler,
            num_workers=workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
        )
    return loader


def _construct_loader(dataset_name, split, batch_size, shuffle, drop_last):
    """Constructs the data loader for the given dataset."""
    if cfg.DATA_LOADER.BACKEND:
        return get_dataloader(dataset_name, split, batch_size, shuffle, drop_last)

    if cfg.DATA_LOADER.DATA_ROOT:
        dataset = _DATASETS[dataset_name](cfg.DATA_LOADER.DATA_ROOT, split)
    else:
        err_str = "Dataset '{}' not supported".format(dataset_name)
        assert dataset_name in _DATASETS and dataset_name in _PATHS, err_str
        # Retrieve the data path for the dataset
        data_path = os.path.join(_DATA_DIR, _PATHS[dataset_name])
        # Construct the dataset
        dataset = _DATASETS[dataset_name](data_path, split)
    # Create a sampler for multi-process training
    sampler = DistributedSampler(dataset) if cfg.NUM_GPUS > 1 else None
    # Create a loader
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(False if sampler else shuffle),
        sampler=sampler,
        num_workers=cfg.DATA_LOADER.NUM_WORKERS,
        pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
        drop_last=drop_last,
    )
    return loader


def construct_train_loader():
    """Train loader wrapper."""
    return _construct_loader(
        dataset_name=cfg.TRAIN.DATASET,
        split=cfg.TRAIN.SPLIT,
        batch_size=int(cfg.TRAIN.BATCH_SIZE / cfg.NUM_GPUS),
        shuffle=True,
        drop_last=True,
    )


def construct_test_loader():
    """Test loader wrapper."""
    return _construct_loader(
        dataset_name=cfg.TEST.DATASET,
        split=cfg.TEST.SPLIT,
        batch_size=int(cfg.TEST.BATCH_SIZE / cfg.NUM_GPUS),
        shuffle=False,
        drop_last=False,
    )


def shuffle(loader, cur_epoch):
    """"Shuffles the data."""
    err_str = "Sampler type '{}' not supported".format(type(loader.sampler))
    assert isinstance(loader.sampler, (RandomSampler, DistributedSampler)), err_str
    # RandomSampler handles shuffling automatically
    if isinstance(loader.sampler, DistributedSampler):
        # DistributedSampler shuffles data based on epoch
        loader.sampler.set_epoch(cur_epoch)
