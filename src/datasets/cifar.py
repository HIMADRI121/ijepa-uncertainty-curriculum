# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import time
import numpy as np
import torch
import torchvision
from logging import getLogger

logger = getLogger()
from torchvision.datasets import CIFAR100

class CIFAR100Subset(CIFAR100):
    def __init__(self, root, train=True, max_samples=1000, **kwargs):
        super().__init__(root, train=train, **kwargs)
        self.data = self.data[:max_samples]
        self.targets = self.targets[:max_samples]
        
    
def make_cifar10(
    transform,
    batch_size,
    collator=None,
    pin_mem=True,
    num_workers=8,
    world_size=1,
    rank=0,
    root_path='./data',
    training=True,
    drop_last=True
):
    # CIFAR-10 dataset
    dataset = CIFAR10Wrapper(
        root=root_path,
        train=training,
        transform=transform,
        download=True
    )
    logger.info('CIFAR-10 dataset created')
    
    # Distributed sampler
    dist_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset=dataset,
        num_replicas=world_size,
        rank=rank
    )
    
    # Data loader
    data_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=collator,
        sampler=dist_sampler,
        batch_size=batch_size,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=False
    )
    logger.info('CIFAR-10 data loader created')

    return dataset, data_loader, dist_sampler

class CIFAR10Wrapper(torchvision.datasets.CIFAR10):
    def __init__(
        self,
        root='./data',
        train=True,
        transform=None,
        download=False,
        index_targets=False
    ):
        super().__init__(
            root=root,
            train=train,
            transform=transform,
            download=download
        )
        
        if index_targets:
            self.targets = np.array(self.targets)
            self.data = np.array(self.data)
            
            self.target_indices = []
            for t in range(len(self.classes)):
                indices = np.where(self.targets == t)[0].tolist()
                self.target_indices.append(indices)
                logger.debug(f'Class {t} has {len(indices)} samples')
                
            logger.info(f'Created target indices for {len(self.classes)} classes')

# Example usage:
if __name__ == "__main__":
    # Example transform
    transform = torchvision.transforms.Compose([
        torchvision.transforms.RandomResizedCrop(32),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2470, 0.2435, 0.2616]
        )
    ])
    
    # Create dataset and loader
    dataset, loader, sampler = make_cifar10(
        transform=transform,
        batch_size=64,
        num_workers=4,
        root_path='./data'
    )
    
    # Test iteration
    for images, labels in loader:
        print(f"Batch shape: {images.shape}")
        print(f"Labels: {labels}")
        break