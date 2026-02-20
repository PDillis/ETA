"""
Weighted distributed sampler for DDP-compatible bucket-based sampling.

Ported from ETA: carformer/carformer/utils/distributedsampler.py
"""

import math
import warnings

import torch
import torch.distributed as dist


class WeightedDistributedSampler(torch.utils.data.Sampler):
    """Sampler that uses per-sample weights for multinomial sampling,
    compatible with DistributedDataParallel.

    Each process gets an exclusive interleaved subset of the sampled indices.

    Args:
        dataset: Dataset used for sampling.
        subsample_ratio: Fraction of the dataset to sample per epoch (default 1.0).
        num_replicas: Number of DDP processes (auto-detected if None).
        rank: Rank of current process (auto-detected if None).
        weights: Per-sample weights (numpy array or list). If None, falls back to
                 uniform random or sequential depending on shuffle.
        shuffle: Whether to shuffle when weights are None.
    """

    def __init__(
        self,
        dataset,
        subsample_ratio=1.0,
        num_replicas=None,
        rank=None,
        weights=None,
        shuffle=True,
    ):
        if weights is not None:
            weights = torch.tensor(weights, dtype=torch.float)

        if num_replicas is None:
            if dist.is_available() and dist.is_initialized():
                num_replicas = dist.get_world_size()
            else:
                num_replicas = 1

        if rank is None:
            if dist.is_available() and dist.is_initialized():
                rank = dist.get_rank()
            else:
                rank = 0

        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(
            math.ceil(len(self.dataset) * 1.0 / self.num_replicas * subsample_ratio)
        )
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle
        self.weights = weights

    def __iter__(self):
        # Deterministically sample based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)

        if self.weights is not None:
            indices = torch.multinomial(
                self.weights, self.total_size, replacement=True, generator=g
            ).tolist()
        elif self.shuffle:
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        # Pad to make evenly divisible
        if len(indices) < self.total_size:
            indices += indices[: (self.total_size - len(indices))]

        # Interleaved split across ranks
        indices = indices[self.rank : self.total_size : self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch
