"""
Samplers for DDP-compatible bucket-based and shard-grouped sampling.

WeightedDistributedSampler: ported from ETA: carformer/carformer/utils/distributedsampler.py
ShardGroupedSampler: reorders any base sampler's output for shard cache efficiency.
"""

import bisect
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


class ShardGroupedSampler(torch.utils.data.Sampler):
    """Reorders indices from a base sampler to group by shard for cache efficiency.

    Wraps any sampler (shuffle, weighted, distributed) and reorders its output
    so consecutive indices come from the same shard. Shard order is shuffled
    each epoch. Within each shard group, the base sampler's order is preserved.

    This is critical for lazy shard loading: without grouping, random access
    across ~950 shards causes each __getitem__ to load a new ~155MB file.
    With grouping, consecutive calls hit the same cached shard.

    Args:
        base_sampler: Any sampler whose output will be reordered.
        shard_offsets: Cumulative sample offsets per shard (from CARLA_Data.shard_offsets).
                       Length = num_shards + 1 (last entry is total_samples sentinel).
        seed: Random seed for deterministic shard shuffling (DDP-safe).
    """

    def __init__(self, base_sampler, shard_offsets, seed=0):
        self.base_sampler = base_sampler
        self.offsets = list(shard_offsets)
        self.seed = seed
        self.epoch = 0

    def __iter__(self):
        indices = list(self.base_sampler)

        # Group indices by shard (preserving base sampler's order within each group)
        num_shards = len(self.offsets) - 1
        buckets = [[] for _ in range(num_shards)]
        for idx in indices:
            shard_idx = bisect.bisect_right(self.offsets, idx) - 1
            buckets[shard_idx].append(idx)

        # Shuffle shard order each epoch (deterministic for DDP consistency)
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        shard_order = torch.randperm(num_shards, generator=g).tolist()

        # Concatenate in shuffled shard order
        result = []
        for s in shard_order:
            result.extend(buckets[s])

        return iter(result)

    def __len__(self):
        return len(self.base_sampler)

    def set_epoch(self, epoch):
        self.epoch = epoch
        if hasattr(self.base_sampler, 'set_epoch'):
            self.base_sampler.set_epoch(epoch)
