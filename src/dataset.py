#!/usr/bin/env python3
"""
dataset.py - Preloaded rollout windows for autoregressive training.

This module provides the dataset and dataloader helper for autoregressive
flow-map training on *pre-normalized* trajectory shards.

What the model expects (per model.py and trainer.py):
- y is already in z-space (normalized). For log-standardized species this means:
      z = (log10(y_phys) - log_mean) / log_std
  and log10(y_phys) can be recovered via:
      log10(y_phys) = z * log_std + log_mean

- dt is already normalized to [0, 1] using log10 + min-max (see preprocessing.py):
      dt_norm = (log10(dt_phys) - log_min) / (log_max - log_min)

- g (globals) is already normalized (method depends on your preprocessing/manifest).

Accordingly, this dataset does NOT apply NormalizationHelper during sampling.
The normalization manifest is still required for training/inference elsewhere
(e.g., loss conversion to log10-space, and post-hoc denormalization).

Data format (per shard NPZ; processing/preprocessing.py writes uncompressed ZIP
archives with np.savez):
    y_mat       : [N, T, S] float32  (z-space trajectories)
    globals     : [N, G]    float32  (normalized globals)
    dt_norm_mat : [N, T-1]  float32  (normalized dt per step in [0,1])

Every split is preloaded into contiguous float32 tensors (the shard dtype) on the
preload device; preload_nbytes() gives a split's preload size from the shard headers
alone, so the caller can check a memory budget before anything is loaded.
create_dataloader() wraps it in a vectorized single-process stream that samples
random windows (with replacement) and yields pre-batched tensors:
    {
        "y":  [B, K+1, S],   # includes the initial state at index 0
        "dt": [B, K],
        "g":  [B, G],
    }

There is no loading path for data larger than RAM. np.load cannot memory-map
.npz members, but because the members are stored uncompressed they can be opened
with np.memmap at their data offset (offsets are not float32-aligned), an input
for any future beyond-RAM loader.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Tuple, Union

import zipfile
from numpy.lib import format as npy_format

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset

# Preloaded tensors keep the shard dtype (float32): bf16 storage would quantize the z-space
# states, and float64 would only double the memory.
_STORAGE_DTYPE = torch.float32
# The training stream reseeds every epoch from SeedSequence([seed, _TRAIN_STREAM_ID, epoch]), so a
# resumed run draws the same batches as an uninterrupted one (the id keeps it apart from any
# other stream derived from the same seed).
_TRAIN_STREAM_ID = 0


# -----------------------------------------------------------------------------
# NPZ header helper (reads only the .npy header of a shard member)
# -----------------------------------------------------------------------------

def _npz_read_array_header(path: Path, *, key: str) -> Tuple[Tuple[int, ...], np.dtype]:
    """Return (shape, dtype) for `key` inside an .npz from its .npy header only.

    np.load(...)[key] reads the whole member; building the shard index only needs the
    header. Header versions other than 1.0 / 2.0 are rejected (regenerate the shards).
    """
    with zipfile.ZipFile(path, "r") as zf:
        with zf.open(f"{key}.npy") as f:
            ver = npy_format.read_magic(f)
            if ver == (1, 0):
                shape, _, dtype = npy_format.read_array_header_1_0(f)
            elif ver == (2, 0):
                shape, _, dtype = npy_format.read_array_header_2_0(f)
            else:
                raise ValueError(
                    f"{path}:{key}.npy has unsupported .npy header version {ver}; "
                    "regenerate the shards with processing/preprocessing.py."
                )
    return tuple(int(s) for s in shape), np.dtype(dtype)


@dataclass(frozen=True)
class ShardIndex:
    """Metadata for a single data shard."""
    path: Path
    n: int


def _scan_shards(split_dir: Path) -> Tuple[List[ShardIndex], int, int, int]:
    """Index a split's shards from header-only reads: (shards, T, S, G).

    The first shard fixes T, S and G; every shard must match them and be float32.
    """
    shard_paths = sorted(split_dir.glob("*.npz"))
    if not shard_paths:
        raise FileNotFoundError(f"No shard NPZ files found in {split_dir}")

    shards: List[ShardIndex] = []
    T = S = G = 0
    for p in shard_paths:
        y_shape, y_dtype = _npz_read_array_header(p, key="y_mat")
        g_shape, g_dtype = _npz_read_array_header(p, key="globals")
        dt_shape, dt_dtype = _npz_read_array_header(p, key="dt_norm_mat")

        if len(y_shape) != 3:
            raise ValueError(f"Shard {p} y_mat must be 3D [N,T,S], got {y_shape}")
        if len(g_shape) != 2:
            raise ValueError(f"Shard {p} globals must be 2D [N,G], got {g_shape}")
        if len(dt_shape) != 2:
            raise ValueError(f"Shard {p} dt_norm_mat must be 2D [N,T-1], got {dt_shape}")

        n, t, s = (int(y_shape[0]), int(y_shape[1]), int(y_shape[2]))
        if not shards:
            T, S, G = t, s, int(g_shape[1])
        if t != T or s != S:
            raise ValueError(f"Shard {p} y_mat shape is {y_shape}; expected [N,{T},{S}].")
        # Strict dtypes: avoid silent CPU copies / perf cliffs.
        if y_dtype != np.float32:
            raise ValueError(f"Shard {p} y_mat dtype must be float32, got {y_dtype}. Regenerate preprocessing outputs.")
        if g_dtype != np.float32:
            raise ValueError(f"Shard {p} globals dtype must be float32, got {g_dtype}. Regenerate preprocessing outputs.")
        if dt_dtype != np.float32:
            raise ValueError(f"Shard {p} dt_norm_mat dtype must be float32, got {dt_dtype}. Regenerate preprocessing outputs.")

        if int(g_shape[0]) != n or int(g_shape[1]) != G:
            raise ValueError(f"Shard {p} globals shape is {g_shape}; expected [{n},{G}].")
        if int(dt_shape[0]) != n or int(dt_shape[1]) != T - 1:
            raise ValueError(f"Shard {p} dt_norm_mat shape is {dt_shape}; expected [{n},{T - 1}].")

        if n <= 0:
            raise ValueError(f"Shard {p} has no trajectories.")
        shards.append(ShardIndex(path=p, n=n))
    return shards, T, S, G


def preload_nbytes(processed_dir: Union[str, Path], split: str) -> Tuple[int, int]:
    """(trajectories, bytes) that preloading `split` allocates: [N,T,S] + [N,T-1] + [N,G] float32."""
    split_dir = Path(processed_dir).expanduser().resolve() / split
    shards, T, S, G = _scan_shards(split_dir)
    n = int(sum(si.n for si in shards))
    return n, n * (T * S + (T - 1) + G) * _STORAGE_DTYPE.itemsize


class FlowMapRolloutDataset:
    """
    One split of trajectory shards, preloaded into contiguous tensors on `device`.

    create_dataloader() wraps it in the vectorized stochastic batch stream, which samples
    random windows of K+1 states (K transitions) with replacement.
    """

    def __init__(
        self,
        processed_dir: Union[str, Path],
        split: str,
        *,
        total_steps: int,
        windows_per_trajectory: int = 1,
        seed: int,
        device: torch.device,
    ) -> None:
        """
        Args:
            processed_dir: Root processed data directory containing the split subdirs.
            split: "train" | "validation" | "test" (or any subdir name under processed_dir).
            total_steps: K. Windows hold K+1 states (y) and K dt values.
            windows_per_trajectory: Controls *epoch size* (windows per trajectory per epoch).
                                  Sampling is still random each time.
            seed: Base seed for the batch stream RNG.
            device: Preload device; all shards are loaded into float32 tensors there.
        """
        processed_dir = Path(processed_dir).expanduser().resolve()
        split_dir = processed_dir / split
        if not split_dir.exists():
            raise FileNotFoundError(f"Split dir not found: {split_dir}")

        if total_steps <= 0:
            raise ValueError(f"total_steps must be positive, got {total_steps}")
        if windows_per_trajectory <= 0:
            raise ValueError(f"windows_per_trajectory must be positive, got {windows_per_trajectory}")

        self.processed_dir = processed_dir
        self.split = str(split)
        self.total_steps = int(total_steps)
        self.windows_per_trajectory = int(windows_per_trajectory)
        self.seed = int(seed)
        self.device = device

        # Build the shard index (trajectory counts per shard) from header-only reads.
        self.shards, self.T, self.S, self.G = _scan_shards(split_dir)

        self.max_start = int(self.T - (self.total_steps + 1))
        if self.max_start < 0:
            raise ValueError(f"Trajectories too short for total_steps={self.total_steps} (T={self.T}).")

        self._total_traj = int(sum(si.n for si in self.shards))

        self._preload_all()

    # -------------------------------------------------------------------------
    # Preload
    # -------------------------------------------------------------------------

    def _preload_all(self) -> None:
        """Load all shards into contiguous [N,T,S] / [N,G] / [N,T-1] tensors on *device*."""
        total = int(self._total_traj)
        if total <= 0:
            raise RuntimeError("no trajectories to preload")

        # Allocate contiguous buffers on the target device.
        y_all = torch.empty((total, self.T, self.S), device=self.device, dtype=_STORAGE_DTYPE)
        g_all = torch.empty((total, self.G), device=self.device, dtype=_STORAGE_DTYPE)
        dt_all = torch.empty((total, self.T - 1), device=self.device, dtype=_STORAGE_DTYPE)

        # Read each shard and copy it into the contiguous buffers.
        cursor = 0
        for si in self.shards:
            with np.load(si.path) as z:
                # Shards are float32 (checked from the headers); asarray does not copy them.
                y_np = np.asarray(z["y_mat"], dtype=np.float32)
                g_np = np.asarray(z["globals"], dtype=np.float32)
                dt_np = np.asarray(z["dt_norm_mat"], dtype=np.float32)

            n = int(y_np.shape[0])
            if n != int(si.n):
                raise ValueError(f"Shard {si.path} count mismatch: index says {si.n}, file has {n}.")
            if cursor + n > total:
                raise RuntimeError("preload overflow (bug)")

            y_all[cursor : cursor + n].copy_(torch.from_numpy(y_np), non_blocking=False)
            g_all[cursor : cursor + n].copy_(torch.from_numpy(g_np), non_blocking=False)
            dt_all[cursor : cursor + n].copy_(torch.from_numpy(dt_np), non_blocking=False)
            cursor += n

        if cursor != total:
            raise RuntimeError(f"preload size mismatch: filled {cursor}, expected {total}")

        self._preloaded_all = (y_all, g_all, dt_all)

    def __len__(self) -> int:
        # windows_per_trajectory controls how many samples you get per trajectory per epoch.
        return self._total_traj * self.windows_per_trajectory


class _PreloadedStochasticBatchStream(IterableDataset):
    """
    Vectorized stochastic batch stream over a preloaded FlowMapRolloutDataset.

    Yields already-batched tensors sampled from the contiguous preload buffers (one
    gather per batch instead of B per-sample calls). When the preload device is CPU,
    DataLoader pinning can be used on the yielded batches for efficient host-to-device
    transfer.

    Sampling is with replacement (SGD-style). The window indices are drawn from a CPU
    generator and moved to the preload device, so the window sequence does not depend
    on where the data is preloaded.

    epoch_seeded=True (the training stream): the generator is reseeded at the start of every
    epoch from the base seed and the epoch set by set_epoch(), so epoch e draws the same windows
    whether or not the run was resumed. Otherwise (validation) it is seeded once and continues.
    """

    def __init__(
        self,
        base: FlowMapRolloutDataset,
        *,
        batch_size: int,
        drop_last: bool,
        seed: int,
        epoch_seeded: bool,
    ) -> None:
        super().__init__()

        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")

        self.base = base
        self.batch_size = int(batch_size)
        self.drop_last = bool(drop_last)

        self.device = base.device

        self.K = int(base.total_steps)
        self.S = int(base.S)
        self.T = int(base.T)
        self.T_dt = int(base.T - 1)

        self._y_all, self._g_all, self._dt_all = base._preloaded_all
        self.N = int(self._y_all.shape[0])

        self._t_y = torch.arange(self.K + 1, device=self.device, dtype=torch.long)
        self._t_dt = torch.arange(self.K, device=self.device, dtype=torch.long)

        self._seed = int(seed)
        self._epoch_seeded = bool(epoch_seeded)
        self._epoch = 0

        self._generator = torch.Generator(device="cpu")
        # The +1 offset is historical (it separated this stream from the removed map-style
        # loader's RNG); it is kept so the validation windows are unchanged. An epoch-seeded
        # stream is reseeded in __iter__ before its first draw.
        self._generator.manual_seed(self._seed + 1)

    def set_epoch(self, epoch: int) -> None:
        """Select the epoch whose windows the next iteration draws (epoch-seeded streams)."""
        self._epoch = int(epoch)

    def __len__(self) -> int:
        total = len(self.base)
        if self.drop_last:
            return total // self.batch_size
        return (total + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        total_samples = len(self.base)
        num_batches = len(self)

        # The body of this generator runs at the first next(), i.e. after Lightning's
        # on_train_epoch_start has called set_epoch (Lightning builds the iterator earlier).
        if self._epoch_seeded:
            state = np.random.SeedSequence([self._seed, _TRAIN_STREAM_ID, self._epoch]).generate_state(1, dtype=np.uint64)
            self._generator.manual_seed(int(state[0]))

        # Views (no allocation).
        y_flat = self._y_all.reshape(-1, self.S)  # [N*T, S]
        dt_flat = self._dt_all.reshape(-1)        # [N*(T-1)]
        g_mat = self._g_all                       # [N, G]

        for b in range(num_batches):
            remaining = total_samples - b * self.batch_size
            B = self.batch_size if (self.drop_last or remaining >= self.batch_size) else remaining
            if B <= 0:
                break

            traj = torch.randint(0, self.N, (B,), generator=self._generator, dtype=torch.long).to(self.device)
            start = torch.randint(
                0,
                self.base.max_start + 1,
                (B,),
                generator=self._generator,
                dtype=torch.long,
            ).to(self.device)

            # Gather windows without materializing [B, T, S] intermediates.
            t_y = start[:, None] + self._t_y[None, :]  # [B, K+1]
            lin_y = (traj[:, None] * self.T + t_y).reshape(-1)
            y_win = y_flat.index_select(0, lin_y).reshape(B, self.K + 1, self.S)

            t_dt = start[:, None] + self._t_dt[None, :]  # [B, K]
            lin_dt = (traj[:, None] * self.T_dt + t_dt).reshape(-1)
            dt_win = dt_flat.index_select(0, lin_dt).reshape(B, self.K)

            g = g_mat.index_select(0, traj)

            yield {"y": y_win, "dt": dt_win, "g": g}


def create_dataloader(
    dataset: FlowMapRolloutDataset,
    *,
    batch_size: int,
    pin_memory: bool,
    drop_last: bool,
    epoch_seeded: bool,
) -> DataLoader:
    """
    Wrap a preloaded FlowMapRolloutDataset in the vectorized stochastic batch stream.

    The stream runs in the main process (num_workers=0) and samples windows with
    replacement; it yields already-batched tensors. epoch_seeded: see the stream class.
    """
    ds_len = len(dataset)
    if ds_len == 0:
        raise ValueError("Dataset is empty - check preprocessing outputs.")

    if drop_last and ds_len < batch_size:
        raise ValueError(
            f"drop_last=True would yield 0 batches (dataset size {ds_len} < batch_size {batch_size})."
        )

    stream = _PreloadedStochasticBatchStream(
        dataset,
        batch_size=batch_size,
        drop_last=bool(drop_last),
        seed=dataset.seed,
        epoch_seeded=bool(epoch_seeded),
    )

    # main.py rejects pin_memory=true with a CUDA preload device before any data is loaded.
    return DataLoader(
        dataset=stream,
        batch_size=None,  # stream yields already-batched tensors
        shuffle=False,
        num_workers=0,
        pin_memory=bool(pin_memory),
        persistent_workers=False,
    )
