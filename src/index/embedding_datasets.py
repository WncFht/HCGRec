from bisect import bisect_right

import numpy as np
import torch
from torch.utils import data


def _as_float_tensor(x: np.ndarray) -> torch.Tensor:
    arr = np.asarray(x, dtype=np.float32)
    # np.load(..., mmap_mode="r") returns a read-only memmap; converting it to a tensor without
    # copying may lead to undefined behavior if any in-place op happens on that tensor.
    if (not arr.flags.writeable) or (not arr.flags.c_contiguous):
        arr = np.array(arr, dtype=np.float32, copy=True)
    return torch.from_numpy(arr)


class EmbDataset(data.Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        # self.embeddings = np.fromfile(data_path, dtype=np.float32).reshape(16859,-1)
        self.embeddings = np.load(data_path, mmap_mode="r").squeeze()
        self.dim = self.embeddings.shape[-1]

    def __getitem__(self, index):
        emb = self.embeddings[index]
        return _as_float_tensor(emb)

    def __len__(self):
        return len(self.embeddings)


class MultiEmbDataset(data.Dataset):
    def __init__(self, data_paths):
        if not data_paths:
            raise ValueError("data_paths must be a non-empty list of .npy paths")

        self.data_paths = list(data_paths)
        self.embeddings_list = []
        self.cumulative_sizes = []
        self.dim = None

        total = 0
        for p in self.data_paths:
            emb = np.load(p, mmap_mode="r").squeeze()
            if emb.ndim == 1:
                emb = emb.reshape(1, -1)
            if emb.ndim != 2:
                raise ValueError(f"Expected a 2D array after squeeze, got shape={emb.shape} from {p}")

            if self.dim is None:
                self.dim = emb.shape[-1]
            elif emb.shape[-1] != self.dim:
                raise ValueError(f"Embedding dim mismatch: expected dim={self.dim}, got dim={emb.shape[-1]} from {p}")

            self.embeddings_list.append(emb)
            total += len(emb)
            self.cumulative_sizes.append(total)

    def __len__(self):
        return self.cumulative_sizes[-1]

    def _get_np_single(self, idx: int):
        if idx < 0:
            idx += len(self)
        if idx < 0 or idx >= len(self):
            raise IndexError("index out of range")

        dataset_idx = bisect_right(self.cumulative_sizes, idx)
        start = 0 if dataset_idx == 0 else self.cumulative_sizes[dataset_idx - 1]
        local_idx = idx - start
        return self.embeddings_list[dataset_idx][local_idx]

    def __getitem__(self, index):
        if isinstance(index, slice):
            idxs = range(*index.indices(len(self)))
            embs_list = [self._get_np_single(i) for i in idxs]
            if not embs_list:
                return torch.empty((0, self.dim), dtype=torch.float32)
            return _as_float_tensor(np.stack(embs_list, axis=0))

        if isinstance(index, torch.Tensor):
            index = index.cpu().numpy()

        if isinstance(index, (list, tuple, np.ndarray)):
            idxs = np.array(index, dtype=np.int64).tolist()
            if not idxs:
                return torch.empty((0, self.dim), dtype=torch.float32)
            embs = np.stack([self._get_np_single(i) for i in idxs], axis=0)
            return _as_float_tensor(embs)

        emb = self._get_np_single(index)
        return _as_float_tensor(emb)
