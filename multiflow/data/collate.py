from collections.abc import Mapping
import numpy as np
import torch
from torch.utils.data._utils.collate import default_collate


def _to_tensor_if_numpy(x):
    """
    Recursively convert numpy objects inside list/dict to torch.Tensor.
    - np.ndarray -> torch.from_numpy
    - np.number  -> torch.tensor(scalar)
    - torch.Tensor stays
    - other types stay (e.g., str, int, float)
    """
    if torch.is_tensor(x):
        return x
    if isinstance(x, np.ndarray):
        # torch.from_numpy shares memory when possible
        return torch.from_numpy(x)
    if isinstance(x, np.generic):  # numpy scalar types, e.g. np.int64
        return torch.tensor(x.item())
    if isinstance(x, list):
        return [_to_tensor_if_numpy(v) for v in x]
    if isinstance(x, tuple):
        return tuple(_to_tensor_if_numpy(v) for v in x)
    if isinstance(x, dict):
        return {k: _to_tensor_if_numpy(v) for k, v in x.items()}
    return x


def keep_list_collate(batch):
    """
    Rules:
    - For dict batch: collate per key.
    - If a key's values are python lists (per-sample), keep them as list-of-list,
      but convert numpy inside those lists into torch tensors.
    - Otherwise, follow default_collate as much as possible.
    """
    if batch is None or len(batch) == 0:
        return batch

    elem = batch[0]

    # Case 1: mapping/dict sample
    if isinstance(elem, Mapping):
        out = {}
        keys = elem.keys()
        for k in keys:
            values = [d[k] for d in batch]

            # key-wise rule: if per-sample value is list => keep list-of-list
            if isinstance(values[0], list):
                # keep as [list(sample0), list(sample1), ...]
                # but convert numpy inside
                out[k] = [_to_tensor_if_numpy(v) for v in values]
            else:
                # not a list => recurse / default behavior
                out[k] = keep_list_collate(values)
        return out

    # Case 2: if the thing itself is a list (rare at top-level)
    # We keep list-of-list; but convert numpy inside each element
    if isinstance(elem, list):
        return [_to_tensor_if_numpy(v) for v in batch]

    # Case 3: everything else: try default_collate
    # (handles tensors, numpy arrays, numbers, nested dicts/tuples, etc.)
    try:
        return default_collate(batch)
    except TypeError:
        # fallback: if default_collate can't handle (e.g. strings), just return raw list
        return batch