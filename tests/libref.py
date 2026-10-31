"""Stored outputs of the torch reference solver for the checks (paths.LIBREF): the call computed by batched fp64 Rodas3
in torch (rate coefficients, exact Jacobian, LU without row exchanges, the call kernel's step control), written once
on exactly the checks' inputs. Every entry carries the sha256 of the inputs it was computed from (the network's
reaction list and NASA-9 files, the inputs, max_steps); a check computes the same digest and refuses a mismatch."""
import hashlib
from pathlib import Path

import numpy as np

import paths

_D = {}


def digest(*items):
    """sha256 over arrays or tensors (dtype, shape and bytes), strings and bytes, in order."""
    h = hashlib.sha256()
    for a in items:
        if isinstance(a, str):
            a = a.encode()
        if isinstance(a, bytes):
            h.update(b"b%d:" % len(a) + a)
            continue
        a = np.ascontiguousarray(a.detach().cpu().numpy() if hasattr(a, "detach") else np.asarray(a))
        h.update(f"a{a.dtype.str}{a.shape}:".encode() + a.tobytes())
    return h.hexdigest()


def netsig(network_file, nasa_dir, species):
    """Digest of a network: its reaction list file, the species order and each species' NASA-9 file."""
    return digest(Path(network_file).read_bytes(), "|".join(species),
                  *(Path(f"{nasa_dir}/{q}.txt").read_bytes() for q in species))


def load(path=None):
    """The stored arrays (an open npz, loaded once per path)."""
    path = Path(path or paths.LIBREF)
    if path not in _D:
        _D[path] = np.load(path)
    return _D[path]


def arrays(key, path=None):
    """The arrays stored directly under key: {name: array}."""
    D = load(path)
    return {k[len(key) + 1:]: D[k] for k in D.files if k.startswith(key + "/") and k.count("/") == key.count("/") + 1}


def entry(key, *inputs, path=None):
    """arrays(key) after checking the digest of the inputs against key/sha."""
    D = load(path)
    if f"{key}/sha" not in D.files:
        raise KeyError(f"{key}: not in {path or paths.LIBREF}")
    if str(D[f"{key}/sha"]) != digest(*inputs):
        raise ValueError(f"{key}: the inputs differ from those the stored reference was written for")
    return arrays(key, path)


def rates(sig, T, P, path=None):
    """The reference rate coefficients kf, kr [B, R] (three-body n_tot excluded) and n_tot [B] at T [B] K, P [B]
    dyn cm^-2 (numpy), stored for these exact T, P on the network sig."""
    D = load(path)
    key = f"rates/{digest(sig, T, P)[:16]}"
    if f"{key}/kf" not in D.files:
        raise ValueError(f"no stored rate coefficients for these T, P ({key} not in {path or paths.LIBREF})")
    return D[f"{key}/kf"], D[f"{key}/kr"], D[f"{key}/nd"]
