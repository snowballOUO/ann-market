"""
SIFT1M loader.

Download URL: ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz
(also mirrored at http://corpus-texmex.irisa.fr/)

This file does NOT download — it only reads the .fvecs files. Use
scripts/download_sift1m.sh to fetch the data first.

Format (.fvecs):
  Each vector is stored as:
    int32 dim (4 bytes)
    float32 values (4 * dim bytes)
"""
import os
import numpy as np


def read_fvecs(path: str) -> np.ndarray:
    """Load a .fvecs file into a float32 (N, dim) array."""
    a = np.fromfile(path, dtype=np.int32)
    if a.size == 0:
        raise IOError(f"Empty or missing file: {path}")
    dim = a[0]
    # Each row is (dim + 1) int32 ints when viewed as int32
    n = a.size // (dim + 1)
    a = a.reshape(n, dim + 1)
    # Drop the leading dim column, reinterpret remaining as float32
    return a[:, 1:].copy().view(np.float32)


def read_ivecs(path: str) -> np.ndarray:
    """Load a .ivecs file (used for ground-truth neighbour ids)."""
    a = np.fromfile(path, dtype=np.int32)
    dim = a[0]
    n = a.size // (dim + 1)
    a = a.reshape(n, dim + 1)
    return a[:, 1:].copy()


def load_sift1m(base_dir: str):
    """Backward-compatible wrapper. Use load_dataset() for new code."""
    return load_fvecs_dataset(base_dir, "sift")


def load_fvecs_dataset(base_dir: str, prefix: str):
    """Load a TexMex-style .fvecs/.ivecs dataset (SIFT1M, etc.).

    Args:
        base_dir: path containing .fvecs/.ivecs files
        prefix:   file prefix, e.g. "sift", "deep", "gist"

    Returns (xb, xq, xt, gt).
    """
    paths = {
        "xb": os.path.join(base_dir, f"{prefix}_base.fvecs"),
        "xq": os.path.join(base_dir, f"{prefix}_query.fvecs"),
        "xt": os.path.join(base_dir, f"{prefix}_learn.fvecs"),
        "gt": os.path.join(base_dir, f"{prefix}_groundtruth.ivecs"),
    }
    for k, p in paths.items():
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"Missing {k} at {p}. "
                f"Run scripts/download_data.py {prefix}1m first."
            )
    return (
        read_fvecs(paths["xb"]),
        read_fvecs(paths["xq"]),
        read_fvecs(paths["xt"]),
        read_ivecs(paths["gt"]),
    )


def load_dataset(cfg: dict):
    """Load any dataset based on config. Returns (xb, xq, xt, gt).

    cfg keys used:
        dataset.path, dataset.format, dataset.file, dataset.prefix
    """
    ds = cfg["dataset"]
    fmt = ds.get("format", "fvecs")
    data_dir = ds["path"]

    if fmt == "hdf5":
        filepath = os.path.join(data_dir, ds["file"])
        return load_hdf5(filepath)
    elif fmt == "fvecs":
        prefix = ds.get("prefix", "sift")
        return load_fvecs_dataset(data_dir, prefix)
    else:
        raise ValueError(f"Unknown dataset format: {fmt}")


def load_hdf5(filepath: str):
    """
    Load an ANN-benchmarks HDF5 dataset.

    HDF5 keys (standard benchmark format):
        train      — base vectors      (N, dim)
        test       — query vectors     (M, dim)
        neighbors  — ground truth IDs  (M, 100)
        distances  — ground truth distances (M, 100) — not used here

    Returns (xb, xq, xt, gt) — same interface as load_sift1m.
    """
    import h5py

    with h5py.File(filepath, "r") as f:
        xb = f["train"][:].astype(np.float32)
        xq = f["test"][:].astype(np.float32)
        gt = f["neighbors"][:].astype(np.int32)

    # use first 100K base vectors as training set for index building
    n_train = min(100000, xb.shape[0])
    xt = xb[:n_train].copy()

    return xb, xq, xt, gt