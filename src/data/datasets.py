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
    """Return (xb, xq, xt, gt) where:
       xb = base vectors  (1_000_000, 128)
       xq = query vectors (10_000, 128)
       xt = train vectors (100_000, 128) — used for index training
       gt = ground truth  (10_000, 100)  — top-100 exact neighbours per query
    """
    paths = {
        "xb": os.path.join(base_dir, "sift_base.fvecs"),
        "xq": os.path.join(base_dir, "sift_query.fvecs"),
        "xt": os.path.join(base_dir, "sift_learn.fvecs"),
        "gt": os.path.join(base_dir, "sift_groundtruth.ivecs"),
    }
    for k, p in paths.items():
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"Missing {k} at {p}. "
                "Run scripts/download_sift1m.sh first."
            )
    return (
        read_fvecs(paths["xb"]),
        read_fvecs(paths["xq"]),
        read_fvecs(paths["xt"]),
        read_ivecs(paths["gt"]),
    )


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