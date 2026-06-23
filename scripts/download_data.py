"""
Download ANN benchmark datasets.

Usage:
    python scripts/download_data.py sift1m         # ~500MB
    python scripts/download_data.py deep1m         # ~1.5GB
    python scripts/download_data.py gist1m         # ~2.6GB
    python scripts/download_data.py msmarco        # ~15GB
    python scripts/download_data.py all            # download all
"""
import argparse
import os
import sys
import tarfile
import urllib.request

DATASETS = {
    "sift1m": {
        "urls": [
            "https://ann-benchmarks.com/sift-128-euclidean.hdf5",
            "http://ann-benchmarks.com/sift-128-euclidean.hdf5",
        ],
        "dir": "data/sift1m",
        "check_file": "sift_base.fvecs",
        "size_estimate": "~500MB",
        "is_hdf5": True,
        "help": "If download fails, get sift.tar.gz from http://corpus-texmex.irisa.fr/ and extract into data/sift1m/",
    },
    "deep1m": {
        "urls": [
            "https://ann-benchmarks.com/deep-image-96-angular.hdf5",
            "http://ann-benchmarks.com/deep-image-96-angular.hdf5",
        ],
        "dir": "data/deep1m",
        "check_file": "deep-image-96-angular.hdf5",
        "size_estimate": "~1GB",
        "is_hdf5": True,
        "help": "If download fails, get deep.tar.gz from http://corpus-texmex.irisa.fr/ and extract into data/deep1m/",
    },
    "gist1m": {
        "urls": [
            "https://ann-benchmarks.com/gist-960-euclidean.hdf5",
            "http://ann-benchmarks.com/gist-960-euclidean.hdf5",
        ],
        "dir": "data/gist1m",
        "check_file": "gist-960-euclidean.hdf5",
        "size_estimate": "~3GB",
        "is_hdf5": True,
        "help": "If download fails, get gist.tar.gz from http://corpus-texmex.irisa.fr/ and extract into data/gist1m/",
    },
    "msmarco": {
        "urls": [
            "https://ann-benchmarks.com/msmarco-passage-384-euclidean.hdf5",
            "http://ann-benchmarks.com/msmarco-passage-384-euclidean.hdf5",
        ],
        "dir": "data/msmarco",
        "check_file": "msmarco-passage-384-euclidean.hdf5",
        "size_estimate": "~8GB",
        "is_hdf5": True,
        "help": "If download fails, try: huggingface.co/datasets/microsoft/msmarco",
    },
}


def download_file(url: str, dest: str) -> None:
    """Download with progress display and browser User-Agent."""

    def _progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100, downloaded * 100 // total_size)
            mb = downloaded / (1024 * 1024)
            total_mb = total_size / (1024 * 1024)
            sys.stdout.write(f"\r  {pct:3d}%  {mb:.0f}/{total_mb:.0f} MB")
            sys.stdout.flush()

    opener = urllib.request.build_opener()
    opener.addheaders = [
        ("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"),
    ]
    urllib.request.install_opener(opener)
    urllib.request.urlretrieve(url, dest, _progress)
    print()


def _is_tar_gz(path: str) -> bool:
    return path.endswith(".tar.gz") or path.endswith(".tgz")


def download_dataset(name: str) -> None:
    cfg = DATASETS[name]
    os.makedirs(cfg["dir"], exist_ok=True)

    check_path = os.path.join(cfg["dir"], cfg["check_file"])
    if os.path.exists(check_path):
        print(f"[{name}] already exists in {cfg['dir']}, skipping.")
        return

    archive_name = os.path.basename(cfg["urls"][0])
    archive_path = os.path.join(cfg["dir"], archive_name)

    print(f"[{name}] downloading {cfg['size_estimate']}...")
    success = False
    last_error = None
    for url in cfg["urls"]:
        try:
            download_file(url, archive_path)
            success = True
            break
        except Exception as e:
            last_error = e
            print(f"  failed: {url}")
            continue

    if not success:
        print(f"\n  All URLs failed. Last error: {last_error}")
        print(f"  {cfg.get('help', 'Download manually and place files in ' + cfg['dir'])}")
        # Clean up partial download
        if os.path.exists(archive_path):
            os.remove(archive_path)
        return

    is_hdf5 = cfg.get("is_hdf5", False)
    if not is_hdf5 and _is_tar_gz(archive_path):
        print(f"[{name}] extracting...")
        with tarfile.open(archive_path, "r:gz") as tar:
            members = tar.getmembers()
            for m in members:
                parts = m.name.split("/", 1)
                m.name = parts[1] if len(parts) > 1 else m.name
                tar.extract(m, cfg["dir"])
        os.remove(archive_path)

    print(f"[{name}] done. Files in {cfg['dir']}:")
    for f in sorted(os.listdir(cfg["dir"])):
        size_mb = os.path.getsize(os.path.join(cfg["dir"], f)) / (1024 * 1024)
        print(f"  {f:40s} {size_mb:.1f} MB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=list(DATASETS) + ["all"],
                    help="Dataset name or 'all'")
    args = ap.parse_args()

    if args.dataset == "all":
        for name in DATASETS:
            download_dataset(name)
    else:
        download_dataset(args.dataset)


if __name__ == "__main__":
    main()
