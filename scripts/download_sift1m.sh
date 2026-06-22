#!/usr/bin/env bash
# Download and extract SIFT1M dataset.
# About 500MB compressed, 1.2GB uncompressed.

set -euo pipefail

DATA_DIR="${1:-data/sift1m}"
mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

URL="ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz"
MIRROR="http://corpus-texmex.irisa.fr/"

if [ -f "sift_base.fvecs" ]; then
    echo "SIFT1M already extracted in $DATA_DIR"
    exit 0
fi

echo "Downloading SIFT1M (~500MB)..."
if command -v curl >/dev/null; then
    curl -O "$URL" || curl -O "${MIRROR}sift.tar.gz"
elif command -v wget >/dev/null; then
    wget "$URL" || wget "${MIRROR}sift.tar.gz"
else
    echo "Need curl or wget."
    exit 1
fi

echo "Extracting..."
tar -xzf sift.tar.gz --strip-components=1
rm sift.tar.gz

echo "Done. Files in $DATA_DIR:"
ls -lh "$PWD"
