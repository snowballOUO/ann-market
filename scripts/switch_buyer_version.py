"""
Switch active BuyerSimulator version.

Usage:
    python scripts/switch_buyer_version.py soft       # 期望价+期望SLA 软约束（默认推荐）
    python scripts/switch_buyer_version.py fixed      # 超价/超SLA 硬拒
    python scripts/switch_buyer_version.py baseline   # 问题1未修复
    python scripts/switch_buyer_version.py show

Requires restarting Python processes after switching.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSIONS_DIR = ROOT / "src" / "data" / "buyer_versions"
ACTIVE_FILE = VERSIONS_DIR / "ACTIVE"
VALID = {"baseline", "fixed", "soft", "hetero", "must_serve", "original"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("version", nargs="?", default="show",
                    help="soft | hetero | fixed | baseline | show")
    args = ap.parse_args()

    if args.version == "show":
        current = ACTIVE_FILE.read_text().strip() if ACTIVE_FILE.exists() else "soft (default)"
        print(f"Active buyer version: {current}")
        for v in sorted(VALID):
            fname = (
                "hetero_soft.py" if v == "hetero"
                else "must_serve.py" if v == "must_serve"
                else f"problem1_{v}.py"
            )
            print(f"  {v:8s} → {VERSIONS_DIR / fname}")
        return

    v = args.version.lower()
    if v not in VALID:
        print(f"Unknown version '{v}'. Choose: {sorted(VALID)}", file=sys.stderr)
        sys.exit(1)

    VERSIONS_DIR.mkdir(parents=True, exist_ok=True)
    ACTIVE_FILE.write_text(v + "\n")
    print(f"Switched active buyer version → {v}")
    print("Restart any running Python processes before re-running experiments.")


if __name__ == "__main__":
    main()
