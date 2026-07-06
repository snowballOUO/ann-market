"""
BuyerSimulator entry point — loads version from buyer_versions/ACTIVE.

Switch versions:
    python scripts/switch_buyer_version.py soft        # v2 期望价+期望SLA 软约束（推荐）
    python scripts/switch_buyer_version.py fixed       # v2 超价/超SLA 硬拒
    python scripts/switch_buyer_version.py baseline    # v2 问题1未修复
    python scripts/switch_buyer_version.py hetero      # v2 persona 锁定 + 满意度
    python scripts/switch_buyer_version.py must_serve  # v2 永不拒单
    python scripts/switch_buyer_version.py original    # 原始版本（GT recall + nprobe_recall + 3-type buyer）
"""
import os
from pathlib import Path

_VERSIONS_DIR = Path(__file__).resolve().parent / "buyer_versions"
_ACTIVE_FILE = _VERSIONS_DIR / "ACTIVE"


def _active_version() -> str:
    if "BUYER_VERSION" in os.environ:
        return os.environ["BUYER_VERSION"].strip().lower()
    if _ACTIVE_FILE.exists():
        return _ACTIVE_FILE.read_text().strip().lower()
    return "soft"


def _import_version(name: str):
    if name == "baseline":
        from src.data.buyer_versions.problem1_baseline import BuyerSimulator, BuyerProfile
    elif name == "fixed":
        from src.data.buyer_versions.problem1_fixed import BuyerSimulator, BuyerProfile
    elif name == "soft":
        from src.data.buyer_versions.problem1_soft import BuyerSimulator, BuyerProfile
    elif name == "hetero":
        from src.data.buyer_versions.hetero_soft import BuyerSimulator, BuyerProfile
    elif name == "must_serve":
        from src.data.buyer_versions.must_serve import BuyerSimulator, BuyerProfile
    elif name == "original":
        from src.data.buyer_versions.gt_recall_original import BuyerSimulator, BuyerProfile
    else:
        raise ValueError(
            f"Unknown buyer version '{name}'. "
            f"Run: python scripts/switch_buyer_version.py soft|fixed|baseline|hetero|must_serve|original"
        )
    return BuyerSimulator, BuyerProfile


_active = _active_version()
BuyerSimulator, BuyerProfile = _import_version(_active)
BUYER_VERSION = _active
