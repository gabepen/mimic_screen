#!/usr/bin/env python3
"""Print monitor_pipeline.py args from a stage2 config (or exec monitor)."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_REPO_SRC = _REPO_ROOT / "src"
if str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))

from auto_lit_search.pipeline_config import load_stage2_config  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True, help="Stage2 YAML config path")
    p.add_argument(
        "--print-only",
        action="store_true",
        help="Print monitor args without running monitor_pipeline.py",
    )
    p.add_argument("monitor_args", nargs=argparse.REMAINDER, help="Extra args for monitor_pipeline.py")
    args = p.parse_args()

    try:
        cfg = load_stage2_config(args.config)
    except (OSError, ValueError, RuntimeError) as e:
        print(str(e), file=sys.stderr)
        return 2

    monitor_script = cfg.cluster.pipeline_root / "scripts" / "monitor_pipeline.py"
    cmd = [
        sys.executable,
        str(monitor_script),
        "--data-root",
        str(cfg.data_root),
        "--output-root",
        str(cfg.output_root),
        "--paper-ids",
        str(cfg.paper_ids_json),
    ]
    extra = [a for a in args.monitor_args if a != "--"]
    if not extra:
        extra = ["--watch", "30"]
    cmd.extend(extra)

    if args.print_only:
        print(" ".join(cmd))
        return 0

    return subprocess.call(cmd)


if __name__ == "__main__":
    sys.exit(main())
