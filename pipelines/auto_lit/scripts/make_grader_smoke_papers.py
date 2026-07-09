#!/usr/bin/env python3
"""Build a small paper-id JSON for A5500 grader smoke tests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True, help="Full search output JSON")
    p.add_argument("--output", required=True, help="Trimmed output JSON")
    p.add_argument(
        "--max-papers",
        type=int,
        default=8,
        help="Number of papers to keep from the first alignment block",
    )
    args = p.parse_args()

    src = Path(args.source)
    out = Path(args.output)
    if not src.is_file():
        raise SystemExit(f"Source not found: {src}")

    data = json.loads(src.read_text(encoding="utf-8"))

    if isinstance(data, dict):
        items = data.get("papers") or data.get("results") or list(data.values())
    elif isinstance(data, list):
        items = data
    else:
        raise SystemExit(f"Unsupported JSON shape in {src}")

    trimmed = items[: max(1, args.max_papers)]
    if isinstance(data, dict):
        payload = dict(data)
        if "papers" in payload:
            payload["papers"] = trimmed
        elif "results" in payload:
            payload["results"] = trimmed
        else:
            payload = {"papers": trimmed}
    else:
        payload = trimmed

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(trimmed)} papers to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
