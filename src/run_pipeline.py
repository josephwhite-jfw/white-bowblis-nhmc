#!/usr/bin/env python
# run_pipeline.py — top-level runner for the NHMC pipeline

from __future__ import annotations
import argparse, os, sys
from typing import Iterable

# ---- Robust imports so it works in both styles ----
# (A) package-style:  python -m src.run_pipeline
# (B) script-style:   python src/run_pipeline.py
try:
    # A: we're inside the src package
    from . import provider, ownership, pbj, chow, panel  # type: ignore
except Exception:
    # B: add this file's folder (src/) to sys.path and import flat
    PKG_DIR = os.path.dirname(__file__)
    if PKG_DIR not in sys.path:
        sys.path.append(PKG_DIR)
    import provider, ownership, pbj, chow, panel  # type: ignore
# ---------------------------------------------------

STEPS = ("provider", "ownership", "pbj", "chow", "panel", "all")


def _run_step(mod, names: Iterable[str] = ("main", "run")):
    """Call mod.main() or mod.run()."""
    for nm in names:
        fn = getattr(mod, nm, None)
        if callable(fn):
            return fn()
    raise RuntimeError(f"{mod.__name__} has no main()/run() entrypoint")


def main():
    ap = argparse.ArgumentParser(
        description="Run NHMC pipeline steps"
    )
    ap.add_argument(
        "--step",
        choices=STEPS,
        default="all",
        help="Which step to run (default: all)",
    )
    args = ap.parse_args()

    steps = (args.step,) if args.step != "all" else ("provider", "ownership", "pbj", "chow", "panel")

    for s in steps:
        print(f"\n=== [{s}] =====================================================")
        if s == "provider":
            _run_step(provider)
        elif s == "ownership":
            _run_step(ownership)
        elif s == "pbj":
            _run_step(pbj)
        elif s == "chow":
            _run_step(chow)
        elif s == "panel":
            _run_step(panel)
        else:
            raise ValueError(s)


if __name__ == "__main__":
    main()