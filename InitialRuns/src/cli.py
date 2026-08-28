"""CLI: python -m src.bent2021_pipeline.cli --data-dir Data --out-dir outputs"""

from __future__ import annotations

import argparse
from pathlib import Path

from .constants import DEFAULT_RANDOM_STATE
from .io import project_root
from .pipeline import run_pipeline


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Reproduce Bent et al. 2021 (npj Digital Medicine) interstitial "
            "glucose classification and prediction pipeline."
        )
    )
    p.add_argument(
        "--data-dir",
        type=Path,
        default=project_root() / "Data",
        help="Directory containing patient folders (default: ./Data)",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=project_root() / "outputs",
        help="Output directory (default: ./outputs)",
    )
    p.add_argument(
        "--patients",
        nargs="*",
        default=None,
        help="Patient folder names (default: discover all under data-dir)",
    )
    p.add_argument("--seed", type=int, default=DEFAULT_RANDOM_STATE)
    p.add_argument(
        "--inspect-only",
        action="store_true",
        help="Load data and engineer features; do not train models",
    )
    p.add_argument("--skip-models", action="store_true")
    p.add_argument(
        "--skip-bvp-inspect",
        action="store_true",
        help="Skip scanning the large BVP file during validation",
    )
    args = p.parse_args()
    run_pipeline(
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        patient_ids=args.patients,
        random_state=args.seed,
        inspect_only=args.inspect_only,
        skip_models=args.skip_models,
        inspect_bvp=not args.skip_bvp_inspect,
    )


if __name__ == "__main__":
    main()
