#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import argparse
import json
import shutil
import subprocess
import sys


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Python wrapper for the item-level / dimension-level unified LMM R pipeline "
            "(S1-S5, B1-B3, IPQ items/dimensions; fixed effects kept consistent across DVs)."
        )
    )
    ap.add_argument("--long-csv", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("results/significance/item_level_lmm"))
    ap.add_argument("--exclude-subjects", default="")
    ap.add_argument("--p-adjust", default="fdr")
    ap.add_argument("--df-method", default="Satterthwaite")
    ap.add_argument("--rscript", default="Rscript", help="Rscript executable name or absolute path")
    args = ap.parse_args()

    rscript = shutil.which(args.rscript) if not Path(args.rscript).exists() else args.rscript
    if not rscript:
        raise SystemExit(
            "Rscript not found. Please install R or provide --rscript <path>. "
            "You can first run `python3 scripts/check_r_item_level_lmm.py` to diagnose the environment, "
            "and see `docs/R_SETUP_FOR_ITEM_LEVEL_LMM.md` for installation commands. "
            "This unified item-level LMM branch depends on the R stack: optparse, readr, dplyr, tidyr, stringr, lme4, lmerTest, emmeans, jsonlite."
        )

    script_path = Path(__file__).with_name("run_item_level_lmm_R.R")
    cmd = [
        str(rscript),
        str(script_path),
        "--long-csv", str(args.long_csv),
        "--out-dir", str(args.out_dir),
        "--exclude-subjects", str(args.exclude_subjects),
        "--p-adjust", str(args.p_adjust),
        "--df-method", str(args.df_method),
    ]
    p = subprocess.run(cmd)
    if p.returncode != 0:
        return p.returncode

    plot_script = Path(__file__).with_name("plot_item_level_lmm_outputs.py")
    plot_cmd = [sys.executable, str(plot_script), "--out-dir", str(args.out_dir)]
    p2 = subprocess.run(plot_cmd, capture_output=True, text=True)
    if p2.returncode != 0:
        print(p2.stderr or p2.stdout, file=sys.stderr)
        return p2.returncode

    summary_json = args.out_dir / "json" / "item_level_lmm_summary.json"
    png_summary_json = args.out_dir / "json" / "item_level_lmm_png_summary.json"
    if summary_json.exists() and png_summary_json.exists():
        try:
            payload = json.loads(summary_json.read_text(encoding="utf-8"))
            png_payload = json.loads(png_summary_json.read_text(encoding="utf-8"))
            outs = list(payload.get("outputs", []))
            for rel in png_payload.get("outputs", []):
                if rel not in outs:
                    outs.append(rel)
            payload["outputs"] = outs
            summary_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
