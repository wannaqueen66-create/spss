#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import argparse
import subprocess
import sys
import json

import matplotlib.pyplot as plt

from plot_style import apply_bae_style

QC_EXCLUDE = "孙校聪,康少勇,张钰鹏,杨可,洪婷婷,陈韬,高梓楠,赵国宏"


def run(cmd: list[str]):
    print("$", " ".join(cmd))
    p = subprocess.run(cmd)
    if p.returncode != 0:
        raise SystemExit(p.returncode)


def _summary_box(ax, title: str, lines: list[str]):
    ax.axis("off")
    ax.set_facecolor("#F7FAF8")
    ax.text(0.03, 0.97, title, va="top", ha="left", fontsize=10.2, fontweight="bold", color="#40534C")
    ax.text(
        0.03, 0.90, "\n".join(lines), va="top", ha="left", fontsize=8.9, color="#50615A", linespacing=1.38,
        bbox=dict(boxstyle="round,pad=0.45", fc="#F4F8F6", ec="#D5DFD9", lw=0.8)
    )


def _write_index(out: Path, branches: list[str]) -> None:
    lines = []
    lines.append("# Significance module index")
    lines.append("")
    lines.append("## Read this order first")
    lines.append("")
    for branch in branches:
        lines.append(f"### {branch}")
        lines.append(f"1. `./{branch}/overall/core_model/md/results_draft_zh.md` — Afford4 core-model narrative")
        lines.append(f"2. `./{branch}/overall/core_model/png/effect_size_summary.png` — Afford4 effect-size quick read")
        lines.append(f"3. `./{branch}/item_level_lmm/md/item_level_lmm_report_zh.md` — S1–S5 / B1–B3 / IPQ 逐题/逐维度统一结构 LMM 汇总")
        lines.append(f"3b. `./{branch}/item_level_lmm/png/item_level_type3_heatmap_all.png` — all-item significance heatmap for PPT/reporting")
        lines.append(f"4. `./{branch}/overall/wwr_polynomial/wwr_polynomial_contrasts.csv` — overall WWR linear/quadratic significance")
        lines.append(f"4. `./{branch}/experience/wwr_polynomial_group_only/wwr_polynomial_contrasts.csv` — experience-group significance")
        lines.append(f"5. `./{branch}/experience/wwr_polynomial_group_round/csv/wwr_polynomial_contrasts.csv` — experience × round follow-up")
        lines.append("")
    (out / "significance_index.md").write_text("\n".join(lines), encoding="utf-8")


def _write_research_map(out: Path, branches: list[str]) -> None:
    lines = []
    lines.append("# Research questions map — significance")
    lines.append("")
    lines.append("## Q1. Does WWR show significant linear or quadratic trends overall?")
    for branch in branches:
        lines.append(f"- {branch}: `./{branch}/overall/wwr_polynomial/wwr_polynomial_contrasts.csv`")
    lines.append("")
    lines.append("## Q2. What is the direction of those trends?")
    for branch in branches:
        lines.append(f"- {branch}: `./{branch}/overall/wwr_polynomial/wwr_polynomial_contrasts.csv` (Direction column) + `./{branch}/overall/wwr_polynomial/wwr_polynomial_figures/*.png`")
    lines.append("")
    lines.append("## Q3. What does the Afford4 core model say about WWR / Complexity / ExperienceGroup?")
    for branch in branches:
        lines.append(f"- {branch}: `./{branch}/overall/core_model/csv/model_comparison.csv` + `./{branch}/overall/core_model/csv/table_main_interactions.csv` + `./{branch}/overall/core_model/csv/effect_size_summary.csv` + `./{branch}/overall/core_model/png/effect_size_summary.png` + `./{branch}/overall/core_model/md/results_draft_zh.md`")
    lines.append("")
    lines.append("## Q4. What do item-level significance results say for S1–S5 / B1–B3 / IPQ1–IPQ6?")
    for branch in branches:
        lines.append(f"- {branch}: `./{branch}/item_level_lmm/md/item_level_lmm_report_zh.md` + `./{branch}/item_level_lmm/csv/item_level_lmm_type3_fixed_effects.csv` + `./{branch}/item_level_lmm/csv/item_level_lmm_type3_fixed_effects_fdr.csv` + `./{branch}/item_level_lmm/csv/item_level_lmm_fixed_effect_estimates.csv` + `./{branch}/item_level_lmm/csv/item_level_lmm_emmeans.csv` + `./{branch}/item_level_lmm/csv/item_level_lmm_pairwise.csv` + `./{branch}/item_level_lmm/png/item_level_type3_heatmap_all.png` + `./{branch}/item_level_lmm/png/item_level_pairwise_sig_summary.png`")
    lines.append("")
    lines.append("## Q5. Do high/low experience groups differ in WWR significance patterns?")
    for branch in branches:
        lines.append(f"- {branch}: `./{branch}/experience/wwr_polynomial_group_only/wwr_polynomial_contrasts.csv`")
    lines.append("")
    lines.append("## Q6. Do experience effects change by round?")
    for branch in branches:
        lines.append(f"- {branch}: `./{branch}/experience/wwr_polynomial_group_round/csv/wwr_polynomial_contrasts.csv`")
    (out / "research_questions_map.md").write_text("\n".join(lines), encoding="utf-8")


def _write_guide_png(out: Path, branches: list[str]) -> str:
    apply_bae_style()
    fig = plt.figure(figsize=(10.6, 6.0))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1.25], wspace=0.16)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    lines1 = []
    for branch in branches:
        lines1 += [
            f"[{branch}]",
            "1. overall / core_model / md / results_draft_zh.md",
            "2. item_level_lmm / md / item_level_lmm_report_zh.md",
            "3. overall / wwr_polynomial / wwr_polynomial_contrasts.csv",
            "4. experience / wwr_polynomial_group_only / wwr_polynomial_contrasts.csv",
            "5. experience / wwr_polynomial_group_round / wwr_polynomial_contrasts.csv",
            "",
        ]
    _summary_box(ax1, "Read this order first", lines1)

    lines2 = []
    for branch in branches:
        lines2 += [
            f"[{branch}] Q1/Q2 → overall / wwr_polynomial",
            f"[{branch}] Q3 → overall / core_model",
            f"[{branch}] Q4 → item_level_lmm / md + csv + png / Type III + EMMs + pairwise",
            f"[{branch}] Q5 → experience / wwr_polynomial_group_only",
            f"[{branch}] Q6 → experience / wwr_polynomial_group_round",
            "",
        ]
    _summary_box(ax2, "Research questions map", lines2)

    path = out / "significance_guide.png"
    fig.savefig(path, dpi=230)
    plt.close(fig)
    return str(path)


def main():
    ap = argparse.ArgumentParser(description="Significance-only pipeline: overall + experience")
    ap.add_argument("--long-csv", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("results/significance"))
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--with-qc", action="store_true", help="Also export QC-excluded outputs")
    args = ap.parse_args()

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    outputs: list[str] = []
    branches = [("raw", "")]
    if args.with_qc:
        branches.append(("qc", QC_EXCLUDE))

    for branch, exclude in branches:
        base = out / branch

        run([
            args.python, "scripts/run_analysis.py",
            "--long-csv", str(args.long_csv),
            "--out-dir", str(base / "overall" / "core_model"),
            "--exclude-subjects", exclude,
        ])
        outputs.append(str((base / "overall" / "core_model").relative_to(out)))

        run([
            args.python, "scripts/wwr_polynomial_significance.py",
            "--long-csv", str(args.long_csv),
            "--out-dir", str(base / "overall" / "wwr_polynomial"),
            "--exclude-subjects", exclude,
        ])
        outputs.append(str((base / "overall" / "wwr_polynomial").relative_to(out)))

        run([
            args.python, "scripts/wwr_polynomial_significance.py",
            "--long-csv", str(args.long_csv),
            "--out-dir", str(base / "experience" / "wwr_polynomial_group_only"),
            "--split-by", "ExperienceGroup",
            "--exclude-subjects", exclude,
        ])
        outputs.append(str((base / "experience" / "wwr_polynomial_group_only").relative_to(out)))

        run([
            args.python, "scripts/wwr_polynomial_significance.py",
            "--long-csv", str(args.long_csv),
            "--out-dir", str(base / "experience" / "wwr_polynomial_group_round"),
            "--split-by", "Repetition,ExperienceGroup",
            "--exclude-subjects", exclude,
        ])
        outputs.append(str((base / "experience" / "wwr_polynomial_group_round").relative_to(out)))

        run([
            args.python, "scripts/run_item_level_lmm.py",
            "--long-csv", str(args.long_csv),
            "--out-dir", str(base / "item_level_lmm"),
            "--exclude-subjects", exclude,
            "--p-adjust", "fdr",
            "--df-method", "Satterthwaite",
        ])
        outputs.append(str((base / "item_level_lmm").relative_to(out)))

    branch_names = [b for b, _ in branches]
    _write_index(out, branch_names)
    _write_research_map(out, branch_names)
    guide_png = _write_guide_png(out, branch_names)

    payload = {
        "task": "significance pipeline",
        "scope": ["overall", "experience"],
        "branches": branch_names,
        "outputs": outputs + ["significance_index.md", "research_questions_map.md", str(Path(guide_png).name)],
    }
    (out / "significance_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
