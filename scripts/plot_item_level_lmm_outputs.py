#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from plot_style import apply_bae_style

PALETTE = {
    "blue": "#2F6DA3",
    "orange": "#E6862A",
    "teal": "#2A9D8F",
    "purple": "#8E7DBE",
    "ink": "#243447",
    "grid": "#E5ECF2",
}


def _sigstar(p: float | None) -> str:
    if p is None or pd.isna(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def _fmt(v, nd=3) -> str:
    if v is None or pd.isna(v):
        return "NA"
    return f"{float(v):.{nd}f}"


def _style(ax, grid_axis: str = "x") -> None:
    ax.spines["left"].set_color(PALETTE["ink"])
    ax.spines["bottom"].set_color(PALETTE["ink"])
    ax.spines["left"].set_linewidth(0.9)
    ax.spines["bottom"].set_linewidth(0.9)
    ax.tick_params(axis="both", colors=PALETTE["ink"], length=3)
    ax.grid(axis=grid_axis, color=PALETTE["grid"], alpha=0.75, linewidth=0.7)
    other = "x" if grid_axis == "y" else "y"
    ax.grid(axis=other, visible=False)


def _dv_family(dv: str) -> str:
    s = str(dv)
    if s.startswith("S"):
        return "S"
    if s.startswith("B"):
        return "B"
    if s.startswith("IPQ"):
        return "IPQ"
    return "Other"


def _effect_short(name: str) -> str:
    return {
        "WWR": "WWR",
        "Complexity": "Complexity",
        "ExperienceGroup": "Experience",
        "WWR:Complexity": "WWR×Comp",
        "WWR:ExperienceGroup": "WWR×Exp",
        "Complexity:ExperienceGroup": "Comp×Exp",
        "WWR:Complexity:ExperienceGroup": "WWR×Comp×Exp",
    }.get(str(name), str(name))


def _plot_type3_heatmaps(type3_df: pd.DataFrame, png_dir: Path) -> list[str]:
    made: list[str] = []
    if type3_df.empty:
        return made

    work = type3_df.copy()
    work["DV"] = work["DV"].astype(str)
    work["EffectShort"] = work["Effect"].map(_effect_short)
    p_col = "p_fdr" if "p_fdr" in work.columns and work["p_fdr"].notna().any() else "p"
    work["plot_p"] = pd.to_numeric(work[p_col], errors="coerce")
    work["minuslog10p"] = -np.log10(work["plot_p"].clip(lower=1e-12))
    work["annot"] = work["plot_p"].map(lambda p: "" if pd.isna(p) else f"p={p:.3f}{_sigstar(p)}")

    for family in ["S", "B", "IPQ", "Other"]:
        sub = work[work["DV"].map(_dv_family) == family].copy()
        if sub.empty:
            continue
        mat = sub.pivot_table(index="EffectShort", columns="DV", values="minuslog10p", aggfunc="first")
        ann = sub.pivot_table(index="EffectShort", columns="DV", values="annot", aggfunc="first")
        if mat.empty:
            continue
        ordered_effects = [e for e in ["WWR", "Complexity", "Experience", "WWR×Comp", "WWR×Exp", "Comp×Exp", "WWR×Comp×Exp"] if e in mat.index]
        mat = mat.reindex(index=ordered_effects)
        ann = ann.reindex(index=ordered_effects)
        fig, ax = plt.subplots(figsize=(max(6.8, 1.0 * len(mat.columns) + 2.8), max(3.8, 0.62 * len(mat.index) + 1.8)))
        sns.heatmap(
            mat,
            cmap=sns.blend_palette(["#F7FBFF", PALETTE["blue"], PALETTE["orange"]], as_cmap=True),
            annot=ann,
            fmt="",
            linewidths=0.7,
            linecolor="#E2EAF1",
            cbar_kws={"label": f"-log10({p_col})"},
            annot_kws={"fontsize": 8.0},
            ax=ax,
        )
        ax.set_title(f"Item-level Type III significance — {family} items", pad=8)
        ax.set_xlabel("Item / dimension")
        ax.set_ylabel("Effect")
        ax.tick_params(axis="x", rotation=25)
        ax.tick_params(axis="y", rotation=0)
        path = png_dir / f"item_level_type3_heatmap_{family.lower()}.png"
        fig.savefig(path, dpi=260)
        plt.close(fig)
        made.append(str(path))

    mat_all = work.pivot_table(index="EffectShort", columns="DV", values="minuslog10p", aggfunc="first")
    ann_all = work.pivot_table(index="EffectShort", columns="DV", values="annot", aggfunc="first")
    if not mat_all.empty:
        ordered_effects = [e for e in ["WWR", "Complexity", "Experience", "WWR×Comp", "WWR×Exp", "Comp×Exp", "WWR×Comp×Exp"] if e in mat_all.index]
        mat_all = mat_all.reindex(index=ordered_effects)
        ann_all = ann_all.reindex(index=ordered_effects)
        fig, ax = plt.subplots(figsize=(max(11.0, 0.72 * len(mat_all.columns) + 4.0), max(4.0, 0.6 * len(mat_all.index) + 1.9)))
        sns.heatmap(
            mat_all,
            cmap=sns.blend_palette(["#F7FBFF", PALETTE["blue"], PALETTE["orange"]], as_cmap=True),
            annot=ann_all,
            fmt="",
            linewidths=0.6,
            linecolor="#E2EAF1",
            cbar_kws={"label": f"-log10({p_col})"},
            annot_kws={"fontsize": 7.2},
            ax=ax,
        )
        ax.set_title("Item-level Type III significance — all items", pad=8)
        ax.set_xlabel("Item / dimension")
        ax.set_ylabel("Effect")
        ax.tick_params(axis="x", rotation=35)
        path = png_dir / "item_level_type3_heatmap_all.png"
        fig.savefig(path, dpi=260)
        plt.close(fig)
        made.append(str(path))
    return made


def _plot_sig_count_summary(type3_df: pd.DataFrame, png_dir: Path) -> str | None:
    if type3_df.empty:
        return None
    work = type3_df.copy()
    p_col = "p_fdr" if "p_fdr" in work.columns and work["p_fdr"].notna().any() else "p"
    work["sig"] = pd.to_numeric(work[p_col], errors="coerce") < 0.05
    summ = work.groupby("Effect", as_index=False).agg(sig_items=("sig", "sum"), total_items=("DV", "nunique"))
    if summ.empty:
        return None
    summ["EffectShort"] = summ["Effect"].map(_effect_short)
    summ = summ.sort_values(["sig_items", "EffectShort"], ascending=[True, True])
    fig, ax = plt.subplots(figsize=(8.0, max(4.0, 0.55 * len(summ) + 1.2)))
    y = np.arange(len(summ))
    ax.barh(y, summ["sig_items"], color=PALETTE["orange"], alpha=0.92)
    ax.set_yticks(y)
    ax.set_yticklabels(summ["EffectShort"])
    ax.set_xlabel("# items/dimensions with p < .05")
    ax.set_title("How many items show each effect", pad=8)
    _style(ax, grid_axis="x")
    for yy, (_, r) in zip(y, summ.iterrows()):
        ax.text(float(r["sig_items"]) + 0.08, yy, f"{int(r['sig_items'])}/{int(r['total_items'])}", va="center", ha="left", fontsize=8.2, color=PALETTE["ink"])
    path = png_dir / "item_level_effect_sig_count.png"
    fig.savefig(path, dpi=260)
    plt.close(fig)
    return str(path)


def _plot_pairwise_summary(pair_df: pd.DataFrame, png_dir: Path) -> str | None:
    if pair_df.empty or "Spec" not in pair_df.columns:
        return None
    work = pair_df.copy()
    work["p.value"] = pd.to_numeric(work["p.value"], errors="coerce")
    work = work.dropna(subset=["p.value", "Spec", "DV"])
    if work.empty:
        return None
    summ = work.groupby(["DV", "Spec"], as_index=False).agg(sig_pairs=("p.value", lambda s: int((s < 0.05).sum())), min_p=("p.value", "min"))
    if summ.empty:
        return None
    plot_df = summ.pivot_table(index="Spec", columns="DV", values="sig_pairs", aggfunc="first").fillna(0)
    annot = plot_df.copy().astype(int).astype(str)
    fig, ax = plt.subplots(figsize=(max(9.0, 0.8 * len(plot_df.columns) + 4.0), max(4.6, 0.45 * len(plot_df.index) + 1.8)))
    sns.heatmap(
        plot_df,
        cmap=sns.light_palette(PALETTE["orange"], as_cmap=True),
        annot=annot,
        fmt="",
        linewidths=0.6,
        linecolor="#E9EEF3",
        cbar_kws={"label": "# significant pairwise contrasts"},
        annot_kws={"fontsize": 7.3},
        ax=ax,
    )
    ax.set_title("Item-level follow-up comparisons by item and question-group", pad=8)
    ax.set_xlabel("Item / dimension")
    ax.set_ylabel("Question-group follow-up")
    ax.tick_params(axis="x", rotation=30)
    ax.tick_params(axis="y", rotation=0)
    path = png_dir / "item_level_pairwise_sig_summary.png"
    fig.savefig(path, dpi=260)
    plt.close(fig)
    return str(path)


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot item-level LMM outputs from exported CSV files")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    apply_bae_style()
    out = args.out_dir
    csv_dir = out / "csv"
    png_dir = out / "png"
    json_dir = out / "json"
    png_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)

    type3_path = csv_dir / "item_level_lmm_type3_fixed_effects_fdr.csv"
    if not type3_path.exists():
        type3_path = csv_dir / "item_level_lmm_type3_fixed_effects.csv"
    pair_path = csv_dir / "item_level_lmm_pairwise.csv"

    type3_df = pd.read_csv(type3_path) if type3_path.exists() else pd.DataFrame()
    pair_df = pd.read_csv(pair_path) if pair_path.exists() else pd.DataFrame()

    outputs: list[str] = []
    outputs.extend([str(Path(p).relative_to(out)) for p in _plot_type3_heatmaps(type3_df, png_dir)])
    p = _plot_sig_count_summary(type3_df, png_dir)
    if p:
        outputs.append(str(Path(p).relative_to(out)))
    p = _plot_pairwise_summary(pair_df, png_dir)
    if p:
        outputs.append(str(Path(p).relative_to(out)))

    payload = {"task": "plot_item_level_lmm_outputs", "outputs": outputs}
    (json_dir / "item_level_lmm_png_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
