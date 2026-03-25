#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import ttest_rel

from plot_style import apply_bae_style

CONSTRUCTS = {
    "task_supportiveness": {
        "title_en": "Task Supportiveness in Space",
        "items": ["S1", "S2", "S3"],
        "complexity_filter": None,
    },
    "affective_behavioral": {
        "title_en": "Affective and Behavioral Outcomes",
        "items": ["S4", "S5"],
        "complexity_filter": None,
    },
    "functional_equipment": {
        "title_en": "Functional Equipment Appraisal",
        "items": ["B1", "B2", "B3"],
        "complexity_filter": 1,
    },
}

ITEM_LABELS = {
    "S1": "S1", "S2": "S2", "S3": "S3", "S4": "S4", "S5": "S5",
    "B1": "B1", "B2": "B2", "B3": "B3",
}

ITEM_COLORS = {
    "S1": "#2F6DA3",
    "S2": "#E6862A",
    "S3": "#2A9D8F",
    "S4": "#8E7DBE",
    "S5": "#D95F5F",
    "B1": "#2F6DA3",
    "B2": "#E6862A",
    "B3": "#2A9D8F",
}

GROUP_STYLES = {
    "High": {"linestyle": "-", "marker": "o", "label": "High group"},
    "Low": {"linestyle": "--", "marker": "s", "label": "Low group"},
}


def _prep_subject_means(df: pd.DataFrame, dv: str, group_col: str) -> pd.DataFrame:
    sub = df.dropna(subset=["SubjectID", "WWR", group_col, dv]).copy()
    if sub.empty:
        return pd.DataFrame()
    subj = (
        sub.groupby(["SubjectID", group_col, "WWR"], as_index=False)[dv]
        .mean()
        .rename(columns={dv: "score", group_col: "Group"})
    )
    out = (
        subj.groupby(["Group", "WWR"], as_index=False)
        .agg(mean=("score", "mean"), std=("score", "std"), n=("score", "count"))
    )
    out["se"] = out["std"] / np.sqrt(out["n"].clip(lower=1))
    out["DV"] = dv
    return out


def build_summary(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for construct_key, meta in CONSTRUCTS.items():
        src = df.copy()
        cfilter = meta.get("complexity_filter")
        if cfilter is not None:
            src = src[pd.to_numeric(src["Complexity"], errors="coerce") == cfilter].copy()
        for dv in meta["items"]:
            if dv not in src.columns:
                continue
            tmp = _prep_subject_means(src, dv, group_col)
            if tmp.empty:
                continue
            tmp["Construct"] = construct_key
            tmp["ConstructEN"] = meta["title_en"]
            rows.append(tmp)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    out["WWR"] = pd.to_numeric(out["WWR"], errors="coerce")
    return out.sort_values(["Construct", "DV", "Group", "WWR"]).reset_index(drop=True)


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


def build_pairwise_table(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    rows: list[dict] = []
    pairs = [(15, 45), (15, 75), (45, 75)]
    for construct_key, meta in CONSTRUCTS.items():
        src = df.copy()
        cfilter = meta.get("complexity_filter")
        if cfilter is not None:
            src = src[pd.to_numeric(src["Complexity"], errors="coerce") == cfilter].copy()
        for dv in meta["items"]:
            if dv not in src.columns:
                continue
            base = src.dropna(subset=["SubjectID", "WWR", group_col, dv]).copy()
            if base.empty:
                continue
            subj = (
                base.groupby(["SubjectID", group_col, "WWR"], as_index=False)[dv]
                .mean()
                .rename(columns={dv: "score", group_col: "Group"})
            )
            desc = subj.groupby(["Group", "WWR"], as_index=False).agg(mean=("score", "mean"), sd=("score", "std"), n=("score", "count"))
            for group in sorted(subj["Group"].dropna().astype(str).unique()):
                sg = subj[subj["Group"].astype(str) == group].copy()
                row = {
                    "ConstructEN": meta["title_en"],
                    "Measure": dv,
                    "Group": group,
                }
                for w in [15, 45, 75]:
                    ds = desc[(desc["Group"].astype(str) == group) & (pd.to_numeric(desc["WWR"], errors="coerce") == w)]
                    if ds.empty:
                        row[f"WWR{w} M ± SD"] = ""
                    else:
                        rr = ds.iloc[0]
                        row[f"WWR{w} M ± SD"] = f"{rr['mean']:.3f} ± {rr['sd']:.3f}" if pd.notna(rr['sd']) else f"{rr['mean']:.3f} ± NA"
                sig_parts = []
                for a, b in pairs:
                    piv = sg[sg["WWR"].isin([a, b])].pivot_table(index="SubjectID", columns="WWR", values="score", aggfunc="first").dropna()
                    if len(piv) < 3:
                        continue
                    res = ttest_rel(piv[a], piv[b], nan_policy="omit")
                    p = float(res.pvalue)
                    if p < 0.05:
                        sig_parts.append(f"WWR{a} - WWR{b} (p={p:.4g})")
                row["Significant pairwise contrasts"] = "; ".join(sig_parts) if sig_parts else "ns"
                rows.append(row)
    return pd.DataFrame(rows)


def plot_combined(summary: pd.DataFrame, out_png: Path, title: str) -> None:
    apply_bae_style()
    fig, axes = plt.subplots(1, 3, figsize=(16.2, 5.3), sharex=False, sharey=False)

    construct_order = list(CONSTRUCTS.keys())
    for ax, construct_key in zip(axes, construct_order):
        meta = CONSTRUCTS[construct_key]
        sub = summary[summary["Construct"] == construct_key].copy()
        if sub.empty:
            ax.set_visible(False)
            continue

        items_present = [dv for dv in meta["items"] if dv in set(sub["DV"])]
        for dv in items_present:
            item_df = sub[sub["DV"] == dv].copy()
            for group in ["High", "Low"]:
                g = item_df[item_df["Group"].astype(str) == group].sort_values("WWR")
                if g.empty:
                    continue
                color = ITEM_COLORS.get(dv, "#2F6DA3")
                style = GROUP_STYLES[group]
                ax.plot(
                    g["WWR"], g["mean"],
                    color=color,
                    linestyle=style["linestyle"],
                    marker=style["marker"],
                    linewidth=2.1,
                    markersize=5,
                    alpha=0.98,
                )
                if g["se"].notna().any():
                    ax.fill_between(g["WWR"], g["mean"] - g["se"], g["mean"] + g["se"], color=color, alpha=0.10)

        ax.set_title(meta['title_en'], fontsize=10.5)
        ax.set_xlabel("WWR")
        ax.set_xticks([15, 45, 75])
        ax.set_xlim(12, 78)
        ax.grid(axis="y", alpha=0.25)
        if construct_key == "functional_equipment":
            ax.text(
                0.98, 0.03,
                "B1–B3 shown for Complexity = 1 only",
                transform=ax.transAxes,
                ha="right", va="bottom", fontsize=8.2, color="#4A5568"
            )

    axes[0].set_ylabel("Mean score")

    item_handles = []
    item_labels = []
    for dv in ["S1", "S2", "S3", "S4", "S5", "B1", "B2", "B3"]:
        if dv in set(summary["DV"]):
            item_handles.append(Line2D([0], [0], color=ITEM_COLORS[dv], lw=2.4))
            item_labels.append(ITEM_LABELS[dv])

    group_handles = [
        Line2D([0], [0], color="#444444", lw=2.2, linestyle=GROUP_STYLES[g]["linestyle"], marker=GROUP_STYLES[g]["marker"])
        for g in ["High", "Low"]
    ]
    group_labels = ["High experience group", "Low experience group"]

    fig.legend(item_handles, item_labels, title="Items", loc="upper center", bbox_to_anchor=(0.35, 1.03), ncol=max(4, min(8, len(item_labels))), frameon=False)
    fig.legend(group_handles, group_labels, title="Group", loc="upper center", bbox_to_anchor=(0.84, 1.03), ncol=2, frameon=False)
    fig.suptitle(title, y=1.08, fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description="Plot construct-wise WWR trend lines split by high/low experience group")
    ap.add_argument("--long-csv", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("results/figure_construct_group"))
    ap.add_argument("--group-col", default="ExperienceGroup")
    args = ap.parse_args()

    df = pd.read_csv(args.long_csv)
    df["WWR"] = pd.to_numeric(df["WWR"], errors="coerce")
    if args.group_col not in df.columns:
        raise SystemExit(f"Missing group column: {args.group_col}")

    summary = build_summary(df, args.group_col)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    summary_csv = args.out_dir / "construct_group_wwr_summary.csv"
    summary.to_csv(summary_csv, index=False, encoding="utf-8-sig")

    pairwise = build_pairwise_table(df, args.group_col)
    pairwise_csv = args.out_dir / "construct_group_wwr_pairwise.csv"
    pairwise_xlsx = args.out_dir / "表5_按构念与高低组重做.xlsx"
    pairwise.to_csv(pairwise_csv, index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(pairwise_xlsx, engine="openpyxl") as writer:
        pairwise.to_excel(writer, sheet_name="表5重做", index=False)

    png = args.out_dir / "construct_group_wwr_combined.png"
    plot_combined(summary, png, "WWR trend by construct and experience group")

    payload = {
        "task": "construct_group_wwr_plot",
        "group_col": args.group_col,
        "outputs": [str(summary_csv), str(pairwise_csv), str(pairwise_xlsx), str(png)],
        "constructs": {
            k: {
                "title_en": v["title_en"],
                "items": v["items"],
                "complexity_filter": v["complexity_filter"],
            }
            for k, v in CONSTRUCTS.items()
        },
    }
    (args.out_dir / "construct_group_wwr_manifest.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
