#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import argparse
import json

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_rel

from plot_style import apply_bae_style

CONSTRUCTS = {
    "task_supportiveness": {
        "title_en": "Task Supportiveness in Space",
        "items": ["S1", "S2", "S3"],
        "complexity_levels": [0, 1],
    },
    "affective_behavioral": {
        "title_en": "Affective and Behavioral Outcomes",
        "items": ["S4", "S5"],
        "complexity_levels": [0, 1],
    },
    "functional_equipment": {
        "title_en": "Functional Equipment Appraisal",
        "items": ["B1", "B2", "B3"],
        "complexity_levels": [1],
    },
}

PALETTE = {"High": "#2F6DA3", "Low": "#E6862A"}


def prep_long(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    rows = []
    for construct_key, meta in CONSTRUCTS.items():
        for dv in meta["items"]:
            if dv not in df.columns:
                continue
            sub = df.dropna(subset=[group_col, dv]).copy()
            if dv.startswith("B"):
                sub = sub[pd.to_numeric(sub["Complexity"], errors="coerce") == 1].copy()
            else:
                sub = sub[pd.to_numeric(sub["Complexity"], errors="coerce").isin(meta["complexity_levels"])]
            if sub.empty:
                continue
            tmp = sub[["SubjectID", group_col, "Complexity", dv]].copy()
            tmp = tmp.rename(columns={group_col: "Group", dv: "Score"})
            tmp["Measure"] = dv
            tmp["Construct"] = construct_key
            tmp["ConstructEN"] = meta["title_en"]
            rows.append(tmp)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    out["Complexity"] = pd.to_numeric(out["Complexity"], errors="coerce")
    out["ComplexityLabel"] = out["Complexity"].map({0: "C0", 1: "C1"})
    return out


def build_table(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    rows = []
    for construct_key, meta in CONSTRUCTS.items():
        for dv in meta["items"]:
            if dv not in df.columns:
                continue
            base = df.dropna(subset=["SubjectID", group_col, dv]).copy()
            if dv.startswith("B"):
                base = base[pd.to_numeric(base["Complexity"], errors="coerce") == 1].copy()
            if base.empty:
                continue
            subj = (
                base.groupby(["SubjectID", group_col, "Complexity"], as_index=False)[dv]
                .mean()
                .rename(columns={group_col: "Group", dv: "Score"})
            )
            desc = subj.groupby(["Group", "Complexity"], as_index=False).agg(mean=("Score", "mean"), sd=("Score", "std"), n=("Score", "count"))
            for group in sorted(subj["Group"].dropna().astype(str).unique()):
                row = {
                    "ConstructEN": meta["title_en"],
                    "Measure": dv,
                    "Group": group,
                }
                for c in [0, 1]:
                    ds = desc[(desc["Group"].astype(str) == group) & (pd.to_numeric(desc["Complexity"], errors="coerce") == c)]
                    if ds.empty:
                        row[f"C{c} M ± SD"] = ""
                    else:
                        rr = ds.iloc[0]
                        row[f"C{c} M ± SD"] = f"{rr['mean']:.3f} ± {rr['sd']:.3f}" if pd.notna(rr['sd']) else f"{rr['mean']:.3f} ± NA"
                if dv.startswith("B"):
                    row["Mean difference (C0–C1)"] = "N/A (B items are C1-only)"
                    row["p"] = "N/A"
                    row["Effect summary"] = "B1–B3 are only available under Complexity=1."
                else:
                    sg = subj[subj['Group'].astype(str) == group].copy()
                    piv = sg.pivot_table(index='SubjectID', columns='Complexity', values='Score', aggfunc='first').dropna()
                    if len(piv) >= 3 and 0 in piv.columns and 1 in piv.columns:
                        res = ttest_rel(piv[0], piv[1], nan_policy='omit')
                        p = float(res.pvalue)
                        diff = float((piv[0] - piv[1]).mean())
                        row['Mean difference (C0–C1)'] = f"{diff:.3f}"
                        row['p'] = f"{p:.4g}"
                        if p < 0.05:
                            direction = 'C0 > C1' if diff > 0 else 'C0 < C1'
                            row['Effect summary'] = f"significant ({direction})"
                        else:
                            row['Effect summary'] = 'ns'
                    else:
                        row['Mean difference (C0–C1)'] = ''
                        row['p'] = ''
                        row['Effect summary'] = 'insufficient paired data'
                rows.append(row)
    return pd.DataFrame(rows)


def plot_panels(long_df: pd.DataFrame, out_png: Path, title: str) -> None:
    apply_bae_style()
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.4), sharey=False)

    for ax, construct_key in zip(axes, CONSTRUCTS.keys()):
        meta = CONSTRUCTS[construct_key]
        sub = long_df[long_df['Construct'] == construct_key].copy()
        if sub.empty:
            ax.set_visible(False)
            continue

        sns.boxplot(
            data=sub,
            x='Measure',
            y='Score',
            hue='Group',
            palette=PALETTE,
            width=0.7,
            linewidth=1.0,
            ax=ax,
        )
        if construct_key != 'functional_equipment':
            sns.stripplot(
                data=sub,
                x='Measure',
                y='Score',
                hue='Group',
                dodge=True,
                palette=PALETTE,
                size=2.2,
                alpha=0.28,
                linewidth=0,
                ax=ax,
            )
        else:
            sns.stripplot(
                data=sub,
                x='Measure',
                y='Score',
                hue='Group',
                dodge=True,
                palette=PALETTE,
                size=2.2,
                alpha=0.28,
                linewidth=0,
                ax=ax,
            )

        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend_.remove()
        ax.set_title(meta['title_en'], fontsize=10.5)
        ax.set_xlabel('')
        ax.set_ylabel('Score')
        ax.grid(axis='y', alpha=0.25)
        if construct_key == 'functional_equipment':
            ax.text(0.98, 0.03, 'B1–B3 are C1-only measures', transform=ax.transAxes, ha='right', va='bottom', fontsize=8.2, color='#4A5568')

    legend_handles, legend_labels = axes[0].get_legend_handles_labels()
    if legend_handles:
        uniq = []
        seen = set()
        for h, l in zip(legend_handles, legend_labels):
            if l in ('High', 'Low') and l not in seen:
                uniq.append((h, l))
                seen.add(l)
        if uniq:
            fig.legend([x[0] for x in uniq], [x[1] for x in uniq], title='Experience group', loc='upper center', bbox_to_anchor=(0.5, 1.03), ncol=2, frameon=False)

    fig.suptitle(title, y=1.08, fontsize=12, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description='Plot construct-wise complexity boxplots split by high/low experience group')
    ap.add_argument('--long-csv', type=Path, required=True)
    ap.add_argument('--out-dir', type=Path, required=True)
    ap.add_argument('--group-col', default='ExperienceGroup')
    args = ap.parse_args()

    df = pd.read_csv(args.long_csv)
    long_df = prep_long(df, args.group_col)
    table_df = build_table(df, args.group_col)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    long_csv = args.out_dir / 'construct_group_complexity_long.csv'
    table_csv = args.out_dir / 'construct_group_complexity_table.csv'
    table_xlsx = args.out_dir / '表6_按构念与经验高低组重做.xlsx'
    png = args.out_dir / '图2_按构念与经验高低组的Complexity箱型图.png'

    long_df.to_csv(long_csv, index=False, encoding='utf-8-sig')
    table_df.to_csv(table_csv, index=False, encoding='utf-8-sig')
    with pd.ExcelWriter(table_xlsx, engine='openpyxl') as writer:
        table_df.to_excel(writer, sheet_name='表6重做', index=False)

    plot_panels(long_df, png, 'Complexity comparison by construct and experience group')

    payload = {
        'task': 'construct_group_complexity_boxplot',
        'group_col': args.group_col,
        'outputs': [str(long_csv), str(table_csv), str(table_xlsx), str(png)],
    }
    (args.out_dir / 'construct_group_complexity_manifest.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
