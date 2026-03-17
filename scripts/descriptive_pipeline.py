#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kurtosis, sem, shapiro, skew, t

QC_EXCLUDE = "孙校聪,康少勇,张钰鹏,杨可,洪婷婷,陈韬,高梓楠,赵国宏"
S_COLS = ["S1", "S2", "S3", "S4", "S5"]
B_COLS = ["B1", "B2", "B3", "Bmean"]
IPQ_COLS = ["IPQ1", "IPQ2", "IPQ3", "IPQ4", "IPQ5", "IPQ6", "IPQ_mean"]
LIKERT_LIMS = (1, 10)
LIKERT_TICKS = list(range(1, 11))

PALETTE = {
    "blue": "#6F97BD",
    "orange": "#E3A86F",
    "ink": "#243447",
    "grid": "#E5ECF2",
    "muted": "#6B7C8F",
    "light_blue": "#E8F1F8",
    "light_orange": "#FAEBDD",
    "gray": "#A8B2BC",
}

SINGLE_GROUP_FILL = PALETTE["light_blue"]
SINGLE_GROUP_EDGE = PALETTE["blue"]
ANNOTATION_TEXT = PALETTE["ink"]
STRIP_FALLBACK = "#7A8896"


def apply_publication_style() -> None:
    sns.set_theme(style="white", context="notebook")
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": PALETTE["ink"],
        "axes.labelcolor": PALETTE["ink"],
        "axes.titlecolor": PALETTE["ink"],
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.9,
        "axes.grid": False,
        "grid.color": PALETTE["grid"],
        "grid.alpha": 0.7,
        "grid.linewidth": 0.7,
        "font.family": "DejaVu Sans",
        "font.size": 9.5,
        "axes.titlesize": 11.5,
        "axes.titleweight": "semibold",
        "axes.labelsize": 10,
        "legend.fontsize": 8.8,
        "legend.title_fontsize": 9,
        "xtick.labelsize": 8.8,
        "ytick.labelsize": 8.8,
        "xtick.color": PALETTE["ink"],
        "ytick.color": PALETTE["ink"],
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "savefig.dpi": 320,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.08,
        "legend.frameon": False,
        "patch.linewidth": 0.9,
        "figure.titleweight": "semibold",
    })


def soften_axes(ax, grid_axis: str = "y") -> None:
    ax.spines["left"].set_color(PALETTE["ink"])
    ax.spines["bottom"].set_color(PALETTE["ink"])
    ax.spines["left"].set_linewidth(0.9)
    ax.spines["bottom"].set_linewidth(0.9)
    ax.tick_params(axis="both", colors=PALETTE["ink"], length=3)
    ax.grid(axis=grid_axis, color=PALETTE["grid"], alpha=0.7, linewidth=0.7)
    other = "x" if grid_axis == "y" else "y"
    ax.grid(axis=other, visible=False)


def _exclude_subjects(df: pd.DataFrame, text: str) -> pd.DataFrame:
    if not text or "SubjectID" not in df.columns:
        return df
    names = [x.strip() for x in str(text).split(",") if x.strip()]
    if not names:
        return df
    sid = df["SubjectID"].astype(str).str.strip()
    return df.loc[~sid.isin(set(names))].copy()


def _ci95(z: pd.Series) -> tuple[float, float]:
    zz = pd.to_numeric(z, errors="coerce").dropna()
    n = int(len(zz))
    if n < 2:
        return np.nan, np.nan
    m = float(zz.mean())
    se = float(sem(zz, nan_policy="omit"))
    h = float(t.ppf(0.975, df=n - 1) * se)
    return m - h, m + h


def _norm_p(z: pd.Series) -> float:
    zz = pd.to_numeric(z, errors="coerce").dropna()
    if len(zz) < 3 or len(zz) > 5000:
        return np.nan
    try:
        return float(shapiro(zz).pvalue)
    except Exception:
        return np.nan


def _desc_stats(z: pd.Series, subject_ids: pd.Series | None = None) -> dict[str, float]:
    zz = pd.to_numeric(z, errors="coerce")
    valid = zz.notna()
    zz = zz[valid]
    n_obs = int(len(zz))
    n_subjects = int(subject_ids[valid].astype(str).str.strip().nunique()) if subject_ids is not None else n_obs
    ci_low, ci_high = _ci95(zz)
    return {
        "n": n_subjects,
        "n_obs": n_obs,
        "mean": float(zz.mean()) if n_obs else np.nan,
        "sd": float(zz.std(ddof=1)) if n_obs > 1 else np.nan,
        "median": float(zz.median()) if n_obs else np.nan,
        "min": float(zz.min()) if n_obs else np.nan,
        "max": float(zz.max()) if n_obs else np.nan,
        "skewness": float(skew(zz, bias=False)) if n_obs > 2 else np.nan,
        "kurtosis": float(kurtosis(zz, fisher=True, bias=False)) if n_obs > 3 else np.nan,
        "ci95_low": ci_low,
        "ci95_high": ci_high,
        "shapiro_p": _norm_p(zz),
    }


def _desc_table(df: pd.DataFrame, cols: list[str], group_cols: list[str] | None = None) -> pd.DataFrame:
    rows = []
    use_cols = [c for c in cols if c in df.columns]
    if not use_cols:
        return pd.DataFrame()

    group_cols = group_cols or []
    if group_cols:
        iter_items = list(df.groupby(group_cols, dropna=False))
    else:
        iter_items = [((), df.copy())]

    for key, sub in iter_items:
        if not isinstance(key, tuple):
            key = (key,)
        key_map = dict(zip(group_cols, key)) if group_cols else {"Group": "ALL"}
        for c in use_cols:
            rows.append({**key_map, "DV": c, **_desc_stats(sub[c], sub["SubjectID"] if "SubjectID" in sub.columns else None)})
    return pd.DataFrame(rows)


def _subject_level_ipq(df: pd.DataFrame) -> pd.DataFrame:
    if "SubjectID" not in df.columns:
        return df.copy()
    return df.groupby("SubjectID", as_index=False).first()


def _fmt_num(v: float | None, nd: int = 2) -> str:
    if v is None or pd.isna(v):
        return "NA"
    return f"{float(v):.{nd}f}"


def _normalize_category_value(v):
    if pd.isna(v):
        return np.nan
    try:
        fv = float(v)
        if fv.is_integer():
            return str(int(fv))
        return str(fv)
    except Exception:
        return str(v)


def _get_order(series: pd.Series) -> list[str]:
    vals = [_normalize_category_value(v) for v in series.dropna().unique().tolist()]
    vals = [v for v in vals if pd.notna(v)]
    try:
        return sorted(vals, key=lambda x: float(x))
    except Exception:
        return sorted(vals, key=lambda x: str(x))


def _get_palette_for_levels(levels: list[str]) -> dict[str, str]:
    if not levels:
        return {}
    if len(levels) == 1:
        return {str(levels[0]): SINGLE_GROUP_EDGE}
    if len(levels) == 2:
        return {str(levels[0]): PALETTE["blue"], str(levels[1]): PALETTE["orange"]}
    fallback = [
        PALETTE["blue"],
        PALETTE["orange"],
        "#8FB7A1",
        "#A99AC6",
        "#D3A6C6",
        PALETTE["gray"],
    ]
    return {str(level): fallback[i % len(fallback)] for i, level in enumerate(levels)}


def _resolve_plot_groups(sub: pd.DataFrame, xcol: str | None, hue: str | None) -> tuple[str | None, list[str] | None, str | None, list[str] | None]:
    x_arg = xcol if xcol and xcol in sub.columns else None
    hue_arg = hue if hue and hue in sub.columns and hue != x_arg else None
    x_order = _get_order(sub[x_arg]) if x_arg else None
    hue_order = _get_order(sub[hue_arg]) if hue_arg else None
    return x_arg, x_order, hue_arg, hue_order


def _cluster_center(index: int, hue_index: int, n_hue: int) -> float:
    if n_hue <= 1:
        return float(index)
    offsets = np.linspace(-0.20, 0.20, n_hue)
    return float(index) + float(offsets[hue_index])


def _group_summary_rows(sub: pd.DataFrame, dv: str, xcol: str | None, hue: str | None) -> list[dict]:
    x_arg, x_order, hue_arg, hue_order = _resolve_plot_groups(sub, xcol, hue)
    rows: list[dict] = []

    if x_arg:
        for i, xv in enumerate(x_order or []):
            subx = sub[sub[x_arg].astype(str) == str(xv)]
            if hue_arg:
                for j, hv in enumerate(hue_order or []):
                    sg = subx[subx[hue_arg].astype(str) == str(hv)]
                    vals = pd.to_numeric(sg[dv], errors="coerce").dropna()
                    if vals.empty:
                        continue
                    low, high = _ci95(vals)
                    rows.append({
                        "x": _cluster_center(i, j, len(hue_order or [])),
                        "mean": float(vals.mean()),
                        "ci_low": low,
                        "ci_high": high,
                        "n": int(sg["SubjectID"].astype(str).str.strip().nunique()) if "SubjectID" in sg.columns else int(len(vals)),
                        "color": _get_palette_for_levels(hue_order or []).get(str(hv), SINGLE_GROUP_EDGE),
                    })
            else:
                vals = pd.to_numeric(subx[dv], errors="coerce").dropna()
                if vals.empty:
                    continue
                low, high = _ci95(vals)
                rows.append({
                    "x": float(i),
                    "mean": float(vals.mean()),
                    "ci_low": low,
                    "ci_high": high,
                    "n": int(subx["SubjectID"].astype(str).str.strip().nunique()) if "SubjectID" in subx.columns else int(len(vals)),
                    "color": SINGLE_GROUP_EDGE,
                })
        return rows

    vals = pd.to_numeric(sub[dv], errors="coerce").dropna()
    if vals.empty:
        return rows
    low, high = _ci95(vals)
    rows.append({
        "x": 0.0,
        "mean": float(vals.mean()),
        "ci_low": low,
        "ci_high": high,
        "n": int(sub["SubjectID"].astype(str).str.strip().nunique()) if "SubjectID" in sub.columns else int(len(vals)),
        "color": SINGLE_GROUP_EDGE,
    })
    return rows


def _annotate_group_summaries(ax, summary_rows: list[dict], dv: str) -> None:
    if not summary_rows:
        return
    ymin, ymax = ax.get_ylim()
    span = max(float(ymax) - float(ymin), 1.0)
    placed: list[tuple[float, float]] = []
    for idx, row in enumerate(summary_rows):
        anchor_y = row["ci_high"] if pd.notna(row["ci_high"]) else row["mean"]
        text_y = float(anchor_y) + 0.04 * span
        for px, py in placed:
            if abs(float(row["x"]) - px) < 0.30 and abs(text_y - py) < 0.08 * span:
                text_y += 0.06 * span
        ci_text = f"[{_fmt_num(row['ci_low'])}, {_fmt_num(row['ci_high'])}]" if pd.notna(row["ci_low"]) and pd.notna(row["ci_high"]) else "[NA, NA]"
        label = f"M={_fmt_num(row['mean'])}\n95% CI {ci_text}\nn={row['n']}"
        ax.annotate(
            label,
            xy=(row["x"], anchor_y),
            xytext=(0, 6 if idx % 2 == 0 else 8),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=6.7,
            color=ANNOTATION_TEXT,
            bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="none", alpha=0.94),
            zorder=8,
        )
        placed.append((float(row["x"]), text_y))


def _dedupe_legend(ax, title: str | None = None) -> None:
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return
    seen = set()
    new_handles = []
    new_labels = []
    for handle, label in zip(handles, labels):
        if not label or label in seen:
            continue
        seen.add(label)
        new_handles.append(handle)
        new_labels.append(label)
    if new_handles:
        ax.legend(
            new_handles,
            new_labels,
            loc="upper left",
            bbox_to_anchor=(0.0, 1.02),
            ncol=min(3, len(new_labels)),
            title=title,
            frameon=False,
            borderaxespad=0,
        )


def _set_likert_axis(ax, dv: str) -> None:
    dv_upper = dv.upper()
    if dv_upper.startswith("S") or dv_upper.startswith("B") or dv_upper.startswith("IPQ"):
        ax.set_ylim(*LIKERT_LIMS)
        ax.set_yticks(LIKERT_TICKS)


def _finalize_axis(ax, dv: str, xcol: str | None, hue: str | None, title: str) -> None:
    ax.set_title(title, pad=10)
    ax.set_xlabel(xcol if xcol else "")
    ax.set_ylabel(dv)
    ax.tick_params(axis="x", rotation=0)
    _set_likert_axis(ax, dv)
    soften_axes(ax)
    if hue and hue in [t.get_text() for t in ax.get_legend().texts] if ax.get_legend() else []:
        pass


def _plot_main_variant(ax, sub: pd.DataFrame, dv: str, xcol: str | None, hue: str | None) -> None:
    x_arg, x_order, hue_arg, hue_order = _resolve_plot_groups(sub, xcol, hue)
    hue_levels = hue_order or []
    palette = _get_palette_for_levels(hue_levels)

    if x_arg:
        box_palette = palette if hue_arg else SINGLE_GROUP_FILL
        sns.boxplot(
            data=sub,
            x=x_arg,
            y=dv,
            hue=hue_arg,
            order=x_order,
            hue_order=hue_order,
            palette=box_palette,
            width=0.62,
            dodge=bool(hue_arg),
            fliersize=0,
            linewidth=0.95,
            saturation=1.0,
            boxprops=dict(alpha=0.42),
            whiskerprops=dict(color=PALETTE["muted"], linewidth=0.9),
            capprops=dict(color=PALETTE["muted"], linewidth=0.9),
            medianprops=dict(color=PALETTE["ink"], linewidth=1.15),
            ax=ax,
        )
        if hue_arg:
            sns.stripplot(
                data=sub,
                x=x_arg,
                y=dv,
                hue=hue_arg,
                order=x_order,
                hue_order=hue_order,
                palette=palette,
                dodge=True,
                jitter=0.10,
                size=2.7,
                alpha=0.25,
                edgecolor="white",
                linewidth=0.28,
                ax=ax,
            )
        else:
            sns.stripplot(
                data=sub,
                x=x_arg,
                y=dv,
                color=STRIP_FALLBACK,
                jitter=0.10,
                size=2.7,
                alpha=0.22,
                edgecolor="white",
                linewidth=0.28,
                ax=ax,
            )
    else:
        sns.boxplot(
            data=sub,
            y=dv,
            color=SINGLE_GROUP_FILL,
            width=0.55,
            fliersize=0,
            linewidth=0.95,
            boxprops=dict(alpha=0.48),
            whiskerprops=dict(color=PALETTE["muted"], linewidth=0.9),
            capprops=dict(color=PALETTE["muted"], linewidth=0.9),
            medianprops=dict(color=PALETTE["ink"], linewidth=1.15),
            ax=ax,
        )
        sns.stripplot(
            data=sub,
            y=dv,
            color=STRIP_FALLBACK,
            jitter=0.08,
            size=2.7,
            alpha=0.22,
            edgecolor="white",
            linewidth=0.28,
            ax=ax,
        )

    summaries = _group_summary_rows(sub, dv, xcol, hue)
    for row in summaries:
        mean = row["mean"]
        low = row["ci_low"]
        high = row["ci_high"]
        if pd.notna(low) and pd.notna(high):
            ax.errorbar(
                [row["x"]],
                [mean],
                yerr=[[mean - low], [high - mean]],
                fmt="D",
                color=row["color"],
                markersize=5.2,
                capsize=4,
                lw=1.1,
                zorder=7,
            )
        else:
            ax.scatter([row["x"]], [mean], marker="D", s=26, color=row["color"], zorder=7)
    _annotate_group_summaries(ax, summaries, dv)
    _dedupe_legend(ax, title=hue_arg)


def _plot_box_variant(ax, sub: pd.DataFrame, dv: str, xcol: str | None, hue: str | None) -> None:
    x_arg, x_order, hue_arg, hue_order = _resolve_plot_groups(sub, xcol, hue)
    palette = _get_palette_for_levels(hue_order or [])

    if x_arg:
        sns.boxplot(
            data=sub,
            x=x_arg,
            y=dv,
            hue=hue_arg,
            order=x_order,
            hue_order=hue_order,
            palette=palette if hue_arg else SINGLE_GROUP_FILL,
            width=0.62,
            dodge=bool(hue_arg),
            fliersize=0,
            linewidth=0.98,
            saturation=1.0,
            boxprops=dict(alpha=0.55),
            whiskerprops=dict(color=PALETTE["muted"], linewidth=0.92),
            capprops=dict(color=PALETTE["muted"], linewidth=0.92),
            medianprops=dict(color=PALETTE["ink"], linewidth=1.2),
            ax=ax,
        )
    else:
        sns.boxplot(
            data=sub,
            y=dv,
            color=SINGLE_GROUP_FILL,
            width=0.55,
            fliersize=0,
            linewidth=0.98,
            boxprops=dict(alpha=0.60),
            whiskerprops=dict(color=PALETTE["muted"], linewidth=0.92),
            capprops=dict(color=PALETTE["muted"], linewidth=0.92),
            medianprops=dict(color=PALETTE["ink"], linewidth=1.2),
            ax=ax,
        )
    _dedupe_legend(ax, title=hue_arg)


def _publication_title(dv: str, xcol: str | None, hue: str | None) -> str:
    if xcol and hue and xcol != hue:
        return f"{dv} by {xcol} and {hue}"
    if xcol:
        return f"{dv} across {xcol}"
    if hue:
        return f"{dv} by {hue}"
    return f"{dv} distribution"


def _plot_distribution_panels(df: pd.DataFrame, cols: list[str], out_dir: Path, prefix: str, hue: str | None = None, xcol: str | None = None) -> list[str]:
    made: list[str] = []
    use_cols = [c for c in cols if c in df.columns]
    if not use_cols:
        return made
    out_dir.mkdir(parents=True, exist_ok=True)

    for dv in use_cols:
        sub = df.dropna(subset=[dv]).copy()
        if sub.empty:
            continue

        if xcol and xcol in sub.columns:
            sub[xcol] = sub[xcol].map(_normalize_category_value)
        if hue and hue in sub.columns:
            sub[hue] = sub[hue].map(_normalize_category_value)

        title = _publication_title(dv, xcol, hue if hue != xcol else None)

        fig, ax = plt.subplots(figsize=(7.4, 4.5))
        _plot_main_variant(ax, sub, dv, xcol, hue)
        _finalize_axis(ax, dv, xcol, hue, title)
        fig.tight_layout()
        path = out_dir / f"{prefix}_{dv}_main.png"
        fig.savefig(path, dpi=320, bbox_inches="tight")
        plt.close(fig)
        made.append(str(path))

        fig2, ax2 = plt.subplots(figsize=(7.0, 4.3))
        _plot_box_variant(ax2, sub, dv, xcol, hue)
        _finalize_axis(ax2, dv, xcol, hue, title)
        fig2.tight_layout()
        path2 = out_dir / f"{prefix}_{dv}_box.png"
        fig2.savefig(path2, dpi=320, bbox_inches="tight")
        plt.close(fig2)
        made.append(str(path2))

    return made


def main():
    ap = argparse.ArgumentParser(description="Descriptive-only pipeline: overall + experience")
    ap.add_argument("--long-csv", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("results/descriptive"))
    ap.add_argument("--with-qc", action="store_true", help="Also export QC-excluded outputs")
    args = ap.parse_args()

    apply_publication_style()

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.long_csv)

    branches = [("raw", "")]
    if args.with_qc:
        branches.append(("qc", QC_EXCLUDE))

    outputs: list[str] = []

    for branch, exclude in branches:
        base = out / branch
        base.mkdir(parents=True, exist_ok=True)
        x = _exclude_subjects(df, exclude)

        overall_dir = base / "overall"
        overall_dir.mkdir(parents=True, exist_ok=True)
        csv_dir_overall = overall_dir / "csv"
        png_dir_overall = overall_dir / "png"
        csv_dir_overall.mkdir(parents=True, exist_ok=True)
        png_dir_overall.mkdir(parents=True, exist_ok=True)
        fig_dir_overall = png_dir_overall

        s_overall = _desc_table(x, S_COLS)
        s_overall_wwr = _desc_table(x, S_COLS, ["WWR"]) if "WWR" in x.columns else pd.DataFrame()
        s_overall_cx = _desc_table(x, S_COLS, ["Complexity"]) if "Complexity" in x.columns else pd.DataFrame()

        b_src = x[x["Complexity"].astype(str).isin(["1", "1.0"])].copy() if "Complexity" in x.columns else x
        b_overall = _desc_table(b_src, B_COLS)
        b_overall_wwr = _desc_table(b_src, B_COLS, ["WWR"]) if "WWR" in b_src.columns else pd.DataFrame()

        ipq_subj = _subject_level_ipq(x)
        ipq_overall = _desc_table(ipq_subj, IPQ_COLS)

        s_overall.to_csv(csv_dir_overall / "s1_s5_descriptives.csv", index=False, encoding="utf-8-sig")
        s_overall_wwr.to_csv(csv_dir_overall / "s1_s5_descriptives_by_wwr.csv", index=False, encoding="utf-8-sig")
        s_overall_cx.to_csv(csv_dir_overall / "s1_s5_descriptives_by_complexity.csv", index=False, encoding="utf-8-sig")
        b_overall.to_csv(csv_dir_overall / "b1_b3_descriptives.csv", index=False, encoding="utf-8-sig")
        b_overall_wwr.to_csv(csv_dir_overall / "b1_b3_descriptives_by_wwr.csv", index=False, encoding="utf-8-sig")
        ipq_overall.to_csv(csv_dir_overall / "ipq_descriptives.csv", index=False, encoding="utf-8-sig")

        outputs += [
            str((csv_dir_overall / "s1_s5_descriptives.csv").relative_to(out)),
            str((csv_dir_overall / "s1_s5_descriptives_by_wwr.csv").relative_to(out)),
            str((csv_dir_overall / "s1_s5_descriptives_by_complexity.csv").relative_to(out)),
            str((csv_dir_overall / "b1_b3_descriptives.csv").relative_to(out)),
            str((csv_dir_overall / "b1_b3_descriptives_by_wwr.csv").relative_to(out)),
            str((csv_dir_overall / "ipq_descriptives.csv").relative_to(out)),
        ]

        for p in _plot_distribution_panels(x, S_COLS, fig_dir_overall, prefix="overall_s", xcol="WWR" if "WWR" in x.columns else None):
            outputs.append(str(Path(p).relative_to(out)))
        for p in _plot_distribution_panels(b_src, B_COLS, fig_dir_overall, prefix="overall_b", xcol="WWR" if "WWR" in b_src.columns else None):
            outputs.append(str(Path(p).relative_to(out)))

        if "ExperienceGroup" in x.columns:
            exp_dir = base / "experience"
            exp_dir.mkdir(parents=True, exist_ok=True)
            csv_dir_exp = exp_dir / "csv"
            png_dir_exp = exp_dir / "png"
            csv_dir_exp.mkdir(parents=True, exist_ok=True)
            png_dir_exp.mkdir(parents=True, exist_ok=True)
            fig_dir_exp = png_dir_exp

            s_exp = _desc_table(x, S_COLS, ["ExperienceGroup"])
            s_exp_wwr = _desc_table(x, S_COLS, ["ExperienceGroup", "WWR"]) if "WWR" in x.columns else pd.DataFrame()
            s_exp_cx = _desc_table(x, S_COLS, ["ExperienceGroup", "Complexity"]) if "Complexity" in x.columns else pd.DataFrame()

            b_exp = _desc_table(b_src, B_COLS, ["ExperienceGroup"])
            b_exp_wwr = _desc_table(b_src, B_COLS, ["ExperienceGroup", "WWR"]) if "WWR" in b_src.columns else pd.DataFrame()

            ipq_exp = _desc_table(ipq_subj, IPQ_COLS, ["ExperienceGroup"])

            s_exp.to_csv(csv_dir_exp / "s1_s5_descriptives_by_experience.csv", index=False, encoding="utf-8-sig")
            s_exp_wwr.to_csv(csv_dir_exp / "s1_s5_descriptives_by_experience_wwr.csv", index=False, encoding="utf-8-sig")
            s_exp_cx.to_csv(csv_dir_exp / "s1_s5_descriptives_by_experience_complexity.csv", index=False, encoding="utf-8-sig")
            b_exp.to_csv(csv_dir_exp / "b1_b3_descriptives_by_experience.csv", index=False, encoding="utf-8-sig")
            b_exp_wwr.to_csv(csv_dir_exp / "b1_b3_descriptives_by_experience_wwr.csv", index=False, encoding="utf-8-sig")
            ipq_exp.to_csv(csv_dir_exp / "ipq_descriptives_by_experience.csv", index=False, encoding="utf-8-sig")

            outputs += [
                str((csv_dir_exp / "s1_s5_descriptives_by_experience.csv").relative_to(out)),
                str((csv_dir_exp / "s1_s5_descriptives_by_experience_wwr.csv").relative_to(out)),
                str((csv_dir_exp / "s1_s5_descriptives_by_experience_complexity.csv").relative_to(out)),
                str((csv_dir_exp / "b1_b3_descriptives_by_experience.csv").relative_to(out)),
                str((csv_dir_exp / "b1_b3_descriptives_by_experience_wwr.csv").relative_to(out)),
                str((csv_dir_exp / "ipq_descriptives_by_experience.csv").relative_to(out)),
            ]

            for p in _plot_distribution_panels(x, S_COLS, fig_dir_exp, prefix="experience_s", hue="ExperienceGroup", xcol="WWR" if "WWR" in x.columns else "ExperienceGroup"):
                outputs.append(str(Path(p).relative_to(out)))
            for p in _plot_distribution_panels(b_src, B_COLS, fig_dir_exp, prefix="experience_b", hue="ExperienceGroup", xcol="WWR" if "WWR" in b_src.columns else "ExperienceGroup"):
                outputs.append(str(Path(p).relative_to(out)))

    payload = {
        "task": "descriptive pipeline",
        "scope": ["overall", "experience"],
        "branches": [b for b, _ in branches],
        "outputs": outputs,
        "stats": ["n", "mean", "sd", "median", "min", "max", "skewness", "kurtosis", "ci95", "shapiro_p"],
        "stratification": ["WWR", "Complexity", "ExperienceGroup"],
        "figure_style": "Eyetrack-inspired publication style; vivid blue/orange palette; clean axes; in-plot statistical annotations; *_main.png uses box + light strip + mean/CI and *_box.png is box-only.",
    }
    (out / "descriptive_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
