#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import f as f_dist
from statsmodels.stats.multitest import multipletests

from plot_style import apply_bae_style

S_DVS = ["S1", "S2", "S3", "S4", "S5"]
LINEAR_COEF = np.array([-1.0, 0.0, 1.0])
QUADRATIC_COEF = np.array([1.0, -2.0, 1.0])


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


def _exclude_subjects(df: pd.DataFrame, text: str) -> pd.DataFrame:
    if not text:
        return df
    names = [x.strip() for x in str(text).split(",") if x.strip()]
    if not names or "SubjectID" not in df.columns:
        return df
    sid = df["SubjectID"].astype(str).str.strip()
    return df.loc[~sid.isin(set(names))].copy()


def _parse_levels(text: str) -> list[float]:
    vals = [float(x.strip()) for x in str(text).split(",") if x.strip()]
    if len(vals) != 3:
        raise SystemExit("--wwr-levels must contain exactly 3 comma-separated values, e.g. 15,45,75")
    return vals


def _level_label(level: float) -> str:
    return str(int(level)) if float(level).is_integer() else str(level)


def _subject_level_wide(df: pd.DataFrame, dv: str, levels: list[float], split_cols: list[str]) -> pd.DataFrame:
    sub = df.dropna(subset=["SubjectID", "WWR", dv]).copy()
    if sub.empty:
        return pd.DataFrame()

    sub["WWR"] = pd.to_numeric(sub["WWR"], errors="coerce")
    sub = sub[sub["WWR"].isin(levels)].copy()
    if sub.empty:
        return pd.DataFrame()

    grp_cols = ["SubjectID"] + split_cols + ["WWR"]
    agg = sub.groupby(grp_cols, as_index=False)[dv].mean()
    wide = agg.pivot_table(index=["SubjectID"] + split_cols, columns="WWR", values=dv, aggfunc="mean")

    for level in levels:
        if level not in wide.columns:
            wide[level] = np.nan

    wide = wide[levels].reset_index()
    rename_map = {level: f"WWR_{_level_label(level)}" for level in levels}
    return wide.rename(columns=rename_map)


def _contrast_glm_like(mat: np.ndarray, coef: np.ndarray) -> dict[str, float]:
    if mat.ndim != 2 or mat.shape[1] != len(coef):
        return {
            "n_subjects": 0,
            "mean_contrast": np.nan,
            "sd_contrast": np.nan,
            "se_contrast": np.nan,
            "ss": np.nan,
            "df1": 1.0,
            "df2": np.nan,
            "ms": np.nan,
            "mse": np.nan,
            "f": np.nan,
            "p": np.nan,
            "t": np.nan,
        }

    score = np.dot(mat, coef)
    score = score[np.isfinite(score)]
    n = int(len(score))
    if n < 2:
        mean_score = float(np.nanmean(score)) if n else np.nan
        return {
            "n_subjects": n,
            "mean_contrast": mean_score,
            "sd_contrast": np.nan,
            "se_contrast": np.nan,
            "ss": np.nan,
            "df1": 1.0,
            "df2": float(max(n - 1, 0)),
            "ms": np.nan,
            "mse": np.nan,
            "f": np.nan,
            "p": np.nan,
            "t": np.nan,
        }

    mean_score = float(np.mean(score))
    sd_score = float(np.std(score, ddof=1))
    se_score = float(sd_score / np.sqrt(n)) if n > 0 else np.nan
    df2 = float(n - 1)
    ss_effect = float(n * (mean_score ** 2))
    ss_error = float(np.sum((score - mean_score) ** 2))
    ms_effect = ss_effect
    mse = float(ss_error / df2) if df2 > 0 else np.nan
    if mse is not None and np.isfinite(mse) and mse > 0:
        f_value = float(ms_effect / mse)
        p_value = float(f_dist.sf(f_value, 1, df2))
    else:
        f_value = np.nan
        p_value = np.nan
    t_value = float(mean_score / se_score) if np.isfinite(se_score) and se_score > 0 else np.nan
    return {
        "n_subjects": n,
        "mean_contrast": mean_score,
        "sd_contrast": sd_score,
        "se_contrast": se_score,
        "ss": ss_effect,
        "df1": 1.0,
        "df2": df2,
        "ms": ms_effect,
        "mse": mse,
        "f": f_value,
        "p": p_value,
        "t": t_value,
    }


def _contrast_vector_column(levels: list[float], coef: np.ndarray) -> list[str]:
    return [f"WWR{_level_label(level)}:{float(c):g}" for level, c in zip(levels, coef)]


def _build_markdown(df: pd.DataFrame, out: Path, split_cols: list[str]) -> None:
    lines: list[str] = []
    lines.append("# WWR Polynomial Significance — Within-Subjects Contrasts")
    lines.append("")
    lines.append("Approximates SPSS `Analyze → General Linear Model → Repeated Measures` with `WWR levels = 3` and `Contrasts = Polynomial`.")
    lines.append("")
    if split_cols:
        lines.append(f"Split columns: {', '.join(split_cols)}")
        lines.append("")

    if df.empty:
        lines.append("No valid results.")
        out.write_text("\n".join(lines), encoding="utf-8")
        return

    display_cols = split_cols + [
        "DV", "Source", "Contrast", "Direction", "n_subjects", "Type III SS", "df", "Mean Square", "F", "Sig.", "SigAdj.", "SigStar"
    ]
    x = df[display_cols].copy()
    for c in ["Type III SS", "df", "Mean Square", "F", "Sig.", "SigAdj."]:
        x[c] = x[c].map(lambda v: "" if pd.isna(v) else f"{float(v):.4f}")
    lines.append(x.to_markdown(index=False))
    out.write_text("\n".join(lines), encoding="utf-8")


def _trend_direction_label(contrast: str, mean_contrast: float | None) -> str:
    if mean_contrast is None or pd.isna(mean_contrast):
        return ""
    val = float(mean_contrast)
    if contrast == "Linear":
        if val > 0:
            return "Linear ↑ (15<45<75)"
        if val < 0:
            return "Linear ↓ (15>45>75)"
        return "Linear ="
    if contrast == "Quadratic":
        if val > 0:
            return "U-shape (45 lowest)"
        if val < 0:
            return "Inverted-U (45 highest)"
        return "Quadratic ="
    return str(contrast)


def _fmt(v: float | None, nd: int = 3) -> str:
    if v is None or pd.isna(v):
        return "NA"
    return f"{float(v):.{nd}f}"


def _clean_label(value) -> str:
    s = str(value)
    if s.lower() == "nan":
        return "NA"
    mapping = {"0": "C0", "1": "C1", "low": "Low", "high": "High"}
    return mapping.get(s, s)


def _split_cell_label(row: pd.Series, cols: list[str]) -> str:
    if not cols:
        return "Overall"
    parts = []
    for c in cols:
        cname = "Round" if c == "Repetition" else c
        parts.append(f"{cname}={_clean_label(row[c])}")
    return " | ".join(parts)


def _style_axis(ax, grid_axis: str = "y"):
    ax.spines["left"].set_color("#243447")
    ax.spines["bottom"].set_color("#243447")
    ax.spines["left"].set_linewidth(0.9)
    ax.spines["bottom"].set_linewidth(0.9)
    ax.tick_params(axis="both", colors="#243447", length=3)
    ax.grid(axis=grid_axis, color="#E5ECF2", alpha=0.75, linewidth=0.7)
    other = "x" if grid_axis == "y" else "y"
    ax.grid(axis=other, visible=False)


def _contrast_lines(rsub: pd.DataFrame, p_col: str) -> list[str]:
    lines: list[str] = []
    for contrast in ["Linear", "Quadratic"]:
        rr = rsub[rsub["Contrast"] == contrast]
        if rr.empty:
            continue
        rec = rr.iloc[0]
        lines.append(
            f"{contrast}: F={_fmt(rec.get('F'), 2)}, p={_fmt(rec.get(p_col), 3)}{_sigstar(rec.get(p_col))}, ηp²={_fmt(rec.get('partial_eta2'), 3)}"
        )
        lines.append(f"{rec.get('Direction', '')}; n={int(rec.get('n_subjects', 0) or 0)}")
    return lines


def _plot_trend_panels(means: pd.DataFrame, results_df: pd.DataFrame, out_dir: Path, split_cols: list[str], levels: list[float]) -> list[str]:
    made: list[str] = []
    if means.empty:
        return made
    out_dir.mkdir(parents=True, exist_ok=True)

    p_col = "SigAdj." if "SigAdj." in results_df.columns and results_df["SigAdj."].notna().any() else "Sig."

    for dv in sorted(means["DV"].dropna().unique()):
        x = means[means["DV"] == dv].copy()
        if x.empty:
            continue

        if split_cols:
            split_cols_plot = split_cols[:]
            split_frame = x[split_cols_plot].drop_duplicates().reset_index(drop=True)
        else:
            split_cols_plot = ["__overall__"]
            split_frame = pd.DataFrame({"__overall__": ["Overall"]})
            x["__overall__"] = "Overall"

        n_panels = len(split_frame)
        ncols = min(3, max(1, n_panels))
        nrows = int(np.ceil(n_panels / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(max(8.6, 4.3 * ncols), max(4.2, 3.7 * nrows)), squeeze=False)
        axes_flat = axes.flatten()

        line_group_cols = [c for c in split_cols_plot if c != "Repetition"]
        hue_col = line_group_cols[0] if len(line_group_cols) == 1 and split_cols_plot != ["__overall__"] else None

        for i, (_, split_row) in enumerate(split_frame.iterrows()):
            ax = axes_flat[i]
            sub = x.copy()
            rsub = results_df[(results_df["DV"] == dv) & (results_df["Source"] == "WWR")].copy()
            for c in split_cols_plot:
                sub = sub[sub[c].astype(str) == str(split_row[c])]
                if c in rsub.columns:
                    rsub = rsub[rsub[c].astype(str) == str(split_row[c])]

            if sub.empty:
                ax.set_visible(False)
                continue

            sub = sub.sort_values(levels[0] if False else "WWR")
            if hue_col and hue_col in sub.columns:
                palette = ["#2F6DA3", "#E6862A", "#2A9D8F", "#8E7DBE"]
                for j, (hv, hg) in enumerate(sub.groupby(hue_col, dropna=False)):
                    hg = hg.sort_values("WWR")
                    color = palette[j % len(palette)]
                    ax.plot(hg["WWR"], hg["mean"], marker="o", label=_clean_label(hv), color=color, linewidth=2.0)
                    if hg["se"].notna().any():
                        ax.fill_between(hg["WWR"], hg["mean"] - hg["se"], hg["mean"] + hg["se"], alpha=0.14, color=color)
                    for _, rr in hg.iterrows():
                        ax.text(rr["WWR"], rr["mean"], f"{_fmt(rr['mean'],2)}", fontsize=8.0, color=color, ha="center", va="bottom")
            else:
                sub = sub.sort_values("WWR")
                ax.plot(sub["WWR"], sub["mean"], marker="o", color="#2F6DA3", linewidth=2.1)
                if sub["se"].notna().any():
                    ax.fill_between(sub["WWR"], sub["mean"] - sub["se"], sub["mean"] + sub["se"], alpha=0.16, color="#2F6DA3")
                for _, rr in sub.iterrows():
                    ax.text(rr["WWR"], rr["mean"], f"{_fmt(rr['mean'],2)}", fontsize=8.0, color="#2F6DA3", ha="center", va="bottom")

            title = _split_cell_label(split_row, [c for c in split_cols_plot if c != "__overall__"])
            ax.set_title(title if title else dv, fontsize=10.2)
            ax.set_xlabel("WWR")
            ax.set_xticks(levels)
            ax.set_xticklabels([_level_label(v) for v in levels])
            if i % ncols == 0:
                ax.set_ylabel(dv)
            else:
                ax.set_ylabel("")
            _style_axis(ax, grid_axis="y")

            info_lines = _contrast_lines(rsub, p_col=p_col)
            if info_lines:
                ax.text(
                    0.02, 0.98, "\n".join(info_lines),
                    transform=ax.transAxes, ha="left", va="top", fontsize=7.6, color="#243447",
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#D6E2EC", lw=0.7, alpha=0.94)
                )

        for j in range(n_panels, len(axes_flat)):
            axes_flat[j].set_visible(False)

        if hue_col:
            handles, labels = axes_flat[0].get_legend_handles_labels()
            if handles:
                fig.legend(handles, labels, title=("Experience" if hue_col == "ExperienceGroup" else hue_col), loc="upper center", ncol=max(1, len(labels)), frameon=False)
                fig.subplots_adjust(top=0.84)

        fig.suptitle(f"WWR trend profile — {dv}", y=0.98, fontsize=11.2, fontweight="bold")
        path = out_dir / f"task5_trend_profile_{dv}.png"
        fig.savefig(path, dpi=230)
        plt.close(fig)
        made.append(str(path))
    return made


def _plot_contrast_heatmaps(df: pd.DataFrame, out_dir: Path, split_cols: list[str], p_col: str, p_label: str) -> list[str]:
    made: list[str] = []
    if df.empty:
        return made
    out_dir.mkdir(parents=True, exist_ok=True)

    x = df[df["Source"] == "WWR"].copy()
    if x.empty:
        return made

    row_label_cols = split_cols[:] if split_cols else []
    if not row_label_cols:
        x["Panel"] = "ALL"
        x["RowLabel"] = "ALL"
    elif len(row_label_cols) == 1:
        x["Panel"] = x[row_label_cols[0]].astype(str)
        x["RowLabel"] = x[row_label_cols[0]].astype(str)
    else:
        x["Panel"] = x[row_label_cols[0]].astype(str)
        x["RowLabel"] = x[row_label_cols[1:]].astype(str).agg(" | ".join, axis=1)

    for contrast in ["Linear", "Quadratic"]:
        sub = x[x["Contrast"] == contrast].copy()
        if sub.empty:
            continue
        sub["Col"] = sub["DV"].astype(str)
        if "Repetition" in split_cols:
            sub["Col"] = sub["DV"].astype(str) + " | R" + sub["Repetition"].astype(str)

        mat = sub.pivot_table(index="RowLabel", columns="Col", values=p_col, aggfunc="first")
        if mat.empty:
            continue

        annot = mat.copy().astype(object)
        for r in annot.index:
            for c in annot.columns:
                p = annot.loc[r, c]
                if pd.isna(p):
                    annot.loc[r, c] = ""
                else:
                    pv = float(p)
                    if pv < 0.001:
                        ptxt = f"{pv:.2e}"
                    else:
                        ptxt = f"{pv:0.3f}"
                    annot.loc[r, c] = f"p={ptxt}{_sigstar(pv)}"

        if len(mat.index) == 1:
            vals = -np.log10(mat.astype(float)).iloc[0].to_numpy(dtype=float)
            cols = list(mat.columns)
            fig, ax = plt.subplots(figsize=(max(7.2, 1.0 * len(cols) + 2.8), 2.4))
            tile = vals.reshape(1, -1)
            sns.heatmap(
                tile,
                cmap=sns.light_palette("#6FA8DC", as_cmap=True),
                annot=np.array([annot.iloc[0].tolist()]),
                fmt="",
                annot_kws={"fontsize": 8.6},
                linewidths=0.8,
                linecolor="#E6ECE8",
                cbar_kws={"label": f"-log10({p_label})"},
                yticklabels=[mat.index[0]],
                xticklabels=cols,
                ax=ax,
            )
            ax.set_title(f"WWR Polynomial {contrast} significance strip ({p_label})")
            ax.set_xlabel("DV")
            ax.set_ylabel("Split cell")
            ax.tick_params(axis="x", rotation=20)
        else:
            fig, ax = plt.subplots(figsize=(max(9.0, 0.9 * len(mat.columns) + 4.0), max(4.0, 0.55 * len(mat.index) + 1.6)))
            sns.heatmap(
                -np.log10(mat.astype(float)),
                cmap=sns.light_palette("#6FA8DC", as_cmap=True),
                annot=annot,
                fmt="",
                annot_kws={"fontsize": 8.0},
                linewidths=0.6,
                linecolor="#E6ECE8",
                cbar_kws={"label": f"-log10({p_label})"},
                ax=ax,
            )
            ax.set_title(f"WWR Polynomial {contrast} significance map ({p_label})")
            ax.set_xlabel("DV")
            ax.set_ylabel("Split cell")
            ax.tick_params(axis="x", rotation=25)

        path = out_dir / f"task5_{contrast.lower()}_contrast_heatmap.png"
        fig.savefig(path, dpi=230)
        plt.close(fig)
        made.append(str(path))
    return made


def main():
    ap = argparse.ArgumentParser(
        description="WWR polynomial significance across 3 WWR levels"
    )
    ap.add_argument("--long-csv", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("results/research"))
    ap.add_argument("--wwr-levels", default="15,45,75", help="Exactly 3 comma-separated WWR levels in ascending order")
    ap.add_argument("--dvs", default=",".join(S_DVS), help="Comma-separated dependent variables, default S1,S2,S3,S4,S5")
    ap.add_argument("--split-by", default="", help="Optional comma-separated split columns, e.g. Repetition or ExperienceGroup")
    ap.add_argument("--exclude-subjects", default="", help="Comma-separated SubjectID list for QC exclusion")
    ap.add_argument("--p-adjust", default="none", help="Multiple-testing adjustment: holm|bonferroni|fdr_bh|none; default none to stay closer to SPSS output")
    args = ap.parse_args()

    apply_bae_style()

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    csv_dir = out / "csv"
    png_dir = out / "png"
    md_dir = out / "md"
    json_dir = out / "json"
    for d in [csv_dir, png_dir, md_dir, json_dir]:
        d.mkdir(parents=True, exist_ok=True)
    fig_dir = png_dir

    levels = _parse_levels(args.wwr_levels)
    dvs = [x.strip() for x in str(args.dvs).split(",") if x.strip()]
    split_cols = [x.strip() for x in str(args.split_by).split(",") if x.strip()]

    df = pd.read_csv(args.long_csv)
    df = _exclude_subjects(df, args.exclude_subjects)

    required = ["SubjectID", "WWR"] + split_cols
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing required columns: {missing}")

    rows: list[dict] = []
    means_rows: list[dict] = []
    wide_exports: list[str] = []

    for dv in dvs:
        if dv not in df.columns:
            continue
        wide = _subject_level_wide(df, dv, levels, split_cols)
        wide_path = csv_dir / f"wwr_subject_means_{dv}.csv"
        wide.to_csv(wide_path, index=False, encoding="utf-8-sig")
        wide_exports.append(str(wide_path.relative_to(out)))

        level_cols = [f"WWR_{_level_label(level)}" for level in levels]
        grouped = wide.groupby(split_cols, dropna=False) if split_cols else [((), wide)]
        for key, g in grouped:
            split_values = {}
            if split_cols:
                if not isinstance(key, tuple):
                    key = (key,)
                split_values = dict(zip(split_cols, key))
            block = g.dropna(subset=level_cols, how="any").copy()
            mat = block[level_cols].to_numpy(dtype=float) if not block.empty else np.empty((0, 3))

            for i, level in enumerate(levels):
                vals = mat[:, i] if mat.size else np.array([], dtype=float)
                n = int(np.sum(np.isfinite(vals)))
                mean_val = float(np.nanmean(vals)) if n else np.nan
                sd_val = float(np.nanstd(vals, ddof=1)) if n >= 2 else np.nan
                se_val = float(sd_val / np.sqrt(n)) if n >= 2 and np.isfinite(sd_val) else np.nan
                mr = dict(split_values)
                mr.update({
                    "DV": dv,
                    "WWR": level,
                    "n_subjects": n,
                    "mean": mean_val,
                    "sd": sd_val,
                    "se": se_val,
                })
                means_rows.append(mr)

            for contrast_name, coef in [("Linear", LINEAR_COEF), ("Quadratic", QUADRATIC_COEF)]:
                stat = _contrast_glm_like(mat, coef)
                item = dict(split_values)
                eta_p = np.nan
                if np.isfinite(stat["f"]) and np.isfinite(stat["df1"]) and np.isfinite(stat["df2"]) and (stat["f"] * stat["df1"] + stat["df2"]) > 0:
                    eta_p = float((stat["f"] * stat["df1"]) / (stat["f"] * stat["df1"] + stat["df2"]))
                item.update({
                    "DV": dv,
                    "Source": "WWR",
                    "Contrast": contrast_name,
                    "Direction": _trend_direction_label(contrast_name, stat["mean_contrast"]),
                    "CoefficientVector": " | ".join(_contrast_vector_column(levels, coef)),
                    "n_subjects": stat["n_subjects"],
                    "Type III SS": stat["ss"],
                    "df": stat["df1"],
                    "Mean Square": stat["ms"],
                    "F": stat["f"],
                    "Sig.": stat["p"],
                    "partial_eta2": eta_p,
                    "Error SS": float(stat["mse"] * stat["df2"]) if np.isfinite(stat["mse"]) and np.isfinite(stat["df2"]) else np.nan,
                    "Error df": stat["df2"],
                    "Error MS": stat["mse"],
                    "t": stat["t"],
                    "mean_contrast": stat["mean_contrast"],
                    "sd_contrast": stat["sd_contrast"],
                    "se_contrast": stat["se_contrast"],
                    "status": "ok" if stat["n_subjects"] >= 2 else "insufficient_complete_subjects",
                })
                rows.append(item)

                err = dict(split_values)
                err.update({
                    "DV": dv,
                    "Source": "Error(WWR)",
                    "Contrast": contrast_name,
                    "Direction": _trend_direction_label(contrast_name, stat["mean_contrast"]),
                    "CoefficientVector": " | ".join(_contrast_vector_column(levels, coef)),
                    "n_subjects": stat["n_subjects"],
                    "Type III SS": float(stat["mse"] * stat["df2"]) if np.isfinite(stat["mse"]) and np.isfinite(stat["df2"]) else np.nan,
                    "df": stat["df2"],
                    "Mean Square": stat["mse"],
                    "F": np.nan,
                    "Sig.": np.nan,
                    "Error SS": np.nan,
                    "Error df": np.nan,
                    "Error MS": np.nan,
                    "t": np.nan,
                    "mean_contrast": np.nan,
                    "sd_contrast": np.nan,
                    "se_contrast": np.nan,
                    "status": "ok" if stat["n_subjects"] >= 2 else "insufficient_complete_subjects",
                })
                rows.append(err)

    res = pd.DataFrame(rows)
    if not res.empty:
        res["SigAdj."] = res["Sig."]
        if args.p_adjust.lower() != "none":
            mask = (res["Source"] == "WWR") & res["Sig."].notna()
            res.loc[:, "SigAdj."] = np.nan
            res.loc[res["Source"] != "WWR", "SigAdj."] = np.nan
            if mask.any():
                _, padj, _, _ = multipletests(res.loc[mask, "Sig."].astype(float), method=args.p_adjust)
                res.loc[mask, "SigAdj."] = padj
        res["SigStar"] = res["SigAdj."].map(_sigstar)
        sort_cols = split_cols + ["DV", "Contrast", "Source"] if split_cols else ["DV", "Contrast", "Source"]
        res = res.sort_values(sort_cols).reset_index(drop=True)

    means_df = pd.DataFrame(means_rows)
    means_path = csv_dir / "wwr_profile_means.csv"
    csv_path = csv_dir / "wwr_polynomial_contrasts.csv"
    md_path = md_dir / "wwr_polynomial_contrasts.md"
    summary_path = json_dir / "wwr_polynomial_summary.json"

    means_df.to_csv(means_path, index=False, encoding="utf-8-sig")
    res.to_csv(csv_path, index=False, encoding="utf-8-sig")
    _build_markdown(res, md_path, split_cols)

    figs: list[str] = []
    for p in _plot_trend_panels(means_df, res, fig_dir, split_cols, levels):
        figs.append(str(Path(p).relative_to(out)))
    p_col = "SigAdj." if args.p_adjust.lower() != "none" else "Sig."
    p_label = "adjusted p" if args.p_adjust.lower() != "none" else "unadjusted p"
    for p in _plot_contrast_heatmaps(res, fig_dir, split_cols, p_col=p_col, p_label=p_label):
        figs.append(str(Path(p).relative_to(out)))

    payload = {
        "task": "wwr polynomial significance",
        "dvs": dvs,
        "wwr_levels": levels,
        "split_by": split_cols,
        "contrast_coefficients": {
            "Linear": LINEAR_COEF.tolist(),
            "Quadratic": QUADRATIC_COEF.tolist(),
        },
        "outputs": [
            str(csv_path.relative_to(out)),
            str(md_path.relative_to(out)),
            str(means_path.relative_to(out)),
            *wide_exports,
            *figs,
        ],
        "notes": [
            "Subject-level means are computed within each WWR level before contrast testing.",
            "For 3 WWR levels, Linear uses [-1, 0, 1] and Quadratic uses [1, -2, 1].",
            "Output table is organized to resemble SPSS Within-Subjects Contrasts (WWR / Error(WWR)).",
            "Default p-adjust is now 'none' to stay closer to SPSS repeated-measures output unless the user explicitly requests multiplicity correction.",
            "Each subject contributes one value per WWR level after within-subject averaging over any non-split repeated rows.",
            "PNG figures include WWR profile plots and contrast significance heatmaps for Linear/Quadratic contrasts.",
            "Interpret results from the observed data only; no expected-significance pattern is imposed or optimized for.",
            "WWR Polynomial figures now use main-panel + side-summary layout to reduce overlap and improve readability.",
        ],
    }
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
