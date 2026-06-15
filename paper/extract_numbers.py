"""Single-source every numeric claim in the paper from the result CSVs -> numbers.json.

The .tex draft reads these via \\newcommand macros, so no number is ever hand-typed into
the paper. Only four constants are taken from CHANGES.md / design (beta_learned,
declining_improvement_pct, n_locations, n_seasons); everything else is read from a CSV.
"""

from __future__ import annotations

import json
import os

import pandas as pd

HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, ".."))
RES, TAB = os.path.join(ROOT, "results"), os.path.join(ROOT, "tables")


def _round(x, n=2):
    return round(float(x), n)


# Display names for the multi-season leaderboard table.
_DISPLAY = {
    "stack_phase_conformal": "MIST-Hybrid (ours)", "stack_phase": "Hybrid (phase, no conf.)",
    "stack_global": "Hybrid (global)", "ens_perf": "Ensemble (perf-wt.)",
    "ens_median": "Ensemble (median)", "ens_mean": "Ensemble (mean)",
    "ens_trimmed": "Ensemble (trimmed)",
    "mist_v2": "MIST", "patchtst": "PatchTST", "tft": "TFT", "arima": "ARIMA",
    "ml": "GBQR", "persistence": "Persistence", "seasonal_naive": "Seasonal-naive",
}


def build() -> dict:
    lb = pd.read_csv(os.path.join(RES, "leaderboard_v2.csv")).set_index("model")
    ph = pd.read_csv(os.path.join(RES, "phase_performance_v2.csv")).set_index("model")
    hb = pd.read_csv(os.path.join(RES, "horizon_breakdown.csv")).set_index("model")
    plc = pd.read_csv(os.path.join(RES, "per_location_coverage.csv"))
    sw = pd.read_csv(os.path.join(RES, "season_wis.csv"))
    dm = pd.read_csv(os.path.join(TAB, "dm_significance.csv"), index_col=0)

    d: dict = {
        "n_locations": 53,
        "n_seasons": 3,
        "beta_learned": 2.71,
        "declining_improvement_pct": 29,
        "mist_overall_wis": _round(lb.loc["mist_v2", "wis"]),
        "arima_overall_wis": _round(lb.loc["arima", "wis"]),
        "mist_cov95": _round(lb.loc["mist_v2", "cov_95"], 3),
        "mist_rising_wis": _round(ph.loc["mist_v2", "Rising"]),
        "arima_rising_wis": _round(ph.loc["arima", "Rising"]),
        "mist_peak_wis": _round(ph.loc["mist_v2", "Peak"]),
        "mist_declining_wis": _round(ph.loc["mist_v2", "Declining"]),
        "per_location_cov50_mean": _round(plc["cov_50"].mean(), 3),
        "per_location_cov50_std": _round(plc["cov_50"].std(), 3),
        "per_location_cov95_mean": _round(plc["cov_95"].mean(), 3),
        "mist_h4_wis": _round(hb.loc["mist_v2", "4"]),
        "arima_h4_wis": _round(hb.loc["arima", "4"]),
        "dm_p_vs_no_blend": float(dm.loc["mist_v2", "mist_no_blend"]),
        "dm_p_vs_no_mech": float(dm.loc["mist_v2", "mist_no_mech"]),
    }

    # ARIMA rising p: prefer the pooled 3-season table, else single-season.
    for fname in ("dm_headline_rising_3seasons.csv", "dm_headline_rising.csv"):
        fp = os.path.join(TAB, fname)
        if os.path.exists(fp):
            hr = pd.read_csv(fp)
            row = hr[hr["vs"] == "arima"]
            if not row.empty:
                d["dm_p_vs_arima_rising"] = float(row["p"].iloc[0])
                d["dm_p_vs_arima_rising_source"] = fname
                break

    # Per-season mist/arima WIS.
    piv = sw.pivot_table(index="season", columns="model", values="wis")
    for tag, season in [("2022", "2022-23"), ("2023", "2023-24"), ("2024", "2024-25")]:
        if season in piv.index:
            if "mist_v2" in piv.columns:
                d[f"season_{tag}_mist"] = _round(piv.loc[season, "mist_v2"], 1)
            if "arima" in piv.columns:
                d[f"season_{tag}_arima"] = _round(piv.loc[season, "arima"], 1)
    # Season-unweighted means per base model (real full-run, from season_wis.csv).
    means = piv.mean(axis=0)
    for col, key in [("mist_v2", "season_mist_mean"), ("arima", "season_arima_mean"),
                     ("tft", "season_tft_mean"), ("patchtst", "season_patch_mean")]:
        if col in piv.columns:
            d[key] = _round(means[col], 1)

    # --- Multi-season hybrid/ensemble numbers: ONLY from a full run. ---
    profile_path = os.path.join(RES, "run_profile.txt")
    profile = open(profile_path).read().strip() if os.path.exists(profile_path) else "unknown"
    ms_path = os.path.join(RES, "multiseason_summary.csv")
    vd_path = os.path.join(RES, "hybrid_verdict.json")
    if profile == "full" and os.path.exists(ms_path):
        ms = pd.read_csv(ms_path).set_index("model")
        for m, key in [("stack_phase_conformal", "hybrid"), ("ens_perf", "ensperf"),
                       ("ens_median", "ensmedian"), ("ens_trimmed", "enstrimmed"),
                       ("ens_mean", "ensmean"), ("patchtst", "patchtst"),
                       ("tft", "tft"), ("mist_v2", "mistms")]:
            if m in ms.index:
                d[f"ms_{key}_wis"] = _round(ms.loc[m, "wis_unweighted"], 1)
        if "stack_phase_conformal" in ms.index:
            d["ms_hybrid_cov50"] = _round(ms.loc["stack_phase_conformal", "cov_50"], 3)
            d["ms_hybrid_cov95"] = _round(ms.loc["stack_phase_conformal", "cov_95"], 3)
    if profile == "full" and os.path.exists(vd_path):
        with open(vd_path) as f:
            v = json.load(f)
        d["hybrid_earned"] = bool(v.get("earned"))
        d["hybrid_best_base"] = _DISPLAY.get(v.get("best_base_model"), v.get("best_base_model"))
        d["hybrid_strongest_competitor"] = _DISPLAY.get(
            v.get("strongest_competitor"), v.get("strongest_competitor"))
    return d


# Map json keys -> LaTeX macro names (\newcommand). Only keys present are emitted.
_MACROS = {
    "n_locations": "nLoc", "n_seasons": "nSeasons", "beta_learned": "betaLearned",
    "declining_improvement_pct": "declImprove",
    "mist_overall_wis": "mistWIS", "arima_overall_wis": "arimaWIS",
    "mist_cov95": "mistCovNinetyfive",
    "mist_rising_wis": "mistRising", "arima_rising_wis": "arimaRising",
    "mist_peak_wis": "mistPeak", "mist_declining_wis": "mistDeclining",
    "per_location_cov50_mean": "covFiftyMean", "per_location_cov50_std": "covFiftyStd",
    "per_location_cov95_mean": "covNinetyfiveMean",
    "mist_h4_wis": "mistHfour", "arima_h4_wis": "arimaHfour",
    "dm_p_vs_no_blend": "dmBlend", "dm_p_vs_no_mech": "dmMech",
    "dm_p_vs_arima_rising": "dmArimaRising",
    "season_2022_mist": "seasonTwoTwoMist", "season_2022_arima": "seasonTwoTwoArima",
    "season_2023_mist": "seasonTwoThreeMist", "season_2023_arima": "seasonTwoThreeArima",
    "season_2024_mist": "seasonTwoFourMist", "season_2024_arima": "seasonTwoFourArima",
    "season_mist_mean": "seasonMistMean", "season_arima_mean": "seasonArimaMean",
    "season_tft_mean": "seasonTFTMean", "season_patch_mean": "seasonPatchMean",
    # Multi-season hybrid / ensemble (Table 1 headline numbers).
    "ms_hybrid_wis": "msHybridWIS", "ms_ensperf_wis": "msEnsPerfWIS",
    "ms_enstrimmed_wis": "msEnsTrimmedWIS", "ms_ensmean_wis": "msEnsMeanWIS",
    "ms_ensmedian_wis": "msEnsMedianWIS", "ms_patchtst_wis": "msPatchTSTWIS",
    "ms_tft_wis": "msTFTWIS", "ms_mistms_wis": "msMistWIS",
    "ms_hybrid_cov50": "msHybridCovFifty", "ms_hybrid_cov95": "msHybridCovNinetyfive",
    "hybrid_best_base": "hybridBestBase",
    "hybrid_strongest_competitor": "hybridStrongestRival",
}


def emit_season_base_table(out_path: str) -> bool:
    """Write paper/season_base_auto.tex from the full-run ``season_wis.csv``.

    This is the honest multi-season base-model leaderboard (real, committed
    full-run numbers — independent of the hybrid dump): it is the table that
    shows MIST does not generalise across seasons.
    """
    sw_path = os.path.join(RES, "season_wis.csv")
    if not os.path.exists(sw_path):
        return False
    sw = pd.read_csv(sw_path)
    piv = sw.pivot_table(index="model", columns="season", values="wis")
    seasons = list(piv.columns)
    piv["Unwt. mean"] = piv[seasons].mean(axis=1)
    order = ["mist_v2", "arima", "tft", "patchtst"]
    piv = piv.reindex([m for m in order if m in piv.index])

    lines = [
        "% Auto-generated by extract_numbers.py -- DO NOT EDIT BY HAND.",
        "\\begin{tabular}{l" + "r" * (len(seasons) + 1) + "}",
        "\\toprule",
        "Model & " + " & ".join(seasons) + " & Unwt. mean \\\\",
        "\\midrule",
    ]
    best_mean = piv["Unwt. mean"].min()
    for m, row in piv.iterrows():
        cells = " & ".join(f"{row[s]:.1f}" for s in seasons)
        mean_cell = f"{row['Unwt. mean']:.1f}"
        if abs(row["Unwt. mean"] - best_mean) < 1e-9:
            mean_cell = f"\\textbf{{{mean_cell}}}"
        lines.append(f"{_DISPLAY.get(m, m)} & {cells} & {mean_cell} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", ""]
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return True


def emit_leaderboard_table(out_path: str) -> bool:
    """Write paper/tables_auto.tex: the multi-season WIS leaderboard (single-sourced)."""
    ms_path = os.path.join(RES, "multiseason_summary.csv")
    sl_path = os.path.join(RES, "season_leaderboard.csv")
    if not (os.path.exists(ms_path) and os.path.exists(sl_path)):
        return False
    ms = pd.read_csv(ms_path)
    sl = pd.read_csv(sl_path)
    seasons = sorted(sl["season"].unique())
    wis_by = sl.pivot_table(index="model", columns="season", values="wis")

    lines = [
        "% Auto-generated by extract_numbers.py -- DO NOT EDIT BY HAND.",
        "\\begin{tabular}{l" + "r" * (len(seasons) + 2) + "}",
        "\\toprule",
        "Model & " + " & ".join(seasons) + " & Unwt. & Cov-50 \\\\",
        "\\midrule",
    ]
    best = ms["wis_unweighted"].min()
    for _, row in ms.sort_values("wis_unweighted").iterrows():
        m = row["model"]
        name = _DISPLAY.get(m, m)
        per_season = " & ".join(f"{wis_by.loc[m, s]:.1f}" if m in wis_by.index and s in wis_by.columns
                                else "--" for s in seasons)
        bold = abs(row["wis_unweighted"] - best) < 1e-9   # highlight the winner
        disp = f"\\textbf{{{name}}}" if bold else name
        lines.append(f"{disp} & {per_season} & {row['wis_unweighted']:.1f} "
                     f"& {row['cov_50']:.3f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", ""]
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return True


def _fmt_p(p: float) -> str:
    """Format a p-value for LaTeX prose."""
    if p < 1e-3:
        return f"{p:.0e}".replace("e-0", "\\times 10^{-").replace("e-", "\\times 10^{-") + "}"
    return f"{p:.3f}"


def _emit_macros(d: dict) -> str:
    lines = ["% Auto-generated by extract_numbers.py — DO NOT EDIT BY HAND.", ""]
    for key, macro in _MACROS.items():
        if key not in d:
            continue
        val = d[key]
        if key.startswith("dm_p_"):
            val = _fmt_p(float(val))
        lines.append(f"\\newcommand{{\\{macro}}}{{{val}}}")
    return "\n".join(lines) + "\n"


def run() -> dict:
    d = build()
    with open(os.path.join(HERE, "numbers.json"), "w") as f:
        json.dump(d, f, indent=2)
    macros = _emit_macros(d)
    # Boolean verdict macro to switch paper framing without hand-editing.
    if "hybrid_earned" in d:
        macros += (f"\\newcommand{{\\hybridEarned}}{{{'true' if d['hybrid_earned'] else 'false'}}}\n")
    with open(os.path.join(HERE, "macros.tex"), "w") as f:
        f.write(macros)
    # Real, full-run multi-season base table (from season_wis.csv) — always safe.
    emit_season_base_table(os.path.join(HERE, "season_base_auto.tex"))
    # Hybrid/ensemble table only from a full run (else the paper keeps its guard).
    profile_path = os.path.join(RES, "run_profile.txt")
    profile = open(profile_path).read().strip() if os.path.exists(profile_path) else "unknown"
    wrote_table = (emit_leaderboard_table(os.path.join(HERE, "tables_auto.tex"))
                   if profile == "full" else False)
    print(json.dumps(d, indent=2))
    print(f"\nsaved -> {os.path.join(HERE, 'numbers.json')} + macros.tex + season_base_auto.tex"
          + (" + tables_auto.tex" if wrote_table else "  (tables_auto.tex: full-run only)"))
    return d


if __name__ == "__main__":
    run()
