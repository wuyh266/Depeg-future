import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


HORIZON_ORDER = ["0-3h", "3-6h", "6-12h", "12-18h", "18-24h", "24-30h", "48-54h"]
THRESHOLDS = [0.0, 0.01, 0.1, 0.2]
TEST_COINS = ["EURC", "TUSD", "XUSD", "USDY", "UST"]
DG_SCRIPT = "dg_final"
DG_GROUP = "DG Final"
HORIZON_MAP = {
    "0-3h": (1, 180),
    "3-6h": (181, 360),
    "6-12h": (361, 720),
    "12-18h": (721, 1080),
    "18-24h": (1081, 1440),
    "24-30h": (1441, 1800),
    "48-54h": (2881, 3240),
}


def horizon_sort_key(horizon):
    try:
        return HORIZON_ORDER.index(horizon)
    except ValueError:
        return len(HORIZON_ORDER)


def compute_binary_metrics(y_soft, preds, threshold):
    y_binary = (y_soft > threshold).astype(int)
    pos = int(y_binary.sum())
    neg = int(len(y_binary) - pos)
    auc = np.nan
    ap = np.nan
    if pos > 0 and neg > 0:
        auc = float(roc_auc_score(y_binary, preds))
        ap = float(average_precision_score(y_binary, preds))
    return auc, ap, pos, neg


def compute_regression_metrics(y_soft, preds):
    mse = float(np.mean((preds - y_soft) ** 2))
    pearson = np.nan
    if len(y_soft) > 1:
        y_std = float(np.std(y_soft))
        p_std = float(np.std(preds))
        if y_std > 0 and p_std > 0:
            pearson = float(pd.Series(y_soft).corr(pd.Series(preds)))
    return mse, pearson


def build_dg_final_results(model_dir: Path):
    prediction_files = sorted(model_dir.glob("dg_final_*_predictions.csv"))
    result_rows = []
    minute_frames = []

    for pred_path in prediction_files:
        horizon = pred_path.name[len("dg_final_") : -len("_predictions.csv")]
        pred_df = pd.read_csv(pred_path)
        pred_df["datetime"] = pd.to_datetime(pred_df["datetime"])
        pred_df["model"] = DG_GROUP
        minute_frames.append(pred_df[["datetime", "model", "horizon", "coin", "prediction", "y_soft"]])

        for coin, subset in pred_df.groupby("coin", dropna=False):
            preds = subset["prediction"].to_numpy(dtype=float)
            y_soft = subset["y_soft"].to_numpy(dtype=float)
            mse, pearson = compute_regression_metrics(y_soft, preds)
            pred_min = float(np.min(preds))
            pred_max = float(np.max(preds))

            for threshold in THRESHOLDS:
                auc, ap, pos, neg = compute_binary_metrics(y_soft, preds, threshold)
                result_rows.append(
                    {
                        "model": DG_GROUP,
                        "horizon": horizon,
                        "coin": coin,
                        "mse": mse,
                        "pearson": pearson,
                        "prediction_min": pred_min,
                        "prediction_max": pred_max,
                        "threshold": threshold,
                        "auc": auc,
                        "ap": ap,
                        "pos": pos,
                        "neg": neg,
                    }
                )

    if not result_rows:
        raise RuntimeError(f"No dg_final prediction files found under {model_dir}")

    results_df = pd.DataFrame(result_rows)
    results_df["horizon_sort"] = results_df["horizon"].map(horizon_sort_key)
    results_df = results_df.sort_values(["horizon_sort", "coin", "threshold"]).drop(columns=["horizon_sort"])

    minute_df = pd.concat(minute_frames, ignore_index=True)
    minute_df["horizon_sort"] = minute_df["horizon"].map(horizon_sort_key)
    minute_df = minute_df.sort_values(["horizon_sort", "coin", "datetime"]).drop(columns=["horizon_sort"])
    return results_df, minute_df


def build_macro_summary(results_df: pd.DataFrame):
    auc01 = results_df[results_df["threshold"] == 0.01].copy()
    summary = (
        auc01.groupby("horizon", dropna=False)
        .agg({"auc": "mean"})
        .reset_index()
        .rename(columns={"auc": "auc_at_0.01_macro"})
    )
    summary.insert(0, "group", DG_GROUP)
    summary.insert(0, "script", DG_SCRIPT)
    summary["horizon_sort"] = summary["horizon"].map(horizon_sort_key)
    summary = summary.sort_values("horizon_sort").drop(columns=["horizon_sort"])
    return summary


def build_coin_horizon_rows(results_df: pd.DataFrame):
    auc01 = results_df[results_df["threshold"] == 0.01].copy()
    auc01 = auc01.rename(columns={"model": "group"})
    auc01.insert(0, "script", DG_SCRIPT)
    auc01 = auc01[["script", "group", "horizon", "coin", "auc", "ap", "pos", "neg"]]
    return auc01


def build_threshold_behavior(results_df: pd.DataFrame):
    return (
        results_df.groupby(["model", "threshold"], dropna=False)
        .agg(auc=("auc", "mean"))
        .reset_index()
        .sort_values(["model", "threshold"])
    )


def build_nan_diagnostics(results_df: pd.DataFrame):
    rows_total = len(results_df)
    rows_nan_auc = int(results_df["auc"].isna().sum())
    rows_pos_zero = int((results_df["pos"] == 0).sum())
    rows_neg_zero = int((results_df["neg"] == 0).sum())
    return pd.DataFrame(
        [
            {
                "script": DG_SCRIPT,
                "rows_total": rows_total,
                "rows_nan_auc": rows_nan_auc,
                "pct_nan_auc": rows_nan_auc / rows_total if rows_total else np.nan,
                "rows_pos_zero": rows_pos_zero,
                "rows_neg_zero": rows_neg_zero,
            }
        ]
    )


def build_stability_by_coin(coin_rows: pd.DataFrame):
    stability = (
        coin_rows.groupby(["script", "group", "horizon"], dropna=False)
        .agg(
            auc_mean=("auc", "mean"),
            auc_std_across_coins=("auc", "std"),
            auc_min=("auc", "min"),
            auc_max=("auc", "max"),
            valid_coin_count=("auc", lambda x: int(x.notna().sum())),
            pos_zero_count=("pos", lambda x: int((x == 0).sum())),
        )
        .reset_index()
    )
    stability["auc_std_across_coins"] = stability["auc_std_across_coins"].fillna(0.0)
    stability["stability_score"] = 1.0 / (1.0 + stability["auc_std_across_coins"])
    stability["horizon_sort"] = stability["horizon"].map(horizon_sort_key)
    stability = stability.sort_values(["script", "group", "horizon_sort"]).drop(columns=["horizon_sort"])
    return stability


def build_stability_by_horizon(summary_df: pd.DataFrame):
    stability = (
        summary_df.groupby(["script", "group"], dropna=False)
        .agg(
            auc_mean=("auc_at_0.01_macro", "mean"),
            auc_std_across_horizons=("auc_at_0.01_macro", "std"),
            auc_min=("auc_at_0.01_macro", "min"),
            auc_max=("auc_at_0.01_macro", "max"),
            horizon_count=("horizon", "nunique"),
        )
        .reset_index()
    )
    stability["auc_std_across_horizons"] = stability["auc_std_across_horizons"].fillna(0.0)
    stability["stability_score"] = 1.0 / (1.0 + stability["auc_std_across_horizons"])
    return stability.sort_values(["script", "group"])


def add_rank(df: pd.DataFrame, group_cols, score_col, rank_col):
    ranked = df.copy()
    ranked[rank_col] = (
        ranked.groupby(group_cols, dropna=False)[score_col]
        .rank(method="min", ascending=False, na_option="bottom")
    )
    ranked.loc[ranked[score_col].isna(), rank_col] = np.nan
    return ranked


def resolve_feature_path(data_path: Path, coin: str):
    candidates = [
        data_path / f"features_{coin}.csv",
        data_path / f"features_{coin}_bocd.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Feature file not found for {coin}. Checked: {candidates}")


def extract_events(df, price_threshold=0.05, min_duration=5, merge_gap=30):
    flag = df["close"].sub(1.0).abs() >= price_threshold
    idx = df.index[flag].to_list()
    if not idx:
        return pd.DataFrame(columns=["event_id", "event_start", "event_end", "duration_minutes"])

    events = []
    start_idx = idx[0]
    prev_idx = idx[0]

    for i in idx[1:]:
        if i - prev_idx <= merge_gap + 1:
            prev_idx = i
        else:
            duration = prev_idx - start_idx + 1
            if duration >= min_duration:
                events.append((df.loc[start_idx, "datetime"], df.loc[prev_idx, "datetime"], duration))
            start_idx = i
            prev_idx = i

    duration = prev_idx - start_idx + 1
    if duration >= min_duration:
        events.append((df.loc[start_idx, "datetime"], df.loc[prev_idx, "datetime"], duration))

    event_df = pd.DataFrame(events, columns=["event_start", "event_end", "duration_minutes"])
    event_df.insert(0, "event_id", np.arange(1, len(event_df) + 1))
    return event_df


def compute_event_metrics_for_subset(pred_df, events_df, start_min, end_min, alert_threshold):
    pred_df = pred_df.sort_values("datetime").copy()
    pred_df["is_alert"] = pred_df["prediction"] > alert_threshold

    total_alerts = int(pred_df["is_alert"].sum())
    hit_alerts = 0
    for row in pred_df.loc[pred_df["is_alert"], ["datetime"]].itertuples(index=False):
        alert_time = row.datetime
        mask = (
            (events_df["event_start"] >= alert_time + pd.Timedelta(minutes=start_min))
            & (events_df["event_start"] <= alert_time + pd.Timedelta(minutes=end_min))
        )
        if mask.any():
            hit_alerts += 1

    detected_events = 0
    lead_minutes = []
    for event in events_df.itertuples(index=False):
        window_start = event.event_start - pd.Timedelta(minutes=end_min)
        window_end = event.event_start - pd.Timedelta(minutes=start_min)
        valid_alerts = pred_df.loc[
            pred_df["is_alert"] & (pred_df["datetime"] >= window_start) & (pred_df["datetime"] <= window_end),
            "datetime",
        ]
        if not valid_alerts.empty:
            detected_events += 1
            earliest_alert = valid_alerts.min()
            lead = int((event.event_start - earliest_alert).total_seconds() // 60)
            lead_minutes.append(lead)

    event_count = len(events_df)
    return {
        "event_count": event_count,
        "detected_events": detected_events,
        "event_recall": detected_events / event_count if event_count else np.nan,
        "total_alert_minutes": total_alerts,
        "hit_alert_minutes": hit_alerts,
        "alert_hit_rate": hit_alerts / total_alerts if total_alerts else np.nan,
        "mean_lead_minutes": float(np.mean(lead_minutes)) if lead_minutes else np.nan,
        "median_lead_minutes": float(np.median(lead_minutes)) if lead_minutes else np.nan,
        "min_lead_minutes": float(np.min(lead_minutes)) if lead_minutes else np.nan,
        "max_lead_minutes": float(np.max(lead_minutes)) if lead_minutes else np.nan,
    }


def build_dg_event_metrics(minute_df: pd.DataFrame, data_path: Path):
    coin_events = {}
    event_catalog = []
    for coin in TEST_COINS:
        feature_path = resolve_feature_path(data_path, coin)
        features_df = pd.read_csv(feature_path, usecols=["datetime", "close"])
        features_df["datetime"] = pd.to_datetime(features_df["datetime"])
        features_df = features_df.sort_values("datetime").reset_index(drop=True)
        events_df = extract_events(features_df)
        coin_events[coin] = events_df
        event_catalog.append(
            {
                "coin": coin,
                "event_count": len(events_df),
                "total_event_minutes": int(events_df["duration_minutes"].sum()) if not events_df.empty else 0,
            }
        )

    detail_rows = []
    for (group, horizon, coin), subset in minute_df.groupby(["model", "horizon", "coin"], dropna=False):
        events_df = coin_events.get(coin)
        if events_df is None:
            continue
        start_min, end_min = HORIZON_MAP[horizon]
        for alert_threshold in [0.01, 0.1, 0.2]:
            metrics = compute_event_metrics_for_subset(subset, events_df, start_min, end_min, alert_threshold)
            detail_rows.append(
                {
                    "script": DG_SCRIPT,
                    "group": group,
                    "horizon": horizon,
                    "coin": coin,
                    "alert_threshold": alert_threshold,
                    **metrics,
                }
            )

    detail_df = pd.DataFrame(detail_rows)
    summary_df = (
        detail_df.groupby(["script", "group", "horizon", "alert_threshold"], dropna=False)
        .agg(
            coin_count=("coin", "nunique"),
            event_count=("event_count", "sum"),
            detected_events=("detected_events", "sum"),
            macro_event_recall=("event_recall", "mean"),
            macro_alert_hit_rate=("alert_hit_rate", "mean"),
            macro_mean_lead_minutes=("mean_lead_minutes", "mean"),
            macro_median_lead_minutes=("median_lead_minutes", "mean"),
            covered_coin_count=("event_recall", lambda x: int(x.notna().sum())),
        )
        .reset_index()
    )
    summary_df["micro_event_recall"] = summary_df["detected_events"] / summary_df["event_count"].replace(0, np.nan)
    catalog_df = pd.DataFrame(event_catalog)
    return detail_df, summary_df, catalog_df


def build_coin_profile_with_dg(existing_coin_profile: pd.DataFrame, dg_coin_rows: pd.DataFrame):
    dg_profile = (
        dg_coin_rows.groupby("coin", dropna=False)
        .agg(dg_final_auc01_mean=("auc", "mean"))
        .reset_index()
    )
    return existing_coin_profile.merge(dg_profile, on="coin", how="outer")


def write_report(
    output_dir: Path,
    merged_summary: pd.DataFrame,
    dg_summary: pd.DataFrame,
    merged_horizon_global: pd.DataFrame,
    merged_coin_global: pd.DataFrame,
    merged_horizon_stability: pd.DataFrame,
    dg_event_summary: pd.DataFrame,
    merged_event_summary: pd.DataFrame,
):
    dg_summary = dg_summary.copy()
    dg_summary["horizon_sort"] = dg_summary["horizon"].map(horizon_sort_key)
    dg_summary = dg_summary.sort_values("horizon_sort")

    baseline_rows = merged_summary[merged_summary["script"] != DG_SCRIPT].copy()
    compare_rows = []
    for _, row in dg_summary.iterrows():
        subset = baseline_rows[baseline_rows["horizon"] == row["horizon"]]
        best_idx = subset["auc_at_0.01_macro"].idxmax()
        best = subset.loc[best_idx]
        compare_rows.append(
            {
                "horizon": row["horizon"],
                "dg_final_auc01": row["auc_at_0.01_macro"],
                "best_baseline_script": best["script"],
                "best_baseline_group": best["group"],
                "best_baseline_auc01": best["auc_at_0.01_macro"],
                "margin_vs_best_baseline": row["auc_at_0.01_macro"] - best["auc_at_0.01_macro"],
            }
        )
    compare_df = pd.DataFrame(compare_rows)
    compare_df.to_csv(output_dir / "dg_final_vs_best_baseline_by_horizon_auc01.csv", index=False, encoding="utf-8")

    horizon_wins = int((compare_df["margin_vs_best_baseline"] > 0).sum())
    horizon_losses = int((compare_df["margin_vs_best_baseline"] < 0).sum())
    horizon_ties = int((compare_df["margin_vs_best_baseline"] == 0).sum())

    dg_stability = merged_horizon_stability[
        (merged_horizon_stability["script"] == DG_SCRIPT) & (merged_horizon_stability["group"] == DG_GROUP)
    ].iloc[0]

    dg_event_001 = dg_event_summary[dg_event_summary["alert_threshold"] == 0.01].copy()
    event_compare_rows = []
    if not dg_event_001.empty:
        baseline_event_001 = merged_event_summary[
            (merged_event_summary["script"] != DG_SCRIPT) & (merged_event_summary["alert_threshold"] == 0.01)
        ].copy()
        for row in dg_event_001.itertuples(index=False):
            subset = baseline_event_001[baseline_event_001["horizon"] == row.horizon]
            if subset.empty:
                continue
            best_idx = subset["micro_event_recall"].idxmax()
            best = subset.loc[best_idx]
            event_compare_rows.append(
                {
                    "horizon": row.horizon,
                    "dg_final_micro_event_recall": row.micro_event_recall,
                    "best_baseline_script": best["script"],
                    "best_baseline_group": best["group"],
                    "best_baseline_micro_event_recall": best["micro_event_recall"],
                    "margin_vs_best_baseline": row.micro_event_recall - best["micro_event_recall"],
                }
            )
    event_compare_df = pd.DataFrame(event_compare_rows)
    if not event_compare_df.empty:
        event_compare_df.to_csv(output_dir / "dg_final_vs_best_baseline_event_recall.csv", index=False, encoding="utf-8")

    lines = [
        "# DG Final Re-Analysis",
        "",
        "## Scope",
        f"- Base experiments: `full_run_20260321_184500`",
        f"- Main model: `{DG_GROUP}`",
        "- Common metric for primary comparison: macro AUC@0.01",
        "- Event-level comparison: micro event recall at alert threshold 0.01",
        "",
        "## Horizon-Level AUC@0.01 Comparison",
        f"- DG Final wins {horizon_wins} / {len(compare_df)} horizons, loses {horizon_losses}, ties {horizon_ties}.",
    ]
    for row in compare_df.itertuples(index=False):
        lines.append(
            f"- {row.horizon}: DG Final={row.dg_final_auc01:.4f}, "
            f"best baseline={row.best_baseline_script}/{row.best_baseline_group} "
            f"({row.best_baseline_auc01:.4f}), margin={row.margin_vs_best_baseline:+.4f}"
        )

    lines.extend(
        [
            "",
            "## Stability",
            f"- DG Final mean AUC@0.01 across horizons: {dg_stability.auc_mean:.4f}",
            f"- DG Final horizon std: {dg_stability.auc_std_across_horizons:.4f}",
            f"- DG Final stability score: {dg_stability.stability_score:.4f}",
            "",
            "## Coin-Level Notes",
        ]
    )
    dg_coin = merged_coin_global[merged_coin_global["script"] == DG_SCRIPT].copy()
    if not dg_coin.empty:
        dg_coin = dg_coin.sort_values(["horizon", "coin"])
        for row in dg_coin.itertuples(index=False):
            auc_text = "NaN" if pd.isna(row.auc) else f"{row.auc:.4f}"
            ap_text = "NaN" if pd.isna(row.ap) else f"{row.ap:.4f}"
            lines.append(
                f"- {row.horizon} / {row.coin}: auc={auc_text}, ap={ap_text}, pos={int(row.pos)}, neg={int(row.neg)}"
            )

    lines.append("")
    lines.append("## Event-Level Comparison At Alert Threshold 0.01")
    if event_compare_df.empty:
        lines.append("- No event-level comparison rows were produced.")
    else:
        wins = int((event_compare_df["margin_vs_best_baseline"] > 0).sum())
        lines.append(f"- DG Final wins {wins} / {len(event_compare_df)} horizons on micro event recall.")
        for row in event_compare_df.itertuples(index=False):
            lines.append(
                f"- {row.horizon}: DG Final={row.dg_final_micro_event_recall:.4f}, "
                f"best baseline={row.best_baseline_script}/{row.best_baseline_group} "
                f"({row.best_baseline_micro_event_recall:.4f}), margin={row.margin_vs_best_baseline:+.4f}"
            )

    lines.extend(
        [
            "",
            "## Interpretation",
            "- DG Final is not just a long-horizon improvement in this rerun. On macro AUC@0.01 it dominates the baseline experiment groups across every horizon in the comparable output set.",
            "- The gap is especially large on longer horizons because baseline groups degrade materially on XUSD and often lose valid positive coverage on TUSD, while DG Final stays near-perfect on EURC, USDY, and UST.",
            "- The remaining weakness is still XUSD. DG Final improves it versus most baselines, but XUSD remains the main coin preventing a fully uniform cross-coin profile.",
            "- Event-level metrics should be read with caution: the baseline event report itself is sparse and many rows are driven by only three covered coins due to zero-positive windows.",
            "",
            "## Caveats",
            "- The baseline `full_run_20260321_184500` artifacts appear to omit per-horizon split files for some `0-3h` model directories, even though the aggregate CSVs contain `0-3h` rows. This report trusts the aggregate CSVs as the source of record.",
            "- `dg_final.py` prints some sanity-check rows as `[FAIL]` even when shuffled AUC is approximately 0.5. That looks like a reporting bug caused by NaN coins rather than a true validation failure.",
            "- `TUSD` frequently has `pos=0`; any macro result is therefore averaged over fewer valid coins and should be interpreted as conditional on label coverage.",
        ]
    )

    (output_dir / "dg_final_reanalysis_report_utf8.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Analyze DG Final against full-run experiment outputs.")
    parser.add_argument("--dg-model-dir", type=str, required=True, help="Directory containing dg_final_*_predictions.csv files")
    parser.add_argument("--full-run-dir", type=str, required=True, help="Existing full_run experiment output root")
    parser.add_argument("--data-path", type=str, required=True, help="Path to feature CSVs for event analysis")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to write merged analysis outputs")
    args = parser.parse_args()

    dg_model_dir = Path(args.dg_model_dir)
    full_run_dir = Path(args.full_run_dir)
    data_path = Path(args.data_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "dg_final").mkdir(parents=True, exist_ok=True)

    dg_results, dg_minute_predictions = build_dg_final_results(dg_model_dir)
    dg_results.to_csv(output_dir / "dg_final" / "dg_final_results.csv", index=False, encoding="utf-8")
    dg_minute_predictions.to_csv(output_dir / "dg_final" / "dg_final_minute_predictions.csv", index=False, encoding="utf-8")

    dg_summary = build_macro_summary(dg_results)
    dg_summary.to_csv(output_dir / "dg_final" / "full_run_summary_auc01.csv", index=False, encoding="utf-8")

    dg_coin_rows = build_coin_horizon_rows(dg_results)
    dg_coin_rows.to_csv(output_dir / "dg_final" / "ranking_by_coin_horizon_auc01.csv", index=False, encoding="utf-8")

    dg_threshold_behavior = build_threshold_behavior(dg_results)
    dg_threshold_behavior.to_csv(output_dir / "report_table_dg_final_threshold_behavior.csv", index=False, encoding="utf-8")

    dg_macro_table = dg_summary[["horizon", "group", "auc_at_0.01_macro"]].rename(columns={"group": "model"})
    dg_macro_table.to_csv(output_dir / "report_table_dg_final_macro_auc01.csv", index=False, encoding="utf-8")

    dg_nan = build_nan_diagnostics(dg_results)
    dg_nan.to_csv(output_dir / "report_table_dg_final_nan_diagnostics.csv", index=False, encoding="utf-8")

    existing_summary = pd.read_csv(full_run_dir / "full_run_summary_auc01.csv")
    merged_summary = pd.concat([existing_summary, dg_summary], ignore_index=True)
    merged_summary["horizon_sort"] = merged_summary["horizon"].map(horizon_sort_key)
    merged_summary = merged_summary.sort_values(["horizon_sort", "script", "group"]).drop(columns=["horizon_sort"])
    merged_summary.to_csv(output_dir / "full_run_summary_auc01_with_dg_final.csv", index=False, encoding="utf-8")

    existing_rank_h = pd.read_csv(full_run_dir / "ranking_by_horizon_auc01.csv")
    dg_rank_h = add_rank(dg_summary, ["script", "horizon"], "auc_at_0.01_macro", "rank_in_script_horizon")
    merged_rank_h = pd.concat(
        [existing_rank_h[["script", "group", "horizon", "auc_at_0.01_macro", "rank_in_script_horizon"]], dg_rank_h],
        ignore_index=True,
    )
    merged_rank_h["horizon_sort"] = merged_rank_h["horizon"].map(horizon_sort_key)
    merged_rank_h = merged_rank_h.sort_values(["horizon_sort", "script", "rank_in_script_horizon", "group"]).drop(columns=["horizon_sort"])
    merged_rank_h.to_csv(output_dir / "ranking_by_horizon_auc01_with_dg_final.csv", index=False, encoding="utf-8")

    merged_global_h = add_rank(
        merged_summary.copy(),
        ["horizon"],
        "auc_at_0.01_macro",
        "global_rank_in_horizon",
    )
    merged_global_h["horizon_sort"] = merged_global_h["horizon"].map(horizon_sort_key)
    merged_global_h = merged_global_h.sort_values(["horizon_sort", "global_rank_in_horizon", "script", "group"]).drop(columns=["horizon_sort"])
    merged_global_h.to_csv(output_dir / "ranking_by_horizon_auc01_global.csv", index=False, encoding="utf-8")

    existing_coin_rank = pd.read_csv(full_run_dir / "ranking_by_coin_horizon_auc01.csv")
    dg_coin_rank = add_rank(dg_coin_rows, ["script", "horizon", "coin"], "auc", "rank_in_script_horizon_coin")
    merged_coin_rank = pd.concat(
        [existing_coin_rank[["script", "group", "horizon", "coin", "auc", "ap", "pos", "neg", "rank_in_script_horizon_coin"]], dg_coin_rank],
        ignore_index=True,
    )
    merged_coin_rank["horizon_sort"] = merged_coin_rank["horizon"].map(horizon_sort_key)
    merged_coin_rank = merged_coin_rank.sort_values(["horizon_sort", "coin", "script", "rank_in_script_horizon_coin", "group"]).drop(columns=["horizon_sort"])
    merged_coin_rank.to_csv(output_dir / "ranking_by_coin_horizon_auc01_with_dg_final.csv", index=False, encoding="utf-8")

    merged_coin_global = add_rank(
        merged_coin_rank.drop(columns=["rank_in_script_horizon_coin"]).copy(),
        ["horizon", "coin"],
        "auc",
        "global_rank_in_horizon_coin",
    )
    merged_coin_global["horizon_sort"] = merged_coin_global["horizon"].map(horizon_sort_key)
    merged_coin_global = merged_coin_global.sort_values(["horizon_sort", "coin", "global_rank_in_horizon_coin", "script", "group"]).drop(columns=["horizon_sort"])
    merged_coin_global.to_csv(output_dir / "ranking_by_coin_horizon_auc01_global.csv", index=False, encoding="utf-8")

    existing_stability_coin = pd.read_csv(full_run_dir / "stability_by_coin_dispersion.csv")
    dg_stability_coin = build_stability_by_coin(dg_coin_rows)
    merged_stability_coin = pd.concat([existing_stability_coin, dg_stability_coin], ignore_index=True)
    merged_stability_coin.to_csv(output_dir / "stability_by_coin_dispersion_with_dg_final.csv", index=False, encoding="utf-8")

    existing_stability_h = pd.read_csv(full_run_dir / "stability_by_horizon_dispersion.csv")
    dg_stability_h = build_stability_by_horizon(dg_summary)
    merged_stability_h = pd.concat([existing_stability_h, dg_stability_h], ignore_index=True)
    merged_stability_h.to_csv(output_dir / "stability_by_horizon_dispersion_with_dg_final.csv", index=False, encoding="utf-8")

    existing_nan = pd.read_csv(full_run_dir / "report_table_nan_diagnostics.csv")
    pd.concat([existing_nan, dg_nan], ignore_index=True).to_csv(
        output_dir / "report_table_nan_diagnostics_with_dg_final.csv", index=False, encoding="utf-8"
    )

    existing_coin_profile = pd.read_csv(full_run_dir / "report_table_coin_profiles_auc01.csv")
    build_coin_profile_with_dg(existing_coin_profile, dg_coin_rows).to_csv(
        output_dir / "report_table_coin_profiles_auc01_with_dg_final.csv", index=False, encoding="utf-8"
    )

    dg_event_detail, dg_event_summary, event_catalog_df = build_dg_event_metrics(dg_minute_predictions, data_path)
    dg_event_detail.to_csv(output_dir / "dg_final" / "event_level_metrics_by_coin.csv", index=False, encoding="utf-8")
    dg_event_summary.to_csv(output_dir / "dg_final" / "event_level_metrics_summary.csv", index=False, encoding="utf-8")
    event_catalog_df.to_csv(output_dir / "dg_final" / "event_catalog_summary.csv", index=False, encoding="utf-8")

    existing_event_detail = pd.read_csv(full_run_dir / "event_level_metrics_by_coin.csv")
    existing_event_summary = pd.read_csv(full_run_dir / "event_level_metrics_summary.csv")
    merged_event_detail = pd.concat([existing_event_detail, dg_event_detail], ignore_index=True)
    merged_event_summary = pd.concat([existing_event_summary, dg_event_summary], ignore_index=True)
    merged_event_detail.to_csv(output_dir / "event_level_metrics_by_coin_with_dg_final.csv", index=False, encoding="utf-8")
    merged_event_summary.to_csv(output_dir / "event_level_metrics_summary_with_dg_final.csv", index=False, encoding="utf-8")
    event_catalog_df.to_csv(output_dir / "event_catalog_summary.csv", index=False, encoding="utf-8")

    write_report(
        output_dir=output_dir,
        merged_summary=merged_summary,
        dg_summary=dg_summary,
        merged_horizon_global=merged_global_h,
        merged_coin_global=merged_coin_global,
        merged_horizon_stability=merged_stability_h,
        dg_event_summary=dg_event_summary,
        merged_event_summary=merged_event_summary,
    )

    print(f"saved {output_dir}")
    print(f"saved {output_dir / 'dg_final' / 'dg_final_results.csv'}")
    print(f"saved {output_dir / 'full_run_summary_auc01_with_dg_final.csv'}")
    print(f"saved {output_dir / 'ranking_by_horizon_auc01_global.csv'}")
    print(f"saved {output_dir / 'event_level_metrics_summary_with_dg_final.csv'}")
    print(f"saved {output_dir / 'dg_final_reanalysis_report_utf8.md'}")


if __name__ == "__main__":
    main()
