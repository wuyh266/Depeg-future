import argparse
from pathlib import Path

import numpy as np
import pandas as pd


HORIZON_MAP = {
    "0-3h": (1, 180),
    "3-6h": (181, 360),
    "6-12h": (361, 720),
    "12-18h": (721, 1080),
    "18-24h": (1081, 1440),
    "24-30h": (1441, 1800),
    "48-54h": (2881, 3240),
}

PREDICTION_SOURCES = {
    "killer_test": ("killer_test/killer_test_minute_predictions.csv", "test_name"),
    "fair_comparison": ("fair_comparison/fair_comparison_minute_predictions.csv", "model"),
    "lstm_comparison": ("lstm_comparison/lstm_comparison_minute_predictions.csv", "model"),
    "quick_ablation": ("quick_ablation/quick_ablation_minute_predictions.csv", "model"),
}

TEST_COINS = ["EURC", "TUSD", "XUSD", "USDY", "UST"]


def resolve_feature_path(data_path, coin):
    candidates = [
        Path(data_path) / f"features_{coin}.csv",
        Path(data_path) / f"features_{coin}_bocd.csv",
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


def main():
    parser = argparse.ArgumentParser(description="Compute event-level metrics from saved minute predictions.")
    parser.add_argument("--data-path", type=str, required=True, help="Path to feature CSVs")
    parser.add_argument("--output-dir", type=str, required=True, help="Experiment output root")
    parser.add_argument("--event-threshold", type=float, default=0.05, help="Absolute depeg threshold on close")
    parser.add_argument("--min-event-duration", type=int, default=5, help="Minimum event duration in minutes")
    parser.add_argument("--merge-gap", type=int, default=30, help="Merge nearby event gaps within N minutes")
    parser.add_argument("--alert-thresholds", type=str, default="0.01,0.1,0.2", help="Comma-separated alert thresholds")
    args = parser.parse_args()

    output_root = Path(args.output_dir)
    alert_thresholds = [float(x) for x in args.alert_thresholds.split(",") if x.strip()]

    coin_events = {}
    coin_event_summary = []
    for coin in TEST_COINS:
        feature_path = resolve_feature_path(args.data_path, coin)
        df = pd.read_csv(feature_path, usecols=["datetime", "close"])
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.sort_values("datetime").reset_index(drop=True)
        events_df = extract_events(
            df,
            price_threshold=args.event_threshold,
            min_duration=args.min_event_duration,
            merge_gap=args.merge_gap,
        )
        coin_events[coin] = events_df
        coin_event_summary.append(
            {
                "coin": coin,
                "event_count": len(events_df),
                "total_event_minutes": int(events_df["duration_minutes"].sum()) if not events_df.empty else 0,
            }
        )

    detail_rows = []
    for script, (rel_path, group_col) in PREDICTION_SOURCES.items():
        pred_path = output_root / rel_path
        if not pred_path.exists():
            continue
        pred_df = pd.read_csv(pred_path)
        pred_df["datetime"] = pd.to_datetime(pred_df["datetime"])

        for (group, horizon, coin), subset in pred_df.groupby([group_col, "horizon", "coin"], dropna=False):
            events_df = coin_events.get(coin)
            if events_df is None:
                continue
            start_min, end_min = HORIZON_MAP[horizon]
            for alert_threshold in alert_thresholds:
                metrics = compute_event_metrics_for_subset(
                    subset,
                    events_df,
                    start_min=start_min,
                    end_min=end_min,
                    alert_threshold=alert_threshold,
                )
                detail_rows.append(
                    {
                        "script": script,
                        "group": group,
                        "horizon": horizon,
                        "coin": coin,
                        "alert_threshold": alert_threshold,
                        **metrics,
                    }
                )

    detail_df = pd.DataFrame(detail_rows)
    detail_path = output_root / "event_level_metrics_by_coin.csv"
    detail_df.to_csv(detail_path, index=False, encoding="utf-8")

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
    summary_path = output_root / "event_level_metrics_summary.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8")

    coin_event_df = pd.DataFrame(coin_event_summary)
    coin_event_path = output_root / "event_catalog_summary.csv"
    coin_event_df.to_csv(coin_event_path, index=False, encoding="utf-8")

    primary = summary_df[summary_df["alert_threshold"] == 0.01].copy()
    primary = primary.sort_values("micro_event_recall", ascending=False, na_position="last")
    md_lines = [
        "# Event-Level Metrics",
        "",
        "Definition:",
        f"- Event: |close-1| >= {args.event_threshold:.2f}, minimum duration {args.min_event_duration} minutes, merge gap {args.merge_gap} minutes.",
        "- Event recall: fraction of actual events with at least one alert in the valid pre-event horizon window.",
        "- Alert hit rate: fraction of alert minutes that correctly point to at least one future event start in the horizon window.",
        "- Lead time: minutes from the earliest valid alert to the event start.",
        "",
        "## Event Catalog Summary",
    ]
    for row in coin_event_df.itertuples(index=False):
        md_lines.append(f"- {row.coin}: events={row.event_count}, total_event_minutes={row.total_event_minutes}")

    md_lines.append("")
    md_lines.append("## Top Results at Alert Threshold 0.01")
    for row in primary.head(20).itertuples(index=False):
        micro_recall = "NaN" if pd.isna(row.micro_event_recall) else f"{row.micro_event_recall:.4f}"
        hit_rate = "NaN" if pd.isna(row.macro_alert_hit_rate) else f"{row.macro_alert_hit_rate:.4f}"
        lead = "NaN" if pd.isna(row.macro_mean_lead_minutes) else f"{row.macro_mean_lead_minutes:.1f}"
        md_lines.append(
            f"- {row.script} / {row.group} / {row.horizon}: "
            f"micro_event_recall={micro_recall}, macro_alert_hit_rate={hit_rate}, "
            f"mean_lead_minutes={lead}, events={int(row.event_count)}"
        )

    md_path = output_root / "event_level_metrics_report_utf8.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    print(f"saved {detail_path}")
    print(f"saved {summary_path}")
    print(f"saved {coin_event_path}")
    print(f"saved {md_path}")
    print(f"detail_rows {len(detail_df)}")
    print(f"summary_rows {len(summary_df)}")


if __name__ == "__main__":
    main()
