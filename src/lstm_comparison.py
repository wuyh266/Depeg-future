"""
LSTM comparison: fix LSTM architecture and compare feature groups.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from experiment_utils import (
    DEFAULT_DATA_PATH,
    DEFAULT_THRESHOLDS,
    HORIZONS,
    TEST_COINS,
    TRAIN_COINS,
    evaluate_binary_metrics,
    prepare_coin_sequences,
    set_seed,
    summarize_regression,
)


class SimpleLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=32):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        z = h[-1]
        return torch.sigmoid(self.head(z)), z


def resolve_device(device_arg="auto"):
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")
    if device_arg == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_horizons(horizons_arg):
    if not horizons_arg:
        return list(HORIZONS)
    horizons = [h.strip() for h in horizons_arg.split(",") if h.strip()]
    invalid = [h for h in horizons if h not in HORIZONS]
    if invalid:
        raise ValueError(f"Invalid horizons: {invalid}. Allowed: {HORIZONS}")
    return horizons


def slugify(name):
    return name.lower().replace(":", "").replace(" ", "_").replace("-", "_")


def get_completed_horizons(existing_df, model_name):
    if existing_df is None or existing_df.empty:
        return set()
    subset = existing_df[existing_df["model"] == model_name]
    if subset.empty:
        return set()
    return set(subset["horizon"].astype(str).tolist())


def save_model_artifact(model, scaler, model_name, horizon, output_dir):
    models_dir = Path(output_dir) / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / f"{slugify(model_name)}__{horizon}.pt"
    payload = {
        "model_state_dict": model.state_dict(),
        "scaler_mean": scaler.mean_,
        "scaler_scale": scaler.scale_,
        "model_name": model_name,
        "horizon": horizon,
    }
    torch.save(payload, model_path)
    return str(model_path)


def train_model_for_horizon(
    horizon,
    feature_cols,
    data_path,
    seq_len=60,
    epochs=10,
    batch_size=256,
    device=torch.device("cpu"),
):
    train_X, train_y = [], []
    for coin in TRAIN_COINS:
        prepared = prepare_coin_sequences(
            coin, data_path=data_path, horizon=horizon, seq_len=seq_len, feature_cols=feature_cols
        )
        if prepared is None:
            continue
        train_X.append(prepared["X_seq"])
        train_y.append(prepared["y_seq"])

    if not train_X:
        raise RuntimeError("No valid training samples found.")

    X_train = np.concatenate(train_X, axis=0)
    y_train = np.concatenate(train_y)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)

    model = SimpleLSTM(X_train.shape[-1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    dataset = TensorDataset(
        torch.FloatTensor(X_train_scaled),
        torch.FloatTensor(y_train).unsqueeze(1),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for _ in range(epochs):
        model.train()
        for X_t, y_t in loader:
            X_t = X_t.to(device)
            y_t = y_t.to(device)
            pred, _ = model(X_t)
            loss = criterion(pred, y_t)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return model, scaler


def evaluate_model(
    model,
    scaler,
    horizon,
    data_path,
    feature_cols=None,
    seq_len=60,
    device=torch.device("cpu"),
    eval_batch_size=256,
    model_name="",
):
    results = {}
    prediction_frames = []
    for coin in TEST_COINS:
        prepared = prepare_coin_sequences(
            coin, data_path=data_path, horizon=horizon, seq_len=seq_len, feature_cols=feature_cols
        )
        if prepared is None:
            continue

        X_seq = prepared["X_seq"]
        y_seq = prepared["y_seq"]
        X_scaled = scaler.transform(X_seq.reshape(-1, X_seq.shape[-1])).reshape(X_seq.shape)

        model.eval()
        with torch.no_grad():
            preds = []
            for i in range(0, len(X_scaled), eval_batch_size):
                batch = torch.FloatTensor(X_scaled[i : i + eval_batch_size]).to(device)
                pred, _ = model(batch)
                preds.append(pred.cpu().numpy().flatten())
            preds = np.concatenate(preds)

        results[coin] = {
            **summarize_regression(y_seq, preds),
            "threshold_metrics": evaluate_binary_metrics(y_seq, preds, DEFAULT_THRESHOLDS),
        }
        n = min(len(preds), len(y_seq), len(prepared["seq_datetimes"]))
        prediction_frames.append(
            pd.DataFrame(
                {
                    "datetime": prepared["seq_datetimes"][:n],
                    "model": model_name,
                    "horizon": horizon,
                    "coin": coin,
                    "prediction": preds[:n],
                    "y_soft": y_seq[:n],
                }
            )
        )

    predictions_df = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
    return results, predictions_df


def flatten_results(model_name, horizon, results):
    rows = []
    for coin, metrics in results.items():
        base = {
            "model": model_name,
            "horizon": horizon,
            "coin": coin,
            "mse": metrics["mse"],
            "rmse": metrics["rmse"],
            "mae": metrics["mae"],
            "pearson": metrics["pearson"],
            "balanced_mse_t001": metrics["balanced_mse_t001"],
            "pos_mse_t001": metrics["pos_mse_t001"],
            "neg_mse_t001": metrics["neg_mse_t001"],
            "bucket_balanced_mse": metrics["bucket_balanced_mse"],
            "prediction_min": metrics["prediction_min"],
            "prediction_max": metrics["prediction_max"],
        }
        for threshold_row in metrics["threshold_metrics"]:
            rows.append({**base, **threshold_row})
    return rows


def print_horizon_summary(model_name, horizon, results):
    aucs = []
    for coin, metrics in results.items():
        threshold_metric = next((m for m in metrics["threshold_metrics"] if m["threshold"] == 0.01), None)
        auc = threshold_metric["auc"] if threshold_metric else np.nan
        aucs.append(auc)
        print(
            f"  {horizon} | {coin}: mse={metrics['mse']:.4f}, "
            f"balanced_mse@0.01={metrics['balanced_mse_t001']:.4f}, auc@0.01={auc:.4f}, "
            f"pred_range=[{metrics['prediction_min']:.4f}, {metrics['prediction_max']:.4f}]"
        )
    print(f"  {horizon} | {model_name} average AUC@0.01: {np.nanmean(aucs):.4f}")


def main():
    parser = argparse.ArgumentParser(description="LSTM comparison")
    parser.add_argument("--data-path", type=str, default=DEFAULT_DATA_PATH, help="Feature data directory")
    parser.add_argument("--output-dir", type=str, default="data/experiment_outputs/lstm_comparison", help="Output dir")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="Device")
    parser.add_argument("--batch-size", type=int, default=256, help="Train batch size")
    parser.add_argument("--eval-batch-size", type=int, default=256, help="Eval batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs")
    parser.add_argument("--seq-len", type=int, default=60, help="Sequence length")
    parser.add_argument("--horizons", type=str, default=",".join(HORIZONS), help="Comma-separated horizons")
    parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Skip model+horizon pairs already in lstm_comparison_results.csv",
    )
    parser.add_argument("--no-save-models", action="store_true", help="Do not save trained model artifacts")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = output_dir / "metrics"
    preds_dir = output_dir / "predictions"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    preds_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    device = resolve_device(args.device)
    horizons = parse_horizons(args.horizons)

    result_path = output_dir / "lstm_comparison_results.csv"
    pred_path = output_dir / "lstm_comparison_minute_predictions.csv"
    existing_df = pd.read_csv(result_path) if (args.resume and result_path.exists()) else pd.DataFrame()
    existing_pred_df = pd.read_csv(pred_path) if (args.resume and pred_path.exists()) else pd.DataFrame()

    print("=" * 70)
    print("LSTM Comparison")
    print(f"Device: {device}")
    print("=" * 70)

    feature_configs = [
        ("Deviation Only", ["price", "deviation", "dev_mean", "dev_max", "deviation_diff"]),
        (
            "Volatility Only",
            [
                "price",
                "rolling_std",
                "rolling_skew",
                "rolling_kurt",
                "volatility_long",
                "vol_ratio",
                "volume",
                "volume_ratio",
            ],
        ),
        ("Full Features", None),
    ]

    all_rows = []
    all_new_preds = []

    for name, cols in feature_configs:
        completed_horizons = get_completed_horizons(existing_df, name)
        print(f"\n[{name}]")
        for horizon in horizons:
            if horizon in completed_horizons:
                print(f"  [skip] {name} / {horizon} already exists")
                continue

            model, scaler = train_model_for_horizon(
                horizon=horizon,
                feature_cols=cols,
                data_path=args.data_path,
                seq_len=args.seq_len,
                epochs=args.epochs,
                batch_size=args.batch_size,
                device=device,
            )
            if not args.no_save_models:
                model_path = save_model_artifact(model, scaler, name, horizon, output_dir)
                print(f"  saved model: {model_path}")

            results, preds_df = evaluate_model(
                model,
                scaler,
                horizon=horizon,
                data_path=args.data_path,
                feature_cols=cols,
                seq_len=args.seq_len,
                device=device,
                eval_batch_size=args.eval_batch_size,
                model_name=name,
            )
            print_horizon_summary(name, horizon, results)

            metric_rows = flatten_results(name, horizon, results)
            all_rows.extend(metric_rows)
            metric_file = metrics_dir / f"{slugify(name)}__{horizon}_metrics.csv"
            pd.DataFrame(metric_rows).to_csv(metric_file, index=False)
            print(f"  saved metrics: {metric_file}")

            pred_file = preds_dir / f"{slugify(name)}__{horizon}_minute_predictions.csv"
            preds_df.to_csv(pred_file, index=False)
            print(f"  saved predictions: {pred_file}")
            if not preds_df.empty:
                all_new_preds.append(preds_df)

    new_rows_df = pd.DataFrame(all_rows)
    if not existing_df.empty:
        merged_rows = pd.concat([existing_df, new_rows_df], ignore_index=True)
        merged_rows = merged_rows.drop_duplicates(subset=["model", "horizon", "coin", "threshold"], keep="last")
    else:
        merged_rows = new_rows_df
    merged_rows.to_csv(result_path, index=False)
    print(f"\nSaved results: {result_path}")

    if all_new_preds:
        new_pred_df = pd.concat(all_new_preds, ignore_index=True)
        if not existing_pred_df.empty:
            merged_preds = pd.concat([existing_pred_df, new_pred_df], ignore_index=True)
            merged_preds = merged_preds.drop_duplicates(subset=["model", "horizon", "coin", "datetime"], keep="last")
        else:
            merged_preds = new_pred_df
        merged_preds.to_csv(pred_path, index=False)
        print(f"Saved minute predictions: {pred_path}")
    else:
        print("No new horizons were run; minute predictions unchanged.")


if __name__ == "__main__":
    main()
