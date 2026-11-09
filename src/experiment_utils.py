import os
import random

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, mean_squared_error, roc_auc_score


TRAIN_COINS = ['USDC', 'DAI', 'SUSD']
TEST_COINS = ['EURC', 'TUSD', 'XUSD', 'USDY', 'UST']
HORIZONS = ['0-3h', '3-6h', '6-9h', '6-12h', '12-18h', '18-24h', '24-30h', '48-54h']
DEFAULT_THRESHOLDS = [0.0, 0.01, 0.1, 0.2]
DEFAULT_DATA_PATH = 'data/ml_predictions/'
STATS_COLS = ['volatility_long', 'dev_mean', 'dev_max', 'return_cum', 'rolling_std', 'rolling_skew']
DEFAULT_BALANCED_MSE_THRESHOLD = 0.01
DEFAULT_BALANCED_MSE_BUCKETS = [0.0, 0.01, 0.1, 0.5, 1.01]


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_features(coin, data_path=DEFAULT_DATA_PATH):
    files = [
        os.path.join(data_path, f'features_{coin}.csv'),
        os.path.join(data_path, f'features_{coin}_bocd.csv'),
    ]
    for path in files:
        if os.path.exists(path):
            return pd.read_csv(path)
    raise FileNotFoundError(f'Feature file not found for {coin}. Checked: {files}')


def build_features_v3(df, window=10, drop_na=True, include_deviation=True):
    result = pd.DataFrame(index=df.index)
    result['price'] = df['close']
    result['log_return'] = np.log(df['close'] / df['close'].shift(1))

    if include_deviation:
        result['deviation'] = df['close'] - 1.0

    result['rolling_mean'] = df['close'].rolling(window).mean()
    result['rolling_std'] = result['log_return'].rolling(window).std()
    result['rolling_skew'] = result['log_return'].rolling(window * 3).skew()
    result['rolling_kurt'] = result['log_return'].rolling(window * 3).kurt()

    rolling_max = df['close'].rolling(window * 6, min_periods=1).max()
    result['max_drawdown'] = (df['close'] - rolling_max) / rolling_max

    vol_short = result['log_return'].rolling(window).std()
    vol_long = result['log_return'].rolling(window * 3).std()
    result['vol_ratio'] = vol_short / (vol_long + 1e-10)

    if include_deviation:
        result['deviation_diff'] = result['deviation'].diff()

    result['volatility_long'] = result['log_return'].rolling(window * 6).std()

    if include_deviation:
        result['dev_mean'] = result['deviation'].rolling(window).mean()
        result['dev_max'] = result['deviation'].rolling(window).max()

    result['return_cum'] = result['log_return'].rolling(window).sum()

    if 'volume' in df.columns:
        result['volume'] = df['volume']
        result['volume_ratio'] = df['volume'] / df['volume'].rolling(window).mean()

    if drop_na:
        result = result.dropna()

    valid_indices = result.index.to_list()
    valid_datetimes = df.loc[valid_indices, 'datetime'].values if 'datetime' in df.columns else None
    return result.values, result.columns.to_list(), valid_indices, valid_datetimes


def compute_multi_level_labels(df, horizon='0-3h', valid_idx=None):
    horizon_map = {
        '0-3h': (1, 180),
        '3-6h': (181, 360),
        '6-9h': (361, 540),
        '6-12h': (361, 720),
        '12-18h': (721, 1080),
        '18-24h': (1081, 1440),
        '24-30h': (1441, 1800),
        '48-54h': (2881, 3240),
    }
    start, end = horizon_map.get(horizon, (1, 180))

    dev = np.abs(df['close'] - 1.0) * 100
    labels_soft = np.full(len(df), np.nan, dtype=float)
    labels_level = np.full(len(df), -1, dtype=int)

    max_start_idx = len(df) - end - 1
    for i in range(max_start_idx + 1):
        future_dev = dev.iloc[i + start:i + end + 1]
        labels_soft[i] = (future_dev >= 5.0).mean()

        max_dev = future_dev.max()
        if max_dev < 1.0:
            labels_level[i] = 0
        elif max_dev < 5.0:
            labels_level[i] = 1
        else:
            labels_level[i] = 2

    if valid_idx is not None:
        labels_soft = labels_soft[valid_idx]
        labels_level = labels_level[valid_idx]
    else:
        labels_soft = labels_soft[:max_start_idx + 1]
        labels_level = labels_level[:max_start_idx + 1]

    return labels_soft, labels_level


def create_sequences(X, y, seq_len):
    X_seq, y_seq = [], []
    for i in range(seq_len, len(X)):
        X_seq.append(X[i - seq_len:i])
        y_seq.append(y[i])
    return np.array(X_seq), np.array(y_seq)


def extract_stats_features(features, col_names, min_len, valid_mask):
    stats_idx = [col_names.index(c) for c in STATS_COLS if c in col_names]
    if not stats_idx:
        return None

    stats = features[:, stats_idx]
    stats = stats[-min_len:]
    stats = stats[valid_mask]
    return stats


def align_sequence_datetimes(valid_datetimes, min_len, valid_mask, seq_len):
    datetimes = np.asarray(valid_datetimes)[-min_len:]
    datetimes = datetimes[valid_mask]
    return datetimes[seq_len:]


def prepare_coin_sequences(
    coin,
    data_path=DEFAULT_DATA_PATH,
    horizon='0-3h',
    include_deviation=True,
    seq_len=60,
    min_samples=100,
    feature_cols=None,
):
    df = load_features(coin, data_path)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime').reset_index(drop=True)

    required_cols = ['close']
    if 'volume' in df.columns:
        required_cols.append('volume')
    df = df.dropna(subset=required_cols)

    features, col_names, valid_idx, valid_datetimes = build_features_v3(
        df, window=10, drop_na=True, include_deviation=include_deviation
    )
    if feature_cols is not None:
        feature_idx = [col_names.index(c) for c in feature_cols]
        features = features[:, feature_idx]
        selected_col_names = feature_cols
    else:
        selected_col_names = col_names

    y_soft, _ = compute_multi_level_labels(df, horizon, valid_idx=valid_idx)
    min_len = min(len(features), len(y_soft))

    X = features[-min_len:]
    y_soft = y_soft[-min_len:]
    valid_mask = ~np.isnan(y_soft)
    X = X[valid_mask]
    y_soft = y_soft[valid_mask]

    if len(y_soft) < min_samples:
        return None

    stats = None
    if feature_cols is None:
        stats = extract_stats_features(features, col_names, min_len, valid_mask)

    X_seq, y_seq = create_sequences(X, y_soft, seq_len)
    if stats is not None:
        stats_seq, _ = create_sequences(stats, y_soft, seq_len)
        if stats_seq.ndim == 1:
            stats_seq = stats_seq.reshape(-1, 1)
    else:
        stats_seq = None

    seq_datetimes = align_sequence_datetimes(valid_datetimes, min_len, valid_mask, seq_len)

    return {
        'df': df,
        'X': X,
        'y_soft': y_soft,
        'X_seq': X_seq,
        'y_seq': y_seq,
        'stats_seq': stats_seq,
        'seq_datetimes': seq_datetimes,
        'feature_names': selected_col_names,
    }


def prepare_tabular_dataset(
    coin,
    data_path=DEFAULT_DATA_PATH,
    horizon='0-3h',
    include_deviation=True,
    feature_cols=None,
    min_samples=100,
):
    prepared = prepare_coin_sequences(
        coin=coin,
        data_path=data_path,
        horizon=horizon,
        include_deviation=include_deviation,
        seq_len=1,
        min_samples=min_samples,
        feature_cols=feature_cols,
    )
    if prepared is None:
        return None

    return {
        'X': prepared['X_seq'][:, -1, :],
        'y_soft': prepared['y_seq'],
        'datetimes': prepared['seq_datetimes'],
        'feature_names': prepared['feature_names'],
    }


def evaluate_binary_metrics(y_soft, preds, thresholds=DEFAULT_THRESHOLDS):
    metrics = []
    for threshold in thresholds:
        y_binary = (y_soft > threshold).astype(int)
        pos = int(y_binary.sum())
        neg = len(y_binary) - pos
        if pos == 0 or neg == 0:
            metrics.append({
                'threshold': threshold,
                'auc': np.nan,
                'ap': np.nan,
                'pos': pos,
                'neg': neg,
            })
            continue
        metrics.append({
            'threshold': threshold,
            'auc': roc_auc_score(y_binary, preds),
            'ap': average_precision_score(y_binary, preds),
            'pos': pos,
            'neg': neg,
        })
    return metrics


def _safe_mean(values):
    return float(np.mean(values)) if len(values) > 0 else np.nan


def balanced_mse_components(y_true, preds, threshold=DEFAULT_BALANCED_MSE_THRESHOLD):
    y_true = np.asarray(y_true, dtype=float)
    preds = np.asarray(preds, dtype=float)

    pos_mask = y_true > threshold
    neg_mask = ~pos_mask

    pos_err = (preds[pos_mask] - y_true[pos_mask]) ** 2
    neg_err = (preds[neg_mask] - y_true[neg_mask]) ** 2

    pos_mse = _safe_mean(pos_err)
    neg_mse = _safe_mean(neg_err)

    parts = [x for x in (pos_mse, neg_mse) if not np.isnan(x)]
    balanced_mse = float(np.mean(parts)) if parts else np.nan
    return balanced_mse, pos_mse, neg_mse


def bucket_balanced_mse(y_true, preds, buckets=DEFAULT_BALANCED_MSE_BUCKETS):
    y_true = np.asarray(y_true, dtype=float)
    preds = np.asarray(preds, dtype=float)

    bucket_mses = []
    for lo, hi in zip(buckets[:-1], buckets[1:]):
        mask = (y_true >= lo) & (y_true < hi)
        if mask.any():
            bucket_mses.append(float(np.mean((preds[mask] - y_true[mask]) ** 2)))

    return float(np.mean(bucket_mses)) if bucket_mses else np.nan


def summarize_regression(y_true, preds):
    y_true = np.asarray(y_true, dtype=float)
    preds = np.asarray(preds, dtype=float)

    mse = mean_squared_error(y_true, preds)
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(preds - y_true)))
    y_std = float(np.std(y_true))
    p_std = float(np.std(preds))
    if len(y_true) <= 1 or y_std == 0.0 or p_std == 0.0:
        pearson = np.nan
    else:
        pearson = float(np.corrcoef(y_true, preds)[0, 1])

    balanced_mse_t001, pos_mse_t001, neg_mse_t001 = balanced_mse_components(
        y_true, preds, threshold=DEFAULT_BALANCED_MSE_THRESHOLD
    )
    bucket_balanced_mse_val = bucket_balanced_mse(y_true, preds)

    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'pearson': pearson,
        'balanced_mse_t001': balanced_mse_t001,
        'pos_mse_t001': pos_mse_t001,
        'neg_mse_t001': neg_mse_t001,
        'bucket_balanced_mse': bucket_balanced_mse_val,
        'prediction_min': float(np.min(preds)),
        'prediction_max': float(np.max(preds)),
    }
