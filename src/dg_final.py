"""
DG (Domain Generalization) Feature Optimization - 简化版
核心改进：Anti-collapse Regularization
"""

import argparse
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score, average_precision_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import pickle
import os

HORIZON_MAP = {
    '0-3h': (1, 180),
    '3-6h': (181, 360),
    '6-9h': (361, 540),
    '6-12h': (361, 720),
    '12-18h': (721, 1080),
    '18-24h': (1081, 1440),
    '24-30h': (1441, 1800),
    '48-54h': (2881, 3240),
}


# ==================== 特征构建 ====================

def build_features_v3(df, window=10, drop_na=True, include_deviation=True):
    """构建增强特征集 + 统计特征（用于 hybrid environment）
    
    Args:
        include_deviation: 是否包含deviation相关特征（用于消融实验）
    """
    result = pd.DataFrame(index=df.index)
    
    # 基础特征
    result['price'] = df['close']
    result['log_return'] = np.log(df['close'] / df['close'].shift(1))
    
    # Deviation相关特征（可选）
    if include_deviation:
        result['deviation'] = df['close'] - 1.0
    
    # 滚动特征
    result['rolling_mean'] = df['close'].rolling(window).mean()
    result['rolling_std'] = result['log_return'].rolling(window).std()
    
    # Pseudo-mechanism 信号
    result['rolling_skew'] = result['log_return'].rolling(window*3).skew()
    result['rolling_kurt'] = result['log_return'].rolling(window*3).kurt()
    
    rolling_max = df['close'].rolling(window*6, min_periods=1).max()
    result['max_drawdown'] = (df['close'] - rolling_max) / rolling_max
    
    vol_short = result['log_return'].rolling(window).std()
    vol_long = result['log_return'].rolling(window*3).std()
    result['vol_ratio'] = vol_short / (vol_long + 1e-10)
    
    if include_deviation:
        result['deviation_diff'] = result['deviation'].diff()
    
    # ========== 统计特征（用于 hybrid environment）==========
    result['volatility_long'] = result['log_return'].rolling(window*6).std()
    
    if include_deviation:
        result['dev_mean'] = result['deviation'].rolling(window).mean()
        result['dev_max'] = result['deviation'].rolling(window).max()
    
    result['return_cum'] = result['log_return'].rolling(window).sum()
    
    if 'volume' in df.columns:
        result['volume'] = df['volume']
        result['volume_ratio'] = df['volume'] / df['volume'].rolling(window).mean()
    
    if drop_na:
        result = result.dropna()
    
    valid_indices = result.index.tolist()
    valid_datetimes = df.loc[valid_indices, 'datetime'].values if 'datetime' in df.columns else None
    
    return result.values, result.columns.tolist(), valid_indices, valid_datetimes


def build_features_v3_standard(df, window=10):
    """仅返回标准特征（用于模型输入）"""
    features, _, _, _ = build_features_v3(df, window, drop_na=True)
    return features


def build_features_v3_stats(df, window=10):
    """返回统计特征（用于 hybrid environment）"""
    features, col_names, _, _ = build_features_v3(df, window, drop_na=True)
    stats_cols = ['volatility_long', 'dev_mean', 'dev_max', 'return_cum', 'rolling_std', 'rolling_skew']
    stats_idx = [col_names.index(c) for c in stats_cols if c in col_names]
    return features[:, stats_idx] if stats_idx else None


# ==================== 分层 Label ====================

def compute_multi_level_labels(df, horizon='0-3h', valid_idx=None):
    """
    标签定义：
    - '0-3h': 未来 0-3 小时内的脱锚比例 (t+1 到 t+180)
    - '3-6h': 未来 3-6 小时内的脱锚比例 (t+181 到 t+360)
    - '6-12h': 未来 6-12 小时内的脱锚比例 (t+361 到 t+720)
    
    valid_idx: 可选的保留索引，用于与特征对齐
    """
    horizon_map = {
        '6-9h': (361, 540),
        '0-3h': (1, 180),       # 未来3小时内
        '3-6h': (181, 360),     # 未来3-6小时
        '6-12h': (361, 720),    # 未来6-12小时
        '12-18h': (721, 1080),  # 未来12-18小时
        '18-24h': (1081, 1440), # 未来18-24小时
        '24-30h': (1441, 1800), # 未来24-30小时
        '48-54h': (2881, 3240), # 未来48-54小时
    }
    
    start, end = horizon_map.get(horizon, (1, 180))
    
    dev = abs(df['close'] - 1.0) * 100
    
    labels_soft = np.full(len(df), np.nan, dtype=float)
    labels_level = np.full(len(df), -1, dtype=int)
    
    # 需要留出足够的数据窗口
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
    
    # 只返回有效标签，如果提供了 valid_idx，则返回对应的标签
    if valid_idx is not None:
        labels_soft = labels_soft[valid_idx]
        labels_level = labels_level[valid_idx]
    else:
        labels_soft = labels_soft[:max_start_idx + 1]
        labels_level = labels_level[:max_start_idx + 1]
    
    return labels_soft, labels_level


# ==================== 模型定义 ====================

class PriceTransformer(nn.Module):
    def __init__(self, input_dim=17, d_model=128, nhead=4, num_layers=2):
        super().__init__()
        
        self.input_proj = nn.Linear(input_dim, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=256, 
            dropout=0.3, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
    
    def forward(self, x):
        x = self.input_proj(x)
        h = self.encoder(x)
        z = h[:, -1]
        return z


class DGModelFinal(nn.Module):
    """DG 模型：Hybrid Environment (z + stats) + Continuous Risk"""
    def __init__(self, input_dim=17, d_model=128, num_env=5, temperature=0.1, stats_dim=6, gate_floor=0.0):
        super().__init__()
        
        self.num_env = num_env
        self.temperature = temperature
        self.stats_dim = stats_dim
        self.gate_floor = gate_floor
        
        self.encoder = PriceTransformer(input_dim=input_dim, d_model=d_model)
        
        if stats_dim > 0:
            self.stats_encoder = nn.Linear(stats_dim, d_model // 4)
            self.env_encoder = nn.Linear(d_model + d_model // 4, d_model)
        else:
            self.stats_encoder = None
            self.env_encoder = nn.Linear(d_model, d_model)
        
        self.env_heads = nn.ModuleList([
            nn.Linear(d_model, 1) for _ in range(num_env)
        ])
        
        self.register_buffer("env_centers", torch.randn(num_env, d_model))
    
    def compute_hybrid_env(self, z, stats=None):
        if stats is not None and self.stats_encoder is not None:
            stats_emb = F.relu(self.stats_encoder(stats))
            env_input = torch.cat([z, stats_emb], dim=-1)
        else:
            env_input = z
        
        env_repr = F.relu(self.env_encoder(env_input))
        dist = torch.cdist(env_repr, self.env_centers)
        logits = -dist / self.temperature
        p_env = F.softmax(logits, dim=1)
        if self.gate_floor > 0:
            p_env = (1.0 - self.gate_floor) * p_env + self.gate_floor / self.num_env
        return p_env, env_repr
    
    def forward(self, x, stats=None):
        z = self.encoder(x)
        p_env, env_repr = self.compute_hybrid_env(z, stats)
        
        preds = torch.stack([torch.sigmoid(head(env_repr)) for head in self.env_heads], dim=1)
        y = (p_env.unsqueeze(-1) * preds).sum(dim=1)
        
        return y, z, p_env, preds


class ERMModel(nn.Module):
    def __init__(self, input_dim=17, d_model=128):
        super().__init__()
        
        self.encoder = PriceTransformer(input_dim=input_dim, d_model=d_model)
        self.head = nn.Linear(d_model, 1)
    
    def forward(self, x):
        z = self.encoder(x)
        y = torch.sigmoid(self.head(z))
        return y, z


# ==================== Loss 函数 ====================

def vrex_loss(preds, y, p_env, lambda_vrex=1.0):
    """VREx Loss"""
    risks = []
    
    for k in range(preds.shape[1]):
        pred_k = preds[:, k]
        loss_k = (pred_k - y) ** 2
        weight = p_env[:, k].unsqueeze(-1)
        weight_sum = weight.sum().clamp_min(1e-8)
        risk_k = (loss_k * weight).sum() / weight_sum
        risks.append(risk_k)
    
    risks = torch.stack(risks)
    mean_risk = risks.mean()
    var_risk = ((risks - mean_risk) ** 2).mean()
    
    return mean_risk + lambda_vrex * var_risk


def entropy_loss(p_env):
    """Anti-collapse: 鼓励均匀分布"""
    return (p_env * torch.log(p_env + 1e-10)).sum(dim=1).mean()


def routing_entropy(p_env):
    return -(p_env * torch.log(p_env + 1e-10)).sum(dim=1)


def min_routing_entropy_penalty(p_env, min_entropy):
    if min_entropy <= 0:
        return p_env.new_tensor(0.0)
    entropy = routing_entropy(p_env)
    return F.relu(min_entropy - entropy).mean()


def erm_loss(preds, y):
    return ((preds - y) ** 2).mean()


# ==================== 辅助函数 ====================

def create_sequences(X, y, seq_len):
    X_seq, y_seq = [], []
    for i in range(seq_len, len(X)):
        X_seq.append(X[i-seq_len:i])
        y_seq.append(y[i])
    return np.array(X_seq), np.array(y_seq)


def align_sequence_datetimes(valid_datetimes, min_len, valid_mask, seq_len):
    """Align timestamps with sequence labels after trimming and NaN filtering."""
    datetimes = np.asarray(valid_datetimes)[-min_len:]
    datetimes = datetimes[valid_mask]
    return datetimes[seq_len:]


def predict_batch(model, X_seq, stats_seq, scaler, stats_scaler, device, batch_size=512):
    """批量预测"""
    X_flat = X_seq.reshape(-1, X_seq.shape[-1])
    X_scaled = scaler.transform(X_flat).reshape(X_seq.shape)
    
    if stats_seq is not None and stats_scaler is not None:
        stats_scaled = stats_scaler.transform(stats_seq)
    else:
        stats_scaled = None
    
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X_scaled), batch_size):
            X_batch = torch.FloatTensor(X_scaled[i:i+batch_size]).to(device)
            if stats_scaled is not None:
                stats_batch = torch.FloatTensor(stats_scaled[i:i+batch_size]).to(device)
            else:
                stats_batch = None
            pred, _, _, _ = model(X_batch, stats_batch)
            preds.append(pred.cpu().numpy().flatten())
    
    return np.concatenate(preds)


def load_features(coin, data_path='../data/ml_predictions/'):
    files = [
        os.path.join(data_path, f'features_{coin}.csv'),
        os.path.join(data_path, f'features_{coin}_bocd.csv')
    ]
    
    for f in files:
        if os.path.exists(f):
            return pd.read_csv(f)
    
    raise FileNotFoundError(f"Feature file not found for {coin}. Checked: {files}")


def extract_stats_features(features, col_names, min_len, valid_mask):
    stats_cols = ['volatility_long', 'dev_mean', 'dev_max', 'return_cum', 'rolling_std', 'rolling_skew']
    stats_idx = [col_names.index(c) for c in stats_cols if c in col_names]

    if not stats_idx:
        return None

    stats = features[:, stats_idx]
    stats = stats[-min_len:]
    stats = stats[valid_mask]
    return stats


def prepare_coin_sequences(coin, data_path, horizon, include_deviation=True, seq_len=60, min_samples=100):
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
    y_soft, _ = compute_multi_level_labels(df, horizon, valid_idx=valid_idx)

    min_len = min(len(features), len(y_soft))
    X = features[-min_len:]
    y_soft = y_soft[-min_len:]

    valid_mask = ~np.isnan(y_soft)
    X = X[valid_mask]
    y_soft = y_soft[valid_mask]

    if len(y_soft) < min_samples:
        return None

    stats = extract_stats_features(features, col_names, min_len, valid_mask)

    X_seq, y_seq = create_sequences(X, y_soft, seq_len)
    if stats is not None:
        stats_windows, _ = create_sequences(stats, y_soft, seq_len)
        if stats_windows.ndim == 3:
            # Use the latest stats snapshot for each sequence so stats_encoder sees one vector per sample.
            stats_seq = stats_windows[:, -1, :]
        else:
            stats_seq = stats_windows
        if stats_seq.ndim == 1:
            stats_seq = stats_seq.reshape(-1, 1)
    else:
        stats_seq = None

    seq_datetimes = align_sequence_datetimes(valid_datetimes, min_len, valid_mask, seq_len)

    return {
        'df': df,
        'X_seq': X_seq,
        'y_seq': y_seq,
        'stats_seq': stats_seq,
        'seq_datetimes': seq_datetimes,
        'feature_dim': X_seq.shape[-1],
    }


def fit_kmeans(model, X_scaled, stats_scaled=None, num_env=5, device='cpu'):
    model.eval()
    model = model.to(device)

    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_scaled).to(device)
        stats_tensor = torch.FloatTensor(stats_scaled).to(device) if stats_scaled is not None else None
        env_repr_list = []

        for i in range(0, len(X_tensor), 512):
            X_batch = X_tensor[i:i+512]
            stats_batch = stats_tensor[i:i+512] if stats_tensor is not None else None
            z = model.encoder(X_batch)
            _, env_repr = model.compute_hybrid_env(z, stats_batch)
            env_repr_list.append(env_repr)

        env_repr_all = torch.cat(env_repr_list, dim=0).cpu().numpy()

    kmeans = KMeans(n_clusters=num_env, random_state=0, n_init=10).fit(env_repr_all)
    centers = torch.tensor(kmeans.cluster_centers_, dtype=torch.float)

    return centers


def shuffle_time_within_sequence(X_seq, seed=42):
    shuffled = X_seq.copy()
    rng = np.random.default_rng(seed)
    for i in range(len(shuffled)):
        idx = rng.permutation(shuffled.shape[1])
        shuffled[i] = shuffled[i, idx, :]
    return shuffled


def safe_binary_auc(y_true, y_score):
    y_true = np.asarray(y_true).astype(int)
    pos = int(y_true.sum())
    neg = len(y_true) - pos
    if pos == 0 or neg == 0:
        return np.nan
    return roc_auc_score(y_true, y_score)


def linear_anneal(start_value, end_value, epoch_idx, total_epochs):
    if total_epochs <= 1:
        return end_value
    progress = epoch_idx / float(total_epochs - 1)
    return start_value + (end_value - start_value) * progress


def refresh_env_centers(model, X_scaled, stats_scaled, num_env, device):
    centers = fit_kmeans(model, X_scaled, stats_scaled, num_env=num_env, device=device)
    model.env_centers.data = centers.to(device)
    return centers


# ==================== Sanity Check ====================

def run_sanity_check(dg_model, scaler, stats_scaler, TEST_COINS, HORIZON, THRESHOLDS, MODEL_PATH, device, DATA_PATH, INCLUDE_DEVIATION=True, SEQ_LEN=60):
    """
    Shuffle Time Sanity Check:
    - 只打乱 X_seq 顺序，y_seq 保持不变
    - 预期: Real AUC ≈ 高, Shuffled AUC ≈ 0.5
    - 输出每分钟预测结果到 CSV
    """
    
    print(f"\n{'='*60}")
    print("Sanity Check: Shuffle Time (只打乱 X_seq)")
    print(f"{'='*60}")
    
    all_real_preds = []
    all_shuffled_preds = []
    
    for coin in TEST_COINS:
        try:
            prepared = prepare_coin_sequences(
                coin, DATA_PATH, HORIZON, include_deviation=INCLUDE_DEVIATION, seq_len=SEQ_LEN
            )
        except FileNotFoundError:
            print(f"  {coin}: 数据文件不存在，跳过")
            continue

        if prepared is None:
            print(f"  {coin}: 有效样本不足，跳过")
            continue

        X_seq = prepared['X_seq']
        y_seq = prepared['y_seq']
        stats_seq = prepared['stats_seq']
        seq_datetimes = prepared['seq_datetimes']
        n_samples = len(X_seq)
        
        # Real prediction (原始顺序)
        real_preds = predict_batch(dg_model, X_seq, stats_seq, scaler, stats_scaler, device)
        
        # Shuffle Time: 只打乱 X_seq 和 stats_seq，y_seq 保持原始顺序
        np.random.seed(42)  # 可复现
        idx = np.random.permutation(n_samples)
        X_seq_shuffled = shuffle_time_within_sequence(X_seq, seed=42)
        stats_seq_shuffled = stats_seq.copy() if stats_seq is not None else None
        # y_seq 保持不变（原始时间顺序）
        
        # Shuffled prediction (用打乱的特征预测，但对比原始 y_seq)
        shuffled_preds = predict_batch(dg_model, X_seq_shuffled, stats_seq_shuffled, scaler, stats_scaler, device)
        
        # 保存每分钟预测结果（按原始时间顺序）
        pred_df = pd.DataFrame({
            'datetime': seq_datetimes[:n_samples],
            'coin': coin,
            'horizon': HORIZON,
            'y_true': y_seq,
            'pred_real': real_preds,
            'pred_shuffled': shuffled_preds,
            'shuffle_idx': idx  # 保存打乱索引，方便追溯
        })
        all_real_preds.append(pred_df)
        
        # 打印对比结果
        print(f"\n  {coin} ({n_samples} 样本):")
        print(f"    {'Threshold':<10} | {'Real AUC':>10} | {'Shuffled AUC':>13} | {'差值':>8}")
        print(f"    {'-'*10}-+-{'-'*10}-+-{'-'*13}-+-{'-'*8}")
        
        for THRESHOLD in THRESHOLDS:
            y_binary = (y_seq[:n_samples] > THRESHOLD).astype(int)
            pos = int(y_binary.sum())
            neg = len(y_binary) - pos
            
            if pos > 0 and neg > 0:
                real_auc = safe_binary_auc(y_binary, real_preds)
                shuffle_auc = safe_binary_auc(y_binary, shuffled_preds)
                diff = real_auc - shuffle_auc
                print(f"    t={THRESHOLD:<8} | {real_auc:>10.4f} | {shuffle_auc:>13.4f} | {diff:>+8.4f}")
    
    # 保存所有预测结果
    if all_real_preds:
        predictions_df = pd.concat(all_real_preds, ignore_index=True)
        sanity_output = os.path.join(MODEL_PATH, f'dg_final_{HORIZON}_sanity_check.csv')
        predictions_df.to_csv(sanity_output, index=False)
        print(f"\n  预测结果已保存: {sanity_output}")
        print(f"  总行数: {len(predictions_df)}")
        
        # 计算平均差异
        print(f"\n  汇总:")
        print(f"    Real AUC 均值: {predictions_df.groupby('coin').apply(lambda x: roc_auc_score((x['y_true'] > 0.01).astype(int), x['pred_real'])).mean():.4f}")
        print(f"    Shuffled AUC 均值: {predictions_df.groupby('coin').apply(lambda x: roc_auc_score((x['y_true'] > 0.01).astype(int), x['pred_shuffled'])).mean():.4f}")
        
        real_aucs = []
        shuffle_aucs = []
        for coin in predictions_df['coin'].unique():
            coin_df = predictions_df[predictions_df['coin'] == coin]
            if coin_df['y_true'].sum() > 0 and (1 - coin_df['y_true']).sum() > 0:
                real_aucs.append(safe_binary_auc((coin_df['y_true'] > 0.01).astype(int), coin_df['pred_real']))
                shuffle_aucs.append(safe_binary_auc((coin_df['y_true'] > 0.01).astype(int), coin_df['pred_shuffled']))
        
        if real_aucs and shuffle_aucs:
            avg_real = np.mean(real_aucs)
            avg_shuffle = np.mean(shuffle_aucs)
            print(f"\n  {'='*50}")
            if 0.45 <= avg_shuffle <= 0.55:
                print(f"  [PASS] Sanity Check passed! (Shuffled AUC = {avg_shuffle:.4f} ~= 0.5)")
            else:
                print(f"  [FAIL] Sanity Check failed! (Shuffled AUC = {avg_shuffle:.4f} ~= 0.5)")
            print(f"  {'='*50}")
    
    print(f"\n{'='*60}")
    print("Sanity Check 完成")
    print(f"{'='*60}")


# ==================== 主程序 ====================

def run_sanity_check(dg_model, scaler, stats_scaler, TEST_COINS, HORIZON, THRESHOLDS, MODEL_PATH, device, DATA_PATH, INCLUDE_DEVIATION=True, SEQ_LEN=60):
    """Sanity check by shuffling time within each sequence."""

    print(f"\n{'='*60}")
    print("Sanity Check: Shuffle time within each sequence")
    print(f"{'='*60}")

    prediction_frames = []

    for coin in TEST_COINS:
        try:
            prepared = prepare_coin_sequences(
                coin, DATA_PATH, HORIZON, include_deviation=INCLUDE_DEVIATION, seq_len=SEQ_LEN
            )
        except FileNotFoundError:
            print(f"  {coin}: feature file not found, skip")
            continue

        if prepared is None:
            print(f"  {coin}: not enough valid samples, skip")
            continue

        X_seq = prepared['X_seq']
        y_seq = prepared['y_seq']
        stats_seq = prepared['stats_seq']
        seq_datetimes = prepared['seq_datetimes']
        n_samples = len(X_seq)

        real_preds = predict_batch(dg_model, X_seq, stats_seq, scaler, stats_scaler, device)
        X_seq_shuffled = shuffle_time_within_sequence(X_seq, seed=42)
        stats_seq_shuffled = stats_seq.copy() if stats_seq is not None else None
        shuffled_preds = predict_batch(dg_model, X_seq_shuffled, stats_seq_shuffled, scaler, stats_scaler, device)

        prediction_frames.append(pd.DataFrame({
            'datetime': seq_datetimes[:n_samples],
            'coin': coin,
            'horizon': HORIZON,
            'y_true': y_seq[:n_samples],
            'pred_real': real_preds,
            'pred_shuffled': shuffled_preds,
        }))

        print(f"\n  {coin} ({n_samples} samples):")
        print(f"    {'Threshold':<10} | {'Real AUC':>10} | {'Shuffled AUC':>13} | {'Diff':>8}")
        print(f"    {'-'*10}-+-{'-'*10}-+-{'-'*13}-+-{'-'*8}")

        for threshold in THRESHOLDS:
            y_binary = (y_seq[:n_samples] > threshold).astype(int)
            real_auc = safe_binary_auc(y_binary, real_preds)
            shuffle_auc = safe_binary_auc(y_binary, shuffled_preds)
            if np.isnan(real_auc) or np.isnan(shuffle_auc):
                continue
            diff = real_auc - shuffle_auc
            print(f"    t={threshold:<8} | {real_auc:>10.4f} | {shuffle_auc:>13.4f} | {diff:>+8.4f}")

    if prediction_frames:
        predictions_df = pd.concat(prediction_frames, ignore_index=True)
        sanity_output = os.path.join(MODEL_PATH, f'dg_final_{HORIZON}_sanity_check.csv')
        predictions_df.to_csv(sanity_output, index=False)
        print(f"\n  Saved sanity-check predictions: {sanity_output}")
        print(f"  Total rows: {len(predictions_df)}")

        real_auc_by_coin = predictions_df.groupby('coin').apply(
            lambda x: safe_binary_auc((x['y_true'] > 0.01).astype(int), x['pred_real'])
        ).dropna()
        shuffle_auc_by_coin = predictions_df.groupby('coin').apply(
            lambda x: safe_binary_auc((x['y_true'] > 0.01).astype(int), x['pred_shuffled'])
        ).dropna()

        if len(real_auc_by_coin) > 0:
            print("\n  Summary")
            print(f"    Real AUC mean: {real_auc_by_coin.mean():.4f}")
            print(f"    Shuffled AUC mean: {shuffle_auc_by_coin.mean():.4f}")

        if len(shuffle_auc_by_coin) > 0:
            avg_shuffle = float(shuffle_auc_by_coin.mean())
            print(f"\n  {'='*50}")
            if 0.45 <= avg_shuffle <= 0.55:
                print(f"  [PASS] Sanity check passed (shuffled AUC = {avg_shuffle:.4f})")
            else:
                print(f"  [FAIL] Sanity check failed (shuffled AUC = {avg_shuffle:.4f})")
            print(f"  {'='*50}")

    print(f"\n{'='*60}")
    print("Sanity Check finished")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description='DG Final')
    parser.add_argument('--horizon', type=str, default='0-3h', choices=['0-3h', '3-6h', '6-9h', '6-12h', '12-18h', '18-24h', '24-30h', '48-54h'])
    parser.add_argument('--num_env', type=int, default=5)
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lambda_vrex', type=float, default=1.0)
    parser.add_argument('--temperature', type=float, default=0.1, help='soft assignment temperature for environment routing')
    parser.add_argument('--gate-floor', type=float, default=0.0, help='minimum routing floor mixed into p_env')
    parser.add_argument('--lambda_entropy', type=float, default=0.1, help='anti-collapse weight')
    parser.add_argument('--lambda-entropy-final', type=float, default=0.01, help='final anti-collapse weight after annealing')
    parser.add_argument('--min-routing-entropy', type=float, default=0.0, help='per-sample minimum routing entropy target')
    parser.add_argument('--lambda-min-routing-entropy', type=float, default=0.0, help='weight for the minimum routing entropy hinge penalty')
    parser.add_argument('--warmup_epochs', type=int, default=3)
    parser.add_argument('--seq-len', type=int, default=120, help='input sequence length')
    parser.add_argument('--refresh-centers-every', type=int, default=3, help='recompute environment centers every N epochs; 0 disables refresh')
    parser.add_argument('--refresh-centers-max', type=int, default=0, help='maximum number of center refreshes; 0 means unlimited')
    parser.add_argument('--thresholds', type=str, default='0,0.01,0.1,0.2',
                       help='逗号分隔的threshold列表，如"0,0.01,0.1,0.2"')
    parser.add_argument('--no-sanity-check', action='store_true', help='跳过 sanity check')
    parser.add_argument(
        '--include-deviation',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='是否包含deviation相关特征',
    )
    parser.add_argument('--data-path', type=str, default='data/ml_predictions/', help='特征数据目录')
    parser.add_argument('--output-dir', type=str, default='data/models/', help='结果输出目录')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda'])
    args = parser.parse_args()
    
    HORIZON = args.horizon
    NUM_ENV = args.num_env
    D_MODEL = args.d_model
    EPOCHS = args.epochs
    LR = args.lr
    LAMBDA_VREX = args.lambda_vrex
    TEMPERATURE = args.temperature
    GATE_FLOOR = args.gate_floor
    LAMBDA_ENTROPY = args.lambda_entropy
    LAMBDA_ENTROPY_FINAL = args.lambda_entropy_final if args.lambda_entropy_final is not None else args.lambda_entropy
    MIN_ROUTING_ENTROPY = args.min_routing_entropy
    LAMBDA_MIN_ROUTING_ENTROPY = args.lambda_min_routing_entropy
    WARMUP_EPOCHS = args.warmup_epochs
    SEQ_LEN = args.seq_len
    REFRESH_CENTERS_EVERY = args.refresh_centers_every
    REFRESH_CENTERS_MAX = args.refresh_centers_max
    THRESHOLDS = [float(t) for t in args.thresholds.split(',')]
    NO_SANITY_CHECK = args.no_sanity_check
    INCLUDE_DEVIATION = args.include_deviation
    DATA_PATH = args.data_path
    MODEL_PATH = args.output_dir
    DEVICE_ARG = args.device
    
    print(f"参数: horizon={HORIZON}, num_env={NUM_ENV}, d_model={D_MODEL}, epochs={EPOCHS}")
    print(f"包含deviation特征: {INCLUDE_DEVIATION}")
    print(
        f"seq_len={SEQ_LEN}, temperature={TEMPERATURE}, gate_floor={GATE_FLOOR}, "
        f"lambda_entropy={LAMBDA_ENTROPY} -> {LAMBDA_ENTROPY_FINAL}, "
        f"min_routing_entropy={MIN_ROUTING_ENTROPY}, lambda_min_routing_entropy={LAMBDA_MIN_ROUTING_ENTROPY}, "
        f"refresh_centers_every={REFRESH_CENTERS_EVERY}, refresh_centers_max={REFRESH_CENTERS_MAX}"
    )
    print(f"Thresholds: {THRESHOLDS}")
    print(f"="*60)
    
    TRAIN_COINS = ['USDC', 'SUSD', 'USDY']
    TEST_COINS = ['DAI', 'EURC', 'TUSD', 'UST', 'XUSD']
    print(f"Train coins: {TRAIN_COINS}")
    print(f"Test coins: {TEST_COINS}")
    
    # Ensure output directories exist to avoid runtime errors when saving models/predictions
    os.makedirs(DATA_PATH, exist_ok=True)
    os.makedirs(MODEL_PATH, exist_ok=True)
    
    # ==================== 数据准备 ====================
    print("准备训练数据...")
    
    all_X, all_y, all_stats = [], [], []
    
    for coin in TRAIN_COINS:
        try:
            prepared = prepare_coin_sequences(
                coin, DATA_PATH, HORIZON, include_deviation=INCLUDE_DEVIATION, seq_len=SEQ_LEN
            )
        except FileNotFoundError:
            print(f"  {coin}: 数据文件不存在，跳过")
            continue

        if prepared is None:
            print(f"  {coin}: 有效样本不足，跳过")
            continue

        X_seq = prepared['X_seq']
        y_seq = prepared['y_seq']
        stats_seq = prepared['stats_seq']
        df = prepared['df']
        all_X.append(X_seq)
        all_y.append(y_seq)
        all_stats.append(stats_seq)
        
        # 获取该coin的时间范围
        coin_start = df['datetime'].min()
        coin_end = df['datetime'].max()
        print(f"  {coin}: {len(X_seq)} 序列, y_mean={y_seq.mean():.4f}, y_max={y_seq.max():.4f}, 时间: {coin_start} ~ {coin_end}")
    
    if not all_X:
        raise RuntimeError("No valid training sequences were built. Check TRAIN_COINS and feature files.")

    X_all = np.concatenate(all_X, axis=0)
    y_all = np.concatenate(all_y)
    
    valid_stats = [s for s in all_stats if s is not None and len(s.shape) > 1 and s.shape[-1] > 0]
    if valid_stats:
        stats_all = np.concatenate(valid_stats, axis=0)
    else:
        stats_all = None
    
    print(f"训练数据: {len(X_all)} 序列, 特征数={X_all.shape[-1]}")
    
    # 标准化
    scaler = StandardScaler()
    X_all_flat = X_all.reshape(-1, X_all.shape[-1])
    scaler.fit(X_all_flat)
    X_all_scaled = scaler.transform(X_all_flat).reshape(X_all.shape)
    
    # Stats 标准化 (启用 hybrid environment)
    if stats_all is not None and len(stats_all.shape) > 1 and stats_all.shape[-1] > 0:
        stats_dim = stats_all.shape[-1]
        stats_scaler = StandardScaler()
        stats_scaler.fit(stats_all)
        stats_all_scaled = stats_scaler.transform(stats_all)
        print(f"Stats 特征数: {stats_dim}")
    else:
        stats_dim = 0
        stats_scaler = None
        stats_all_scaled = None
        print(f"Stats: 无有效数据")
    
    if DEVICE_ARG == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but not available in the selected Python environment.')
        device = torch.device('cuda')
    elif DEVICE_ARG == 'cpu':
        device = torch.device('cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # ==================== 阶段1: ERM 预训练 ====================
    print(f"\n阶段1: ERM预训练 encoder ({WARMUP_EPOCHS} epochs)...")
    
    erm_model = ERMModel(input_dim=X_all.shape[-1], d_model=D_MODEL).to(device)
    erm_optimizer = torch.optim.Adam(erm_model.parameters(), lr=LR)
    
    if stats_all_scaled is not None:
        train_dataset = TensorDataset(
            torch.FloatTensor(X_all_scaled),
            torch.FloatTensor(y_all).unsqueeze(1),
            torch.FloatTensor(stats_all_scaled)
        )
    else:
        train_dataset = TensorDataset(
            torch.FloatTensor(X_all_scaled),
            torch.FloatTensor(y_all).unsqueeze(1)
        )
    
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    
    for epoch in range(WARMUP_EPOCHS):
        erm_model.train()
        total_loss = 0
        for batch in train_loader:
            if len(batch) == 3:
                X_batch, y_batch, stats_batch = batch
                stats_batch = stats_batch.to(device)
            else:
                X_batch, y_batch = batch
                stats_batch = None
            
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred, z = erm_model(X_batch)
            loss = erm_loss(y_pred, y_batch)
            
            erm_optimizer.zero_grad()
            loss.backward()
            erm_optimizer.step()
            total_loss += loss.item()
        
        print(f"  Epoch {epoch+1}: ERM Loss={total_loss/len(train_loader):.6f}")
    
    # ==================== 阶段2: Hybrid KMeans ====================
    print(f"\n阶段2: Hybrid KMeans clustering (num_env={NUM_ENV})...")
    centers = None
    print("Clustering 完成!")
    
    # ==================== 阶段3: DG训练 ====================
    print(f"\n阶段3: DG训练 (epochs={EPOCHS})...")
    
    dg_model = DGModelFinal(
        input_dim=X_all.shape[-1], 
        d_model=D_MODEL, 
        num_env=NUM_ENV,
        temperature=TEMPERATURE,
        stats_dim=stats_dim,
        gate_floor=GATE_FLOOR,
    ).to(device)
    dg_model.encoder.load_state_dict(erm_model.encoder.state_dict())
    centers = refresh_env_centers(dg_model, X_all_scaled, stats_all_scaled, num_env=NUM_ENV, device=device)
    print("Initial environment centers ready.")
    
    optimizer = torch.optim.Adam(dg_model.parameters(), lr=LR)
    refresh_count = 0
    
    for epoch in range(EPOCHS):
        lambda_vrex = LAMBDA_VREX * min(1.0, (epoch + 1) / 3)
        lambda_entropy = linear_anneal(LAMBDA_ENTROPY, LAMBDA_ENTROPY_FINAL, epoch, EPOCHS)
        can_refresh = REFRESH_CENTERS_MAX <= 0 or refresh_count < REFRESH_CENTERS_MAX
        if REFRESH_CENTERS_EVERY > 0 and epoch > 0 and epoch % REFRESH_CENTERS_EVERY == 0 and can_refresh:
            centers = refresh_env_centers(dg_model, X_all_scaled, stats_all_scaled, num_env=NUM_ENV, device=device)
            refresh_count += 1
            print(f"  Refreshed environment centers before epoch {epoch + 1}")
        
        dg_model.train()
        total_loss = 0
        total_erm = 0
        total_vrex = 0
        total_entropy = 0
        total_min_entropy_penalty = 0
        total_routing_entropy = 0
        
        for batch in train_loader:
            if len(batch) == 3:
                X_batch, y_batch, stats_batch = batch
                stats_batch = stats_batch.to(device)
            else:
                X_batch, y_batch = batch
                stats_batch = None
            
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            y_pred, z, p_env, preds = dg_model(X_batch, stats_batch)
            
            loss_erm = erm_loss(y_pred, y_batch)
            loss_vrex = vrex_loss(preds, y_batch, p_env, lambda_vrex=lambda_vrex)
            loss_entropy = entropy_loss(p_env)
            loss_min_entropy = min_routing_entropy_penalty(p_env, MIN_ROUTING_ENTROPY)
            
            loss = loss_erm + loss_vrex + lambda_entropy * loss_entropy + LAMBDA_MIN_ROUTING_ENTROPY * loss_min_entropy
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_erm += loss_erm.item()
            total_vrex += loss_vrex.item()
            total_entropy += loss_entropy.item()
            total_min_entropy_penalty += loss_min_entropy.item()
            total_routing_entropy += routing_entropy(p_env).mean().item()
        
        print(f"Epoch {epoch+1}: Loss={total_loss/len(train_loader):.6f} "
              f"(ERM={total_erm/len(train_loader):.6f}, "
              f"VREx={total_vrex/len(train_loader):.6f}, "
              f"Ent={total_entropy/len(train_loader):.6f}, "
              f"RouteEnt={total_routing_entropy/len(train_loader):.6f}, "
              f"MinEntPenalty={total_min_entropy_penalty/len(train_loader):.6f}, "
              f"EntW={lambda_entropy:.6f}, MinEntW={LAMBDA_MIN_ROUTING_ENTROPY:.6f})")
    
    # 保存模型
    model_name = f'dg_final_{HORIZON}_d{D_MODEL}_k{NUM_ENV}.pt'
    torch.save({
        'model_state_dict': dg_model.state_dict(),
        'erm_encoder_state_dict': erm_model.encoder.state_dict(),
        'env_centers': centers,
        'horizon': HORIZON,
        'num_env': NUM_ENV,
        'd_model': D_MODEL,
        'stats_dim': stats_dim,
        'seq_len': SEQ_LEN,
        'temperature': TEMPERATURE,
        'gate_floor': GATE_FLOOR,
        'lambda_entropy_start': LAMBDA_ENTROPY,
        'lambda_entropy_final': LAMBDA_ENTROPY_FINAL,
        'min_routing_entropy': MIN_ROUTING_ENTROPY,
        'lambda_min_routing_entropy': LAMBDA_MIN_ROUTING_ENTROPY,
        'refresh_centers_every': REFRESH_CENTERS_EVERY,
        'refresh_centers_max': REFRESH_CENTERS_MAX,
    }, os.path.join(MODEL_PATH, model_name))
    
    with open(os.path.join(MODEL_PATH, 'dg_final_scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    if stats_scaler is not None:
        with open(os.path.join(MODEL_PATH, 'dg_final_stats_scaler.pkl'), 'wb') as f:
            pickle.dump(stats_scaler, f)
    
    print(f"\n模型已保存: {os.path.join(MODEL_PATH, model_name)}")
    
    # ==================== 泛化测试 ====================
    print(f"\n{'='*60}")
    print(f"泛化测试 - {HORIZON}")
    print(f"{'='*60}")
    
    gen_results = []
    all_predictions = []  # 保存所有预测结果
    
    for coin in TEST_COINS:
        try:
            prepared = prepare_coin_sequences(
                coin, DATA_PATH, HORIZON, include_deviation=INCLUDE_DEVIATION, seq_len=SEQ_LEN
            )
        except FileNotFoundError:
            print(f"  {coin}: 数据文件不存在，跳过")
            continue

        if prepared is None:
            print(f"  {coin}: 有效样本不足，跳过")
            continue

        X_seq = prepared['X_seq']
        y_seq = prepared['y_seq']
        stats_seq = prepared['stats_seq']
        seq_datetimes = prepared['seq_datetimes']
        preds = predict_batch(dg_model, X_seq, stats_seq, scaler, stats_scaler, device)
        
        n_samples = len(preds)
        pred_df = pd.DataFrame({
            'datetime': seq_datetimes[:n_samples],
            'coin': coin,
            'horizon': HORIZON,
            'prediction': preds,
            'y_soft': y_seq[:n_samples]
        })
        all_predictions.append(pred_df)
        
        # 多 threshold 评估
        coin_results = []
        for THRESHOLD in THRESHOLDS:
            y_binary = (y_seq > THRESHOLD).astype(int)
            pos = int(y_binary.sum())
            neg = len(y_binary) - pos
            
            if pos > 0 and neg > 0:
                auc = roc_auc_score(y_binary, preds)
                ap = average_precision_score(y_binary, preds)
                print(f"  {coin} (t={THRESHOLD}): AUC={auc:.4f}, AP={ap:.4f}, pos={pos}, neg={neg}")
                coin_results.append({
                    'coin': coin,
                    'threshold': THRESHOLD,
                    'auc': auc,
                    'ap': ap,
                    'pos': pos,
                    'neg': neg,
                    'total': len(y_seq)
                })
            elif pos == 0:
                print(f"  {coin} (t={THRESHOLD}): 无正例 (neg={neg})")
            else:
                print(f"  {coin} (t={THRESHOLD}): 无负例 (pos={pos})")
        
        gen_results.extend(coin_results)
    
    # 保存所有预测结果到 CSV
    if all_predictions:
        predictions_df = pd.concat(all_predictions, ignore_index=True)
        predictions_output = os.path.join(MODEL_PATH, f'dg_final_{HORIZON}_predictions.csv')
        predictions_df.to_csv(predictions_output, index=False)
        print(f"\n预测结果已保存: {predictions_output}")
        print(f"  总行数: {len(predictions_df)}")
        print(f"  币种: {predictions_df['coin'].unique().tolist()}")
        print(f"  预测值范围: [{predictions_df['prediction'].min():.6f}, {predictions_df['prediction'].max():.6f}]")
    
    results_df = pd.DataFrame(gen_results)
    results_df.to_csv(os.path.join(MODEL_PATH, f'dg_final_{HORIZON}_thresholds.csv'), index=False)
    
    # 按 threshold 分组打印汇总
    if gen_results:
        print(f"\n按 Threshold 汇总:")
        unique_thresholds = sorted(set(r['threshold'] for r in gen_results))
        for thresh in unique_thresholds:
            subset = [r for r in gen_results if r['threshold'] == thresh]
            if subset:
                avg_auc = np.mean([r['auc'] for r in subset])
                avg_ap = np.mean([r['ap'] for r in subset])
                n_coins = len(subset)
                print(f"  threshold={thresh}: 平均 AUC={avg_auc:.4f}, AP={avg_ap:.4f} ({n_coins} coins)")
    
    # ==================== Sanity Check ====================
    if not NO_SANITY_CHECK:
        run_sanity_check(dg_model, scaler, stats_scaler, TEST_COINS, HORIZON, THRESHOLDS, MODEL_PATH, device, DATA_PATH, INCLUDE_DEVIATION, SEQ_LEN=SEQ_LEN)
    else:
        print("\n[跳过 Sanity Check]")
    
    print(f"\n{'='*60}")
    print("实验完成!")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
