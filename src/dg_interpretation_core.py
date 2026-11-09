import argparse
import os
import pickle
from dataclasses import dataclass
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.manifold import TSNE
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import train_test_split

from experiment_utils import (
    DEFAULT_DATA_PATH,
    HORIZONS,
    STATS_COLS,
    TEST_COINS,
    align_sequence_datetimes,
    build_features_v3,
    compute_multi_level_labels,
    create_sequences,
    extract_stats_features,
    load_features,
)


DEFAULT_OUTPUT_ROOT = os.path.join('data', 'experiment_outputs')
SEQ_LEN = 60
LABEL_THRESHOLD = 0.01

plt.rcParams['font.size'] = 10
plt.rcParams['figure.dpi'] = 150
sns.set_style('whitegrid')


class PriceTransformer(nn.Module):
    def __init__(self, input_dim=17, d_model=128, nhead=4, num_layers=2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=256,
            dropout=0.3,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        x = self.input_proj(x)
        h = self.encoder(x)
        return h[:, -1]


class DGModelFinal(nn.Module):
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

        self.env_heads = nn.ModuleList([nn.Linear(d_model, 1) for _ in range(num_env)])
        self.register_buffer('env_centers', torch.randn(num_env, d_model))

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
        return y, z, p_env, preds, env_repr


@dataclass
class ModelBundle:
    model: DGModelFinal
    scaler: object
    stats_scaler: object | None
    input_dim: int
    stats_dim: int
    d_model: int
    num_env: int
    temperature: float
    gate_floor: float


def make_output_dir(root, prefix):
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    path = os.path.join(root, f'{prefix}_{ts}')
    os.makedirs(path, exist_ok=True)
    return path


def load_model_bundle(model_dir, horizon):
    model_path = os.path.join(model_dir, f'dg_final_{horizon}_d128_k5.pt')
    scaler_path = os.path.join(model_dir, 'dg_final_scaler.pkl')
    stats_scaler_path = os.path.join(model_dir, 'dg_final_stats_scaler.pkl')

    if not os.path.exists(model_path):
        raise FileNotFoundError(f'Model not found: {model_path}')
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f'Scaler not found: {scaler_path}')

    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    input_dim = checkpoint['model_state_dict']['encoder.input_proj.weight'].shape[1]
    stats_dim = int(checkpoint.get('stats_dim', 0))
    d_model = int(checkpoint['d_model'])
    num_env = int(checkpoint['num_env'])
    temperature = float(checkpoint.get('temperature', 0.1))
    gate_floor = float(checkpoint.get('gate_floor', 0.0))

    model = DGModelFinal(
        input_dim=input_dim,
        d_model=d_model,
        num_env=num_env,
        temperature=temperature,
        stats_dim=stats_dim,
        gate_floor=gate_floor,
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    stats_scaler = None
    if stats_dim > 0 and os.path.exists(stats_scaler_path):
        with open(stats_scaler_path, 'rb') as f:
            stats_scaler = pickle.load(f)

    return ModelBundle(
        model=model,
        scaler=scaler,
        stats_scaler=stats_scaler,
        input_dim=input_dim,
        stats_dim=stats_dim,
        d_model=d_model,
        num_env=num_env,
        temperature=temperature,
        gate_floor=gate_floor,
    )


def prepare_coin_analysis_data(coin, data_path, horizon, include_deviation, min_samples=100):
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
    x = features[-min_len:]
    y_soft = y_soft[-min_len:]

    valid_mask = ~np.isnan(y_soft)
    x = x[valid_mask]
    y_soft = y_soft[valid_mask]
    if len(y_soft) < min_samples:
        return None

    real_stats = extract_stats_features(features, col_names, min_len, valid_mask)
    x_seq, y_seq = create_sequences(x, y_soft, SEQ_LEN)

    if real_stats is not None:
        real_stats_seq, _ = create_sequences(real_stats, y_soft, SEQ_LEN)
        if real_stats_seq.ndim == 1:
            real_stats_seq = real_stats_seq.reshape(-1, 1)
    else:
        real_stats_seq = None

    real_stats_feature_names = [name for name in STATS_COLS if name in col_names]
    datetimes = align_sequence_datetimes(valid_datetimes, min_len, valid_mask, SEQ_LEN)

    return {
        'df': df,
        'X_seq': x_seq,
        'y_seq': y_seq,
        'real_stats_seq': real_stats_seq,
        'feature_names': col_names,
        'real_stats_feature_names': real_stats_feature_names,
        'datetimes': datetimes,
    }


def infer_include_deviation(data_path, horizon, input_dim, stats_dim):
    matched = []
    for include_deviation in [False, True]:
        for coin in TEST_COINS:
            prepared = prepare_coin_analysis_data(coin, data_path, horizon, include_deviation)
            if prepared is None:
                continue
            sample_input_dim = int(prepared['X_seq'].shape[-1])
            real_stats_dim = 0 if prepared['real_stats_seq'] is None else int(prepared['real_stats_seq'].shape[-1])
            model_compatible = sample_input_dim == input_dim and (
                real_stats_dim == stats_dim or stats_dim in [0, 1]
            )
            if model_compatible:
                matched.append(include_deviation)
                break

    if len(matched) == 1:
        return matched[0]

    if len(matched) > 1:
        return matched[0]

    raise RuntimeError(
        f'Could not infer include_deviation for input_dim={input_dim}, stats_dim={stats_dim}, horizon={horizon}'
    )


def collect_analysis_data(horizon, data_path, include_deviation, expected_model_stats_dim, max_per_coin=None):
    all_x, all_y, all_real_stats, all_model_stats, all_coins, all_datetimes = [], [], [], [], [], []
    feature_names = None
    real_stats_feature_names = None
    stats_mode = None

    for coin in TEST_COINS:
        prepared = prepare_coin_analysis_data(coin, data_path, horizon, include_deviation)
        if prepared is None:
            continue

        x_seq = prepared['X_seq']
        y_seq = prepared['y_seq']
        real_stats_seq = prepared['real_stats_seq']
        datetimes = np.asarray(prepared['datetimes'])

        real_stats_dim = 0 if real_stats_seq is None else int(real_stats_seq.shape[-1])
        if expected_model_stats_dim == 0:
            model_stats_seq = None
            current_stats_mode = 'none'
            model_stats_feature_names = []
        elif real_stats_dim == expected_model_stats_dim:
            model_stats_seq = real_stats_seq
            current_stats_mode = 'real_stats'
            model_stats_feature_names = prepared['real_stats_feature_names']
        elif expected_model_stats_dim == 1:
            model_stats_seq = y_seq.reshape(-1, 1)
            current_stats_mode = 'leaked_y_soft'
            model_stats_feature_names = ['leaked_future_soft_label']
        else:
            raise RuntimeError(
                f'{horizon}/{coin}: real_stats_dim={real_stats_dim} does not match model_stats_dim={expected_model_stats_dim}'
            )

        if max_per_coin is not None and len(x_seq) > max_per_coin:
            idx = np.linspace(0, len(x_seq) - 1, num=max_per_coin, dtype=int)
            x_seq = x_seq[idx]
            y_seq = y_seq[idx]
            datetimes = datetimes[idx]
            if real_stats_seq is not None:
                real_stats_seq = real_stats_seq[idx]
            if model_stats_seq is not None:
                model_stats_seq = model_stats_seq[idx]

        all_x.append(x_seq)
        all_y.append(y_seq)
        all_datetimes.append(datetimes)
        all_coins.extend([coin] * len(x_seq))
        feature_names = prepared['feature_names']
        real_stats_feature_names = prepared['real_stats_feature_names']
        stats_mode = current_stats_mode
        if real_stats_seq is not None:
            all_real_stats.append(real_stats_seq)
        if model_stats_seq is not None:
            all_model_stats.append(model_stats_seq)

    if not all_x:
        raise RuntimeError(f'No usable test sequences found for horizon={horizon} under {data_path}')

    return {
        'X_seq': np.concatenate(all_x, axis=0),
        'y_seq': np.concatenate(all_y, axis=0),
        'real_stats_seq': np.concatenate(all_real_stats, axis=0) if all_real_stats else None,
        'model_stats_seq': np.concatenate(all_model_stats, axis=0) if all_model_stats else None,
        'coins': np.asarray(all_coins),
        'datetimes': np.concatenate(all_datetimes, axis=0),
        'feature_names': feature_names,
        'real_stats_feature_names': real_stats_feature_names or [],
        'model_stats_feature_names': model_stats_feature_names,
        'stats_mode': stats_mode,
        'leakage_detected': stats_mode == 'leaked_y_soft',
    }


def scale_inputs(bundle, x_seq, stats_seq):
    x_scaled = bundle.scaler.transform(x_seq.reshape(-1, x_seq.shape[-1])).reshape(x_seq.shape)
    stats_scaled = None
    if stats_seq is not None:
        if bundle.stats_scaler is None:
            raise RuntimeError('stats_seq exists but stats_scaler is missing')
        stats_scaled = bundle.stats_scaler.transform(stats_seq)
    return x_scaled, stats_scaled


def predict_outputs(bundle, x_seq, stats_seq=None, batch_size=512, device='cpu'):
    x_scaled, stats_scaled = scale_inputs(bundle, x_seq, stats_seq)
    model = bundle.model.to(device)
    model.eval()

    outputs = {'pred': [], 'z': [], 'p_env': [], 'env_repr': [], 'env_head_pred': []}
    with torch.no_grad():
        for start in range(0, len(x_scaled), batch_size):
            end = start + batch_size
            x_batch = torch.tensor(x_scaled[start:end], dtype=torch.float32, device=device)
            stats_batch = None
            if stats_scaled is not None:
                stats_batch = torch.tensor(stats_scaled[start:end], dtype=torch.float32, device=device)

            pred, z, p_env, env_head_pred, env_repr = model(x_batch, stats_batch)
            outputs['pred'].append(pred.squeeze(-1).cpu().numpy())
            outputs['z'].append(z.cpu().numpy())
            outputs['p_env'].append(p_env.cpu().numpy())
            outputs['env_repr'].append(env_repr.cpu().numpy())
            outputs['env_head_pred'].append(env_head_pred.squeeze(-1).cpu().numpy())

    return {key: np.concatenate(value, axis=0) for key, value in outputs.items()}


def macro_auc_by_coin(y_soft, preds, coins, threshold=LABEL_THRESHOLD):
    rows = []
    aucs = []
    for coin in sorted(np.unique(coins)):
        mask = coins == coin
        y_bin = (y_soft[mask] > threshold).astype(int)
        pos = int(y_bin.sum())
        neg = int(len(y_bin) - pos)
        if pos == 0 or neg == 0:
            auc = np.nan
        else:
            auc = float(roc_auc_score(y_bin, preds[mask]))
            aucs.append(auc)
        rows.append({'coin': coin, 'auc01': auc, 'pos': pos, 'neg': neg, 'n': int(mask.sum())})

    macro_auc = float(np.mean(aucs)) if aucs else np.nan
    return macro_auc, pd.DataFrame(rows)


def compute_gradient_importance(bundle, x_seq, stats_seq, sample_size, seed, device='cpu'):
    rng = np.random.default_rng(seed)
    if len(x_seq) > sample_size:
        idx = rng.choice(len(x_seq), size=sample_size, replace=False)
    else:
        idx = np.arange(len(x_seq))

    x_sample = x_seq[idx]
    stats_sample = stats_seq[idx] if stats_seq is not None else None
    x_scaled, stats_scaled = scale_inputs(bundle, x_sample, stats_sample)

    model = bundle.model.to(device)
    model.eval()

    x_tensor = torch.tensor(x_scaled, dtype=torch.float32, requires_grad=True, device=device)
    stats_tensor = None
    if stats_scaled is not None:
        stats_tensor = torch.tensor(stats_scaled, dtype=torch.float32, requires_grad=True, device=device)

    pred, _, _, _, _ = model(x_tensor, stats_tensor)
    pred.sum().backward()

    seq_importance = (x_tensor.grad.abs() * x_tensor.abs()).detach().cpu().numpy()
    per_feature = seq_importance.mean(axis=(0, 1))
    per_timestep = seq_importance.mean(axis=(0, 2))
    per_feature = per_feature / (per_feature.sum() + 1e-12)
    per_timestep = per_timestep / (per_timestep.sum() + 1e-12)

    stats_feature = None
    if stats_tensor is not None and stats_tensor.grad is not None:
        stats_feature = (stats_tensor.grad.abs() * stats_tensor.abs()).detach().cpu().numpy().mean(axis=0)
        stats_feature = stats_feature / (stats_feature.sum() + 1e-12)

    return per_feature, per_timestep, stats_feature


def stratified_coin_subsample(coins, max_per_coin, seed):
    rng = np.random.default_rng(seed)
    idx_list = []
    for coin in sorted(np.unique(coins)):
        coin_idx = np.where(coins == coin)[0]
        if len(coin_idx) > max_per_coin:
            chosen = rng.choice(coin_idx, size=max_per_coin, replace=False)
            idx_list.append(np.sort(chosen))
        else:
            idx_list.append(coin_idx)
    return np.concatenate(idx_list)


def permute_within_coin(values, coins, rng):
    permuted = values.copy()
    for coin in np.unique(coins):
        mask = coins == coin
        order = rng.permutation(np.where(mask)[0])
        permuted[mask] = values[order]
    return permuted


def compute_permutation_importance(
    bundle,
    x_seq,
    y_seq,
    coins,
    stats_seq,
    stats_feature_names,
    feature_names,
    repeats,
    max_per_coin,
    seed,
    device='cpu',
):
    idx = stratified_coin_subsample(coins, max_per_coin=max_per_coin, seed=seed)
    x_sub = x_seq[idx]
    y_sub = y_seq[idx]
    coin_sub = coins[idx]
    stats_sub = stats_seq[idx] if stats_seq is not None else None

    baseline_preds = predict_outputs(bundle, x_sub, stats_sub, device=device)['pred']
    baseline_auc, _ = macro_auc_by_coin(y_sub, baseline_preds, coin_sub)

    rows = []
    rng = np.random.default_rng(seed)

    for feature_idx, feature_name in enumerate(feature_names):
        deltas = []
        for repeat in range(repeats):
            x_perm = x_sub.copy()
            x_perm[:, :, feature_idx] = permute_within_coin(x_sub[:, :, feature_idx], coin_sub, rng)
            perm_preds = predict_outputs(bundle, x_perm, stats_sub, device=device)['pred']
            perm_auc, _ = macro_auc_by_coin(y_sub, perm_preds, coin_sub)
            deltas.append(baseline_auc - perm_auc)
        rows.append({
            'feature': feature_name,
            'source': 'sequence',
            'baseline_macro_auc01': baseline_auc,
            'mean_auc_drop': float(np.mean(deltas)),
            'std_auc_drop': float(np.std(deltas)),
        })

    if stats_sub is not None:
        for stats_idx in range(stats_sub.shape[1]):
            deltas = []
            stats_name = stats_feature_names[stats_idx] if stats_idx < len(stats_feature_names) else f'stats_{stats_idx}'
            for repeat in range(repeats):
                perm_stats = stats_sub.copy()
                perm_stats[:, stats_idx] = permute_within_coin(stats_sub[:, stats_idx], coin_sub, rng)
                perm_preds = predict_outputs(bundle, x_sub, perm_stats, device=device)['pred']
                perm_auc, _ = macro_auc_by_coin(y_sub, perm_preds, coin_sub)
                deltas.append(baseline_auc - perm_auc)
            rows.append({
                'feature': stats_name,
                'source': 'stats',
                'baseline_macro_auc01': baseline_auc,
                'mean_auc_drop': float(np.mean(deltas)),
                'std_auc_drop': float(np.std(deltas)),
            })

    return pd.DataFrame(rows).sort_values(['mean_auc_drop', 'source'], ascending=[False, True])


def safe_corr(x, y):
    x_std = float(np.std(x))
    y_std = float(np.std(y))
    if x_std == 0.0 or y_std == 0.0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def latent_dimension_summary(z, preds, y_seq):
    rows = []
    for dim in range(z.shape[1]):
        values = z[:, dim]
        corr_pred = safe_corr(values, preds)
        corr_y = safe_corr(values, y_seq)
        rows.append({
            'dim': dim,
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'corr_pred': corr_pred,
            'corr_y_soft': corr_y,
            'abs_corr_pred': float(abs(corr_pred)) if not np.isnan(corr_pred) else np.nan,
            'abs_corr_y_soft': float(abs(corr_y)) if not np.isnan(corr_y) else np.nan,
        })
    return pd.DataFrame(rows).sort_values('abs_corr_y_soft', ascending=False)


def compute_linear_probes(z, x_last, y_seq, feature_names, seed):
    z_train, z_test, x_train, x_test, y_train, y_test = train_test_split(
        z, x_last, y_seq, test_size=0.2, random_state=seed
    )

    y_probe = Ridge(alpha=1.0)
    y_probe.fit(z_train, y_train)
    y_pred = y_probe.predict(z_test)
    y_metrics = pd.DataFrame([{
        'target': 'y_soft',
        'test_r2': float(r2_score(y_test, y_pred)),
        'test_corr': safe_corr(y_test, y_pred),
    }])

    x_probe = Ridge(alpha=1.0)
    x_probe.fit(z_train, x_train)
    x_pred = x_probe.predict(z_test)

    feature_rows = []
    coef = x_probe.coef_
    for idx, feature_name in enumerate(feature_names):
        feature_rows.append({
            'feature': feature_name,
            'test_r2': float(r2_score(x_test[:, idx], x_pred[:, idx])),
            'coef_l2': float(np.linalg.norm(coef[idx])),
        })

    coef_rows = []
    for feature_idx, feature_name in enumerate(feature_names):
        feature_coef = coef[feature_idx]
        top_dims = np.argsort(np.abs(feature_coef))[-10:][::-1]
        for rank, dim_idx in enumerate(top_dims, start=1):
            coef_rows.append({
                'feature': feature_name,
                'rank': rank,
                'latent_dim': int(dim_idx),
                'coef': float(feature_coef[dim_idx]),
                'abs_coef': float(abs(feature_coef[dim_idx])),
            })

    return (
        y_metrics,
        pd.DataFrame(feature_rows).sort_values('test_r2', ascending=False),
        pd.DataFrame(coef_rows).sort_values(['feature', 'rank']),
    )


def save_embedding_plot(z, p_env, coins, y_seq, output_path, seed):
    sample_n = min(len(z), 4000)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(z), size=sample_n, replace=False) if len(z) > sample_n else np.arange(len(z))

    z_sample = z[idx]
    env_sample = np.argmax(p_env[idx], axis=1)
    coin_sample = coins[idx]
    y_bin = (y_seq[idx] > LABEL_THRESHOLD).astype(int)

    pca = PCA(n_components=2, random_state=seed)
    z_pca = pca.fit_transform(z_sample)
    tsne = TSNE(n_components=2, perplexity=30, random_state=seed, max_iter=1000)
    z_tsne = tsne.fit_transform(z_sample)

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    axes = axes.flatten()

    sns.scatterplot(x=z_pca[:, 0], y=z_pca[:, 1], hue=coin_sample, s=10, alpha=0.55, ax=axes[0], legend=True)
    axes[0].set_title('PCA colored by coin')

    sns.scatterplot(x=z_pca[:, 0], y=z_pca[:, 1], hue=env_sample, palette='tab10', s=10, alpha=0.55, ax=axes[1], legend=True)
    axes[1].set_title('PCA colored by dominant environment')

    sns.scatterplot(x=z_tsne[:, 0], y=z_tsne[:, 1], hue=coin_sample, s=10, alpha=0.55, ax=axes[2], legend=False)
    axes[2].set_title('t-SNE colored by coin')

    sns.scatterplot(x=z_tsne[:, 0], y=z_tsne[:, 1], hue=y_bin, palette='Set1', s=10, alpha=0.55, ax=axes[3], legend=True)
    axes[3].set_title('t-SNE colored by y_soft > 0.01')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    explained = pca.explained_variance_ratio_
    return pd.DataFrame({
        'pc': [1, 2],
        'explained_variance_ratio': explained,
        'cumulative_explained_variance_ratio': np.cumsum(explained),
    })


def build_environment_summary(preds, p_env, y_seq, coins):
    env_assign = np.argmax(p_env, axis=1)
    env_rows = []
    coin_env_rows = []

    for env_id in range(p_env.shape[1]):
        mask = env_assign == env_id
        if not np.any(mask):
            continue
        env_rows.append({
            'env_id': env_id,
            'count': int(mask.sum()),
            'ratio': float(mask.mean()),
            'avg_pred': float(preds[mask].mean()),
            'avg_y_soft': float(y_seq[mask].mean()),
            'positive_rate_at_001': float((y_seq[mask] > LABEL_THRESHOLD).mean()),
        })

        for coin in sorted(np.unique(coins)):
            coin_mask = mask & (coins == coin)
            if not np.any(coin_mask):
                continue
            coin_env_rows.append({
                'env_id': env_id,
                'coin': coin,
                'count': int(coin_mask.sum()),
                'within_env_ratio': float(coin_mask.sum() / mask.sum()),
                'avg_pred': float(preds[coin_mask].mean()),
                'avg_y_soft': float(y_seq[coin_mask].mean()),
            })

    return pd.DataFrame(env_rows), pd.DataFrame(coin_env_rows)


def save_prediction_frame(output_dir, horizon, data, outputs):
    frame = pd.DataFrame({
        'datetime': data['datetimes'],
        'coin': data['coins'],
        'horizon': horizon,
        'prediction': outputs['pred'],
        'y_soft': data['y_seq'],
        'dominant_env': np.argmax(outputs['p_env'], axis=1),
    })
    frame.to_csv(os.path.join(output_dir, f'{horizon}_prediction_latent_frame.csv'), index=False)


def analyze_single_horizon(
    horizon,
    model_dir,
    data_path,
    output_dir,
    sample_size,
    permutation_repeats,
    max_perm_per_coin,
    seed,
    device,
):
    bundle = load_model_bundle(model_dir, horizon)
    include_deviation = infer_include_deviation(data_path, horizon, bundle.input_dim, bundle.stats_dim)
    data = collect_analysis_data(
        horizon=horizon,
        data_path=data_path,
        include_deviation=include_deviation,
        expected_model_stats_dim=bundle.stats_dim,
    )

    if data['X_seq'].shape[-1] != bundle.input_dim:
        raise RuntimeError(f'{horizon}: feature dim mismatch {data["X_seq"].shape[-1]} != {bundle.input_dim}')

    data_stats_dim = 0 if data['model_stats_seq'] is None else data['model_stats_seq'].shape[-1]
    if data_stats_dim != bundle.stats_dim:
        raise RuntimeError(f'{horizon}: stats dim mismatch {data_stats_dim} != {bundle.stats_dim}')

    outputs = predict_outputs(bundle, data['X_seq'], data['model_stats_seq'], device=device)
    macro_auc, coin_auc = macro_auc_by_coin(data['y_seq'], outputs['pred'], data['coins'])

    per_feature, per_timestep, stats_feature = compute_gradient_importance(
        bundle=bundle,
        x_seq=data['X_seq'],
        stats_seq=data['model_stats_seq'],
        sample_size=sample_size,
        seed=seed,
        device=device,
    )
    permutation_df = compute_permutation_importance(
        bundle=bundle,
        x_seq=data['X_seq'],
        y_seq=data['y_seq'],
        coins=data['coins'],
        stats_seq=data['model_stats_seq'],
        stats_feature_names=data['model_stats_feature_names'],
        feature_names=data['feature_names'],
        repeats=permutation_repeats,
        max_per_coin=max_perm_per_coin,
        seed=seed,
        device=device,
    )
    latent_df = latent_dimension_summary(outputs['z'], outputs['pred'], data['y_seq'])
    y_probe_df, feature_probe_df, probe_coef_df = compute_linear_probes(
        z=outputs['z'],
        x_last=data['X_seq'][:, -1, :],
        y_seq=data['y_seq'],
        feature_names=data['feature_names'],
        seed=seed,
    )
    env_df, coin_env_df = build_environment_summary(outputs['pred'], outputs['p_env'], data['y_seq'], data['coins'])
    pca_df = save_embedding_plot(
        z=outputs['z'],
        p_env=outputs['p_env'],
        coins=data['coins'],
        y_seq=data['y_seq'],
        output_path=os.path.join(output_dir, f'{horizon}_latent_embedding.png'),
        seed=seed,
    )

    gradient_df = pd.DataFrame({
        'feature': data['feature_names'],
        'gradient_x_input_importance': per_feature,
    }).sort_values('gradient_x_input_importance', ascending=False)
    timestep_df = pd.DataFrame({
        'timestep_from_oldest': np.arange(1, len(per_timestep) + 1),
        'gradient_x_input_importance': per_timestep,
    }).sort_values('timestep_from_oldest')

    gradient_df.to_csv(os.path.join(output_dir, f'{horizon}_gradient_feature_importance.csv'), index=False)
    timestep_df.to_csv(os.path.join(output_dir, f'{horizon}_temporal_importance.csv'), index=False)
    permutation_df.to_csv(os.path.join(output_dir, f'{horizon}_permutation_importance_auc01.csv'), index=False)
    latent_df.to_csv(os.path.join(output_dir, f'{horizon}_latent_dim_summary.csv'), index=False)
    y_probe_df.to_csv(os.path.join(output_dir, f'{horizon}_latent_probe_y.csv'), index=False)
    feature_probe_df.to_csv(os.path.join(output_dir, f'{horizon}_latent_probe_features.csv'), index=False)
    probe_coef_df.to_csv(os.path.join(output_dir, f'{horizon}_latent_probe_top_dims.csv'), index=False)
    env_df.to_csv(os.path.join(output_dir, f'{horizon}_environment_summary.csv'), index=False)
    coin_env_df.to_csv(os.path.join(output_dir, f'{horizon}_coin_environment_mix.csv'), index=False)
    coin_auc.to_csv(os.path.join(output_dir, f'{horizon}_coin_auc01.csv'), index=False)
    pca_df.to_csv(os.path.join(output_dir, f'{horizon}_pca_explained_variance.csv'), index=False)
    save_prediction_frame(output_dir, horizon, data, outputs)

    if stats_feature is not None:
        pd.DataFrame({
            'stats_feature': data['model_stats_feature_names'],
            'gradient_x_input_importance': stats_feature,
        }).to_csv(os.path.join(output_dir, f'{horizon}_stats_importance.csv'), index=False)

    pd.DataFrame([{
        'horizon': horizon,
        'include_deviation': include_deviation,
        'input_dim': bundle.input_dim,
        'model_stats_dim': bundle.stats_dim,
        'real_stats_dim': 0 if data['real_stats_seq'] is None else data['real_stats_seq'].shape[-1],
        'stats_mode': data['stats_mode'],
        'leakage_detected': data['leakage_detected'],
    }]).to_csv(os.path.join(output_dir, f'{horizon}_analysis_metadata.csv'), index=False)

    return {
        'horizon': horizon,
        'include_deviation': include_deviation,
        'input_dim': bundle.input_dim,
        'stats_dim': bundle.stats_dim,
        'real_stats_dim': 0 if data['real_stats_seq'] is None else int(data['real_stats_seq'].shape[-1]),
        'stats_mode': data['stats_mode'],
        'leakage_detected': data['leakage_detected'],
        'macro_auc01': macro_auc,
        'n_samples': int(len(data['X_seq'])),
        'n_coins': int(len(np.unique(data['coins']))),
        'top_gradient_feature': gradient_df.iloc[0]['feature'],
        'top_gradient_importance': float(gradient_df.iloc[0]['gradient_x_input_importance']),
        'top_permutation_feature': permutation_df.iloc[0]['feature'],
        'top_permutation_auc_drop': float(permutation_df.iloc[0]['mean_auc_drop']),
        'top_latent_dim': int(latent_df.iloc[0]['dim']),
        'top_latent_abs_corr_y': float(latent_df.iloc[0]['abs_corr_y_soft']),
        'y_probe_r2': float(y_probe_df.iloc[0]['test_r2']),
    }


def save_markdown_summary(summary_df, output_dir, model_dir, data_path):
    header = '| horizon | include_deviation | input_dim | stats_dim | real_stats_dim | stats_mode | leakage_detected | macro_auc01 | n_samples | n_coins | top_gradient_feature | top_gradient_importance | top_permutation_feature | top_permutation_auc_drop | top_latent_dim | top_latent_abs_corr_y | y_probe_r2 |'
    sep = '|---|---:|---:|---:|---:|---|---|---:|---:|---:|---|---:|---|---:|---:|---:|---:|'
    rows = [
        '| ' + ' | '.join(str(row[col]) for col in summary_df.columns) + ' |'
        for _, row in summary_df.iterrows()
    ]
    lines = [
        '# DG Final Interpretation Summary',
        '',
        f'- Model dir: `{model_dir}`',
        f'- Data path: `{data_path}`',
        f'- Label threshold for AUC / risk analysis: `{LABEL_THRESHOLD}`',
        '',
        '## Horizon Summary',
        '',
        header,
        sep,
        *rows,
        '',
        '## Usage Boundary',
        '',
        '- The outputs in this folder are checkpoint-consistent: they load the real DG model, the real sequence scaler, and the real stats scaler expected by the checkpoint.',
        '- Sequence feature importance is reported with two methods: gradient x input sensitivity and permutation-induced macro AUC@0.01 drop.',
        '- Latent analysis uses the real 128-dimensional `z` extracted from the trained checkpoint, plus ridge probes on held-out samples.',
        '- If `stats_mode=leaked_y_soft`, the checkpoint expects a leaked future label instead of real stats features. In that case the interpretation is diagnostic evidence of label leakage, not valid model evidence.',
        '- Older RF scripts are not included in this report because their label / split protocol is not aligned with the formal DG Final evaluation setup.',
    ]
    with open(os.path.join(output_dir, 'README.md'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


def parse_args():
    parser = argparse.ArgumentParser(description='Checkpoint-consistent DG Final interpretation analysis')
    parser.add_argument('--model-dir', required=True, help='Directory containing dg_final_*.pt and scaler files')
    parser.add_argument('--data-path', default=DEFAULT_DATA_PATH, help='Feature CSV directory')
    parser.add_argument('--output-dir', default=None, help='Output directory; default creates timestamped folder')
    parser.add_argument('--horizons', default=','.join(HORIZONS), help='Comma-separated horizons')
    parser.add_argument('--sample-size', type=int, default=4096, help='Sample size for gradient importance')
    parser.add_argument('--permutation-repeats', type=int, default=3, help='Repeats per feature for permutation importance')
    parser.add_argument('--max-perm-per-coin', type=int, default=2000, help='Max samples per coin for permutation importance')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', default='cpu', help='Device: cpu or cuda')
    return parser.parse_args()


def main():
    args = parse_args()
    horizons = [h.strip() for h in args.horizons.split(',') if h.strip()]
    output_dir = args.output_dir or make_output_dir(DEFAULT_OUTPUT_ROOT, 'dg_final_interpretation')
    os.makedirs(output_dir, exist_ok=True)

    summary_rows = []
    for horizon in horizons:
        print(f'Analyzing {horizon} ...')
        row = analyze_single_horizon(
            horizon=horizon,
            model_dir=args.model_dir,
            data_path=args.data_path,
            output_dir=output_dir,
            sample_size=args.sample_size,
            permutation_repeats=args.permutation_repeats,
            max_perm_per_coin=args.max_perm_per_coin,
            seed=args.seed,
            device=args.device,
        )
        summary_rows.append(row)
        print(
            f"  macro_auc01={row['macro_auc01']:.4f}, "
            f"top_gradient={row['top_gradient_feature']}, "
            f"top_perm={row['top_permutation_feature']}"
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_df['horizon'] = pd.Categorical(summary_df['horizon'], categories=HORIZONS, ordered=True)
    summary_df = summary_df.sort_values('horizon')
    summary_df['horizon'] = summary_df['horizon'].astype(str)
    summary_df.to_csv(os.path.join(output_dir, 'summary_by_horizon.csv'), index=False)
    save_markdown_summary(summary_df, output_dir, args.model_dir, args.data_path)
    print(f'Interpretation outputs saved to {output_dir}')


if __name__ == '__main__':
    main()
