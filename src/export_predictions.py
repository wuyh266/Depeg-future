"""
导出 DG Final 模型对每个币种的预测结果
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
import pickle
import os


def compute_multi_level_labels(df, horizon='0-3h'):
    """标签计算"""
    horizon_map = {
        '0-3h': (1, 180),
        '3-6h': (181, 360),
        '6-12h': (361, 720),
    }
    start, end = horizon_map.get(horizon, (1, 180))
    
    dev = abs(df['close'] - 1.0) * 100
    labels_soft = np.zeros(len(df))
    
    for i in range(len(df) - end - 1):
        future_dev = dev.iloc[i+start:i+end]
        labels_soft[i] = (future_dev >= 5.0).mean()
    
    return labels_soft


def build_features_v3(df, window=10):
    """特征构建"""
    result = pd.DataFrame()
    result['price'] = df['close']
    result['log_return'] = np.log(df['close'] / df['close'].shift(1))
    result['deviation'] = df['close'] - 1.0
    result['rolling_mean'] = df['close'].rolling(window).mean()
    result['rolling_std'] = result['log_return'].rolling(window).std()
    result['rolling_skew'] = result['log_return'].rolling(window*3).skew()
    result['rolling_kurt'] = result['log_return'].rolling(window*3).kurt()
    rolling_max = df['close'].rolling(window*6, min_periods=1).max()
    result['max_drawdown'] = (df['close'] - rolling_max) / rolling_max
    vol_short = result['log_return'].rolling(window).std()
    vol_long = result['log_return'].rolling(window*3).std()
    result['vol_ratio'] = vol_short / (vol_long + 1e-10)
    result['deviation_diff'] = result['deviation'].diff()
    result['volatility_long'] = result['log_return'].rolling(window*6).std()
    result['dev_mean'] = result['deviation'].rolling(window).mean()
    result['dev_max'] = result['deviation'].rolling(window).max()
    result['return_cum'] = result['log_return'].rolling(window).sum()
    
    if 'volume' in df.columns:
        result['volume'] = df['volume']
        result['volume_ratio'] = df['volume'] / df['volume'].rolling(window).mean()
    
    result = result.dropna()
    return result.values, result.columns.tolist()


def create_sequences(X, y, seq_len):
    X_seq, y_seq = [], []
    for i in range(seq_len, len(X)):
        X_seq.append(X[i-seq_len:i])
        y_seq.append(y[i])
    return np.array(X_seq), np.array(y_seq), i - seq_len + 1  # 返回起始索引


class PriceTransformer(nn.Module):
    def __init__(self, input_dim=16, d_model=128, nhead=4, num_layers=2):
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
        return h[:, -1]


class DGModelFinal(nn.Module):
    def __init__(self, input_dim=16, d_model=128, num_env=5, temperature=0.1, stats_dim=1):
        super().__init__()
        self.num_env = num_env
        self.temperature = temperature
        self.stats_dim = stats_dim
        self.encoder = PriceTransformer(input_dim=input_dim, d_model=d_model)
        if stats_dim > 0:
            self.stats_encoder = nn.Linear(stats_dim, d_model // 4)
            self.env_encoder = nn.Linear(d_model + d_model // 4, d_model)
        else:
            self.stats_encoder = None
            self.env_encoder = nn.Linear(d_model, d_model)
        self.env_heads = nn.ModuleList([nn.Linear(d_model, 1) for _ in range(num_env)])
        self.register_buffer("env_centers", torch.randn(num_env, d_model))
    
    def forward(self, x, stats=None):
        z = self.encoder(x)
        if stats is not None and self.stats_encoder is not None:
            stats_emb = F.relu(self.stats_encoder(stats))
            env_input = torch.cat([z, stats_emb], dim=-1)
        else:
            env_input = z
        env_repr = F.relu(self.env_encoder(env_input))
        dist = torch.cdist(env_repr, self.env_centers)
        logits = -dist / self.temperature
        p_env = F.softmax(logits, dim=1)
        preds = torch.stack([head(env_repr) for head in self.env_heads], dim=1)
        y = (p_env.unsqueeze(-1) * preds).sum(dim=1)
        return y, z, p_env, preds


def load_model(horizon, model_path):
    """加载指定 horizon 的模型"""
    model_file = f'{model_path}dg_final_{horizon}_d128_k5.pt'
    if not os.path.exists(model_file):
        print(f"模型文件不存在: {model_file}")
        return None, None
    
    checkpoint = torch.load(model_file, map_location='cpu', weights_only=False)
    stats_dim = checkpoint.get('stats_dim', 1)
    
    model = DGModelFinal(input_dim=16, d_model=128, num_env=5, stats_dim=stats_dim)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model, checkpoint


def predict_coin(coin, horizons, data_path, model_path, device):
    """对单个币种进行所有 horizon 的预测"""
    # 加载数据
    try:
        df = pd.read_csv(f'{data_path}features_{coin}.csv')
    except:
        try:
            df = pd.read_csv(f'{data_path}features_{coin}_bocd.csv')
        except:
            print(f"  数据文件不存在: {coin}")
            return None
    
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime').reset_index(drop=True)
    
    print(f"  {coin}: {len(df)} 行, 时间范围 {df['datetime'].min()} ~ {df['datetime'].max()}")
    
    # 计算所有 horizon 的标签
    labels = {}
    for horizon in horizons:
        labels[horizon] = compute_multi_level_labels(df, horizon)
    
    # 构建特征
    X, _ = build_features_v3(df, window=10)
    
    # 构建 stats 特征 - 只使用第一列以匹配训练时的 stats_dim=1
    _, col_names = build_features_v3(df, window=10)
    stats_cols = ['volatility_long', 'dev_mean', 'dev_max', 'return_cum', 'rolling_std', 'rolling_skew']
    stats_idx = [col_names.index(c) for c in stats_cols if c in col_names]
    if stats_idx:
        # 只使用第一列
        stats = X[:, stats_idx[0:1]]  # 保持 2D 形状
    else:
        stats = None
    
    # 对齐长度
    min_len = len(X)
    for horizon, label in labels.items():
        min_len = min(min_len, len(label))
    
    X = X[:min_len]
    labels = {h: l[:min_len] for h, l in labels.items()}
    if stats is not None:
        stats = stats[:min_len]
    
    # 序列化和预测
    seq_len = 60
    results = {
        'datetime': [],
        'prediction_0_3h': [],
        'prediction_3_6h': [],
        'prediction_6_12h': [],
        'y_soft_0_3h': [],
        'y_soft_3_6h': [],
        'y_soft_6_12h': [],
    }
    
    # 对每个起始位置进行预测
    for i in range(seq_len, len(X)):
        results['datetime'].append(df['datetime'].iloc[i])
        results['y_soft_0_3h'].append(labels['0-3h'][i])
        results['y_soft_3_6h'].append(labels['3-6h'][i])
        results['y_soft_6_12h'].append(labels['6-12h'][i])
    
    # 创建序列
    X_seq, _, _ = create_sequences(X, labels['0-3h'], seq_len)
    if stats is not None:
        stats_seq, _, _ = create_sequences(stats, labels['0-3h'], seq_len)
    else:
        stats_seq = None
    
    # 标准化
    scaler_file = f'{model_path}dg_final_scaler.pkl'
    with open(scaler_file, 'rb') as f:
        scaler = pickle.load(f)
    
    X_seq_flat = X_seq.reshape(-1, X_seq.shape[-1])
    X_seq_scaled = scaler.transform(X_seq_flat).reshape(X_seq.shape)
    
    # 标准化 stats
    stats_scaler_file = f'{model_path}dg_final_stats_scaler.pkl'
    if os.path.exists(stats_scaler_file) and stats_seq is not None:
        with open(stats_scaler_file, 'rb') as f:
            stats_scaler = pickle.load(f)
        stats_seq_flat = stats_seq.reshape(-1, stats_seq.shape[-1])
        stats_seq_scaled = stats_scaler.transform(stats_seq_flat).reshape(stats_seq.shape)
    else:
        stats_seq_scaled = None
    
    # 对每个 horizon 进行预测
    for horizon in horizons:
        model, checkpoint = load_model(horizon, model_path)
        if model is None:
            # 使用默认值填充
            results[f'prediction_{horizon.replace("-", "_")}'] = [np.nan] * len(results['datetime'])
            continue
        model = model.to(device)
        model.eval()
        
        preds = []
        with torch.no_grad():
            for i in range(0, len(X_seq_scaled), 512):
                batch = torch.FloatTensor(X_seq_scaled[i:i+512]).to(device)
                # stats 是序列的最后一个时间步
                if stats_seq_scaled is not None:
                    stats_batch = torch.FloatTensor(stats_seq_scaled[i:i+512, -1, :]).to(device)
                else:
                    stats_batch = None
                pred, _, _, _ = model(batch, stats_batch)
                preds.append(pred.cpu().numpy().flatten())
        preds = np.concatenate(preds)
        
        results[f'prediction_{horizon.replace("-", "_")}'] = preds.tolist()
    
    return pd.DataFrame(results)


def main():
    data_path = '../data/ml_predictions/'
    model_path = '../data/models/'
    output_path = '../data/dg_final_prediction/'
    
    # 确保输出目录存在
    os.makedirs(output_path, exist_ok=True)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 测试币种
    test_coins = ['EURC', 'USDY', 'UST', 'XUSD', 'TUSD']
    horizons = ['0-3h', '3-6h', '6-12h']
    
    print(f"\n开始导出预测结果...")
    print(f"输出目录: {output_path}")
    print(f"Horizons: {horizons}")
    print(f"Coins: {test_coins}")
    print("="*60)
    
    for coin in test_coins:
        print(f"\n处理 {coin}...")
        df_result = predict_coin(coin, horizons, data_path, model_path, device)
        
        if df_result is not None:
            output_file = f'{output_path}predictions_{coin}.csv'
            df_result.to_csv(output_file, index=False)
            print(f"  已保存: {output_file}")
            print(f"  行数: {len(df_result)}")
            print(f"  列: {list(df_result.columns)}")
        else:
            print(f"  跳过 {coin}")
    
    print("\n" + "="*60)
    print("导出完成!")


if __name__ == '__main__':
    main()
