import numpy as np
import pandas as pd
import os
from scipy.stats import skew, kurtosis
from bocd_extended import ExtendedBOCD
from depeg_label import DepegLabeler
try:
    from tools import save_csv_with_backup
except ImportError:
    from src.tools import save_csv_with_backup


class FeatureEngineer:
    def __init__(self, windows=[6, 12, 24, 48, 72], horizons=[6, 24, 168]):
        self.windows = windows
        self.horizons = horizons
        self.bocd_cache = {}
        self.metrics_df = None
        self.pool_metrics_df = None

    def load_metrics(self, coin_name, data_dir='data'):
        """加载指定币种的池指标数据"""
        metrics_file = os.path.join(data_dir, coin_name.lower(), 'pool_metrics_full.csv')
        if os.path.exists(metrics_file):
            self.metrics_df = pd.read_csv(metrics_file)
            print(f"Loaded metrics data for {coin_name}: {len(self.metrics_df)} rows")
        else:
            print(f"Warning: Metrics file not found at {metrics_file}")
            self.metrics_df = None
        return self.metrics_df

    def load_curve_data(self, coin_name, data_dir='data'):
        """加载Curve数据"""
        curve_file = os.path.join(data_dir, coin_name.lower(), 'curve_metrics.csv')
        if os.path.exists(curve_file):
            curve_df = pd.read_csv(curve_file)
            print(f"Loaded Curve data for {coin_name}: {len(curve_df)} rows")
            return curve_df
        return None

    def load_token_hour_data(self, coin_name, data_dir='data'):
        """加载TokenHourData"""
        token_hour_file = os.path.join(data_dir, coin_name.lower(), 'token_hour_data.csv')
        if os.path.exists(token_hour_file):
            df = pd.read_csv(token_hour_file)
            print(f"Loaded TokenHourData for {coin_name}: {len(df)} rows")
            return df
        return None

    def load_positions(self, coin_name, data_dir='data'):
        """加载Positions数据"""
        positions_file = os.path.join(data_dir, coin_name.lower(), 'positions.csv')
        if os.path.exists(positions_file):
            df = pd.read_csv(positions_file)
            print(f"Loaded Positions for {coin_name}: {len(df)} rows")
            return df
        return None

    def load_mint_burn_events(self, coin_name, data_dir='data'):
        """加载Mint/Burn事件数据"""
        mints_file = os.path.join(data_dir, coin_name.lower(), 'mints.csv')
        burns_file = os.path.join(data_dir, coin_name.lower(), 'burns.csv')
        
        mints_df = pd.read_csv(mints_file) if os.path.exists(mints_file) else pd.DataFrame()
        burns_df = pd.read_csv(burns_file) if os.path.exists(burns_file) else pd.DataFrame()
        
        if len(mints_df) > 0 or len(burns_df) > 0:
            print(f"Loaded Mint/Burn events for {coin_name}: {len(mints_df)} mints, {len(burns_df)} burns")
        
        return mints_df, burns_df

    def extract_curve_features(self, curve_df, current_idx, anchor=1.0):
        """从Curve数据提取特征"""
        features = {}
        
        if curve_df is None or len(curve_df) == 0:
            return features
        
        if 'hour' not in curve_df.columns:
            return features
        
        if current_idx >= len(curve_df):
            current_idx = len(curve_df) - 1
        
        current_hour = curve_df.iloc[current_idx]['hour']
        
        for window in [1, 6, 24]:
            window_data = curve_df[curve_df['hour'] >= current_hour - window * 3600]
            
            if len(window_data) == 0:
                continue
            
            if 'token0Price' in window_data.columns:
                vp = window_data['token0Price'].dropna()
                if len(vp) > 0:
                    features[f'curve_vp_deviation_{window}h'] = abs(vp.iloc[-1] - anchor)
                    features[f'curve_vp_change_{window}h'] = (vp.iloc[-1] - vp.iloc[0]) / (vp.iloc[0] + 1e-10) if len(vp) > 1 else 0
            
            if 'liquidity' in window_data.columns:
                liq = window_data['liquidity'].dropna()
                if len(liq) > 0 and liq.iloc[0] > 0:
                    features[f'curve_liquidity_change_{window}h'] = (liq.iloc[-1] - liq.iloc[0]) / liq.iloc[0]
            
            if 'volumeUSD' in window_data.columns:
                vol = window_data['volumeUSD'].dropna()
                if len(vol) > 0:
                    features[f'curve_volume_{window}h'] = vol.sum()
        
        return features

    def extract_token_hour_features(self, token_hour_df, current_idx):
        """从TokenHourData提取特征"""
        features = {}
        
        if token_hour_df is None or len(token_hour_df) == 0:
            return features
        
        if 'periodStartUnix' not in token_hour_df.columns:
            return features
        
        if current_idx >= len(token_hour_df):
            current_idx = len(token_hour_df) - 1
        
        current_hour = token_hour_df.iloc[current_idx]['periodStartUnix']
        
        for window in [6, 24]:
            window_data = token_hour_df[token_hour_df['periodStartUnix'] >= current_hour - window * 3600]
            
            if len(window_data) == 0:
                continue
            
            if 'totalValueLockedUSD' in window_data.columns:
                tvl = window_data['totalValueLockedUSD'].dropna()
                if len(tvl) > 0:
                    features[f'token_tvl_{window}h'] = tvl.iloc[-1]
                    if len(tvl) > 1 and tvl.iloc[0] > 0:
                        features[f'token_tvl_change_{window}h'] = (tvl.iloc[-1] - tvl.iloc[0]) / tvl.iloc[0]
            
            if 'volumeUSD' in window_data.columns:
                vol = window_data['volumeUSD'].dropna()
                if len(vol) > 0:
                    features[f'token_volume_{window}h'] = vol.sum()
            
            if 'feesUSD' in window_data.columns:
                fees = window_data['feesUSD'].dropna()
                if len(fees) > 0:
                    features[f'token_fees_{window}h'] = fees.sum()
        
        return features

    def extract_position_features(self, positions_df, mints_df, burns_df, current_ts):
        """从Positions和Mint/Burn事件提取特征"""
        features = {}
        
        if positions_df is not None and len(positions_df) > 0:
            if 'timestamp' in positions_df.columns:
                positions_df['timestamp'] = pd.to_numeric(positions_df['timestamp'], errors='coerce')
                active_positions = positions_df[positions_df['timestamp'] >= current_ts - 24 * 3600]
                features['active_positions_24h'] = len(active_positions)
                
                if 'liquidity' in active_positions.columns:
                    active_positions['liquidity'] = pd.to_numeric(active_positions['liquidity'], errors='coerce')
                    features['position_liquidity_24h'] = active_positions['liquidity'].sum()
        
        if mints_df is not None and len(mints_df) > 0:
            if 'timestamp' in mints_df.columns:
                mints_df['timestamp'] = pd.to_numeric(mints_df['timestamp'], errors='coerce')
                recent_mints = mints_df[mints_df['timestamp'] >= current_ts - 24 * 3600]
                features['mint_count_24h'] = len(recent_mints)
                
                if 'amount' in recent_mints.columns:
                    recent_mints['amount'] = pd.to_numeric(recent_mints['amount'], errors='coerce')
                    features['mint_amount_24h'] = recent_mints['amount'].sum()
        
        if burns_df is not None and len(burns_df) > 0:
            if 'timestamp' in burns_df.columns:
                burns_df['timestamp'] = pd.to_numeric(burns_df['timestamp'], errors='coerce')
                recent_burns = burns_df[burns_df['timestamp'] >= current_ts - 24 * 3600]
                features['burn_count_24h'] = len(recent_burns)
                
                if 'amount' in recent_burns.columns:
                    recent_burns['amount'] = pd.to_numeric(recent_burns['amount'], errors='coerce')
                    features['burn_amount_24h'] = recent_burns['amount'].sum()
        
        if 'mint_count_24h' in features and 'burn_count_24h' in features:
            features['net_liquidity_events_24h'] = features['mint_count_24h'] - features['burn_count_24h']
        
        return features

    def extract_indicator_features(self, prices, horizons=[6, 24, 168]):
        """
        RQ1: 单指标 BOCD 异常分数
        对币价时间序列运行 BOCD，返回异常强度分数
        """
        features = {}
        
        if prices is None or len(prices) < 24:
            return features
        
        # 计算收益率
        returns = np.diff(np.log(prices))
        returns = returns[~np.isnan(returns)]
        returns = returns[~np.isinf(returns)]
        
        if len(returns) < 10:
            return features
        
        # 对整个收益率序列运行 BOCD
        bocd = ExtendedBOCD(hazard_rate=200)
        for x in returns:
            bocd.update(x)
        
        bocd_features = bocd.get_latest_features()
        
        # 提取当前异常分数
        features['price_deviation'] = bocd_features.get('deviation', 0)
        features['price_cp_prob'] = bocd_features.get('cp_prob', 0)
        features['price_surprise'] = bocd_features.get('surprise', 0)
        
        # 对不同时间窗口计算异常
        for h in horizons:
            if len(returns) >= h:
                window_returns = returns[-h:]
                window_bocd = ExtendedBOCD(hazard_rate=200)
                for x in window_returns:
                    window_bocd.update(x)
                wf = window_bocd.get_latest_features()
                
                features[f'price_deviation_{h}h'] = wf.get('deviation', 0)
                features[f'price_cp_prob_{h}h'] = wf.get('cp_prob', 0)
        
        return features

    def extract_pool_features(self, pool_df, current_hour, windows=[6, 24, 48]):
        """
        RQ2: AMM 级别 - 池指标 BOCD 偏离特征
        对池的各个指标计算 BOCD 偏离特征
        """
        features = {}
        
        if pool_df is None or len(pool_df) < 10:
            return features
        
        # 指标列
        indicator_cols = ['entropy', 'netflow', 'markout', 'volume', 'tvl']
        
        for col in indicator_cols:
            if col not in pool_df.columns:
                continue
            
            values = pool_df[col].fillna(0).values
            
            # 对数差分
            try:
                log_values = np.diff(np.log(values + 1e-10))
                log_values = log_values[~np.isnan(log_values)]
                log_values = log_values[~np.isinf(log_values)]
            except:
                continue
            
            if len(log_values) < 10:
                continue
            
            # 运行 BOCD
            bocd = ExtendedBOCD(hazard_rate=200)
            for x in log_values:
                bocd.update(x)
            
            bocd_features = bocd.get_latest_features()
            
            # 当前偏离特征
            features[f'{col}_deviation'] = bocd_features.get('deviation', 0)
            features[f'{col}_cp_prob'] = bocd_features.get('cp_prob', 0)
            features[f'{col}_surprise'] = bocd_features.get('surprise', 0)
            
            # 窗口特征
            for w in windows:
                if len(log_values) >= w:
                    window_returns = log_values[-w:]
                    wbocd = ExtendedBOCD(hazard_rate=200)
                    for x in window_returns:
                        wbocd.update(x)
                    wf = wbocd.get_latest_features()
                    
                    features[f'{col}_deviation_{w}h'] = wf.get('deviation', 0)
                    features[f'{col}_cp_prob_{w}h'] = wf.get('cp_prob', 0)
        
        return features

    def extract_amm_features(self, pool_df, windows=[6, 24, 48]):
        """
        RQ2: 多指标融合特征
        统计异常指标数量和加权异常得分
        """
        features = {}
        
        if pool_df is None or len(pool_df) < 10:
            return features
        
        # 获取偏离特征
        deviation_features = self.extract_pool_features(pool_df, None, windows)
        
        # 统计异常指标数量 (cp_prob > 0.3 为异常)
        anomaly_indicators = []
        for col in ['entropy', 'netflow', 'markout', 'volume']:
            cp_prob_key = f'{col}_cp_prob'
            if cp_prob_key in deviation_features and deviation_features[cp_prob_key] > 0.3:
                anomaly_indicators.append(col)
        
        features['anomaly_count'] = len(anomaly_indicators)
        features['anomaly_ratio'] = len(anomaly_indicators) / 4.0
        
        # 加权异常得分 (所有 cp_prob 的加权平均)
        cp_scores = []
        for col in ['entropy', 'netflow', 'markout', 'volume']:
            cp_key = f'{col}_cp_prob'
            if cp_key in deviation_features:
                cp_scores.append(deviation_features[cp_key])
        
        if cp_scores:
            features['weighted_cp_score'] = np.mean(cp_scores)
            features['max_cp_score'] = np.max(cp_scores)
        
        # 偏离程度
        dev_scores = []
        for col in ['entropy', 'netflow', 'markout', 'volume']:
            dev_key = f'{col}_deviation'
            if dev_key in deviation_features:
                dev_scores.append(abs(deviation_features[dev_key]))
        
        if dev_scores:
            features['weighted_deviation'] = np.mean(dev_scores)
            features['max_deviation'] = np.max(dev_scores)
        
        # 更新池特征
        features.update(deviation_features)
        
        return features

    def extract_coin_features(self, coin, price_df, pool_metrics_df=None, current_idx=None):
        """
        RQ3: Coin 级别 - 整合币级别和池级别特征
        """
        features = {}
        
        if price_df is None or len(price_df) == 0:
            return features
        
        # 币级别特征 (Indicator-level, RQ1)
        prices = price_df['price'].values[:current_idx+1] if current_idx else price_df['price'].values
        indicator_feats = self.extract_indicator_features(prices)
        features.update(indicator_feats)
        
        # 锚定偏差
        if current_idx is not None:
            current_price = price_df.iloc[current_idx]['price']
        else:
            current_price = prices[-1] if len(prices) > 0 else 1.0
        features['anchor_deviation'] = abs(current_price - 1.0) * 100
        
        # 池级别特征 (AMM-level, RQ2)
        if pool_metrics_df is not None and len(pool_metrics_df) > 0:
            # 找到与当前时间对应的池数据
            if current_idx is not None and 'timestamp' in price_df.columns:
                current_ts = int(pd.Timestamp(price_df.iloc[current_idx]['timestamp']).timestamp())
                hour_ts = (current_ts // 3600) * 3600
                
                # 获取最近的小时数据
                recent_pool = pool_metrics_df[pool_metrics_df['hour'] <= hour_ts]
                if len(recent_pool) > 0:
                    recent_pool = recent_pool.sort_values('hour').tail(24)
                    amm_feats = self.extract_amm_features(recent_pool)
                    features.update(amm_feats)
            else:
                # 使用最新数据
                amm_feats = self.extract_amm_features(pool_metrics_df.tail(24))
                features.update(amm_feats)
        
        return features

    def load_pool_metrics(self, coin, data_dir='data', recent=False):
        """
        加载指定币种的池级别指标数据
        
        Args:
            coin: 币种名称 (如 USDC)
            data_dir: 数据目录
            recent: True 加载最近数据，False 加载完整数据
        
        Returns:
            DataFrame: 池指标数据
        """
        if recent:
            file_name = 'pool_metrics_recent.csv'
        else:
            file_name = 'pool_metrics_full.csv'
        
        pool_file = os.path.join(data_dir, coin.lower(), file_name)
        
        if os.path.exists(pool_file):
            self.pool_metrics_df = pd.read_csv(pool_file)
            print(f"Loaded pool metrics for {coin}: {len(self.pool_metrics_df)} rows")
            return self.pool_metrics_df
        else:
            print(f"Warning: Pool metrics file not found at {pool_file}")
            return None

    def load_pools_list(self, coin, data_dir='data'):
        """
        加载币种的池地址列表
        """
        pools_file = os.path.join(data_dir, coin.lower(), 'pools_list.csv')
        if os.path.exists(pools_file):
            return pd.read_csv(pools_file)
        return None

    def compute_bocd_from_series(self, series_values, hazard_rate=200):
        """对任意时间序列计算BOCD特征，返回特征字典"""
        if series_values is None or len(series_values) < 10:
            return None
        
        series = pd.Series(series_values).dropna()
        if len(series) < 10:
            return None
        
        try:
            returns = np.diff(np.log(series.values))
            returns = returns[~np.isnan(returns)]
            returns = returns[~np.isinf(returns)]
            
            if len(returns) < 10:
                return None
            
            bocd = ExtendedBOCD(hazard_rate=hazard_rate)
            for x in returns[-500:]:
                bocd.update(x)
            
            return bocd.get_features()
        except Exception as e:
            print(f"Error computing BOCD: {e}")
            return None

    def compute_bocd_from_series_fast(self, series_values, hazard_rate=200):
        """快速BOCD计算 - 限制计算点数"""
        if series_values is None or len(series_values) < 10:
            return None
        
        series = pd.Series(series_values).dropna()
        if len(series) < 10:
            return None
        
        try:
            returns = np.diff(np.log(series.values))
            returns = returns[~np.isnan(returns)]
            returns = returns[~np.isinf(returns)]
            
            if len(returns) < 10:
                return None
            
            # 限制计算点数以提高速度
            max_points = 200
            returns = returns[-max_points:]
            
            bocd = ExtendedBOCD(hazard_rate=hazard_rate)
            for x in returns:
                bocd.update(x)
            
            return bocd.get_features()
        except Exception:
            return None

    def extract_bocd_window_features(self, bocd_features, window_size, prefix):
        """从BOCD特征中提取窗口特征"""
        features = {}
        
        if bocd_features is None:
            return features
        
        cp_prob = bocd_features.get('cp_prob', np.array([0]))
        deviation = bocd_features.get('deviation', np.array([0]))
        entropy = bocd_features.get('entropy', np.array([0]))
        
        if len(cp_prob) < window_size:
            return features
        
        window_cp = cp_prob[-window_size:]
        window_dev = deviation[-window_size:]
        window_ent = entropy[-window_size:]
        
        features[f'{prefix}_cp_prob_mean_{window_size}h'] = np.mean(window_cp)
        features[f'{prefix}_cp_prob_max_{window_size}h'] = np.max(window_cp)
        features[f'{prefix}_cp_prob_std_{window_size}h'] = np.std(window_cp)
        
        features[f'{prefix}_deviation_mean_{window_size}h'] = np.mean(window_dev)
        features[f'{prefix}_deviation_max_{window_size}h'] = np.max(window_dev)
        features[f'{prefix}_deviation_std_{window_size}h'] = np.std(window_dev)
        
        features[f'{prefix}_entropy_mean_{window_size}h'] = np.mean(window_ent)
        features[f'{prefix}_entropy_trend_{window_size}h'] = window_ent[-1] - window_ent[0] if len(window_ent) > 0 else 0
        
        return features

    def compute_bocd_features(self, returns):
        if len(returns) < 10:
            return {
                'cp_prob': np.array([0]),
                'deviation': np.array([0]),
                'surprise': np.array([0]),
                'prediction_error': np.array([0]),
                'entropy': np.array([0]),
            }
        
        bocd = ExtendedBOCD(hazard_rate=200)
        for x in returns[-500:]:  # 只用最近500个点
            bocd.update(x)
        return bocd.get_features()

    def extract_window_features(self, data, window_size):
        features = {}
        
        cp_prob = data['cp_prob']
        deviation = data['deviation']
        surprise = data['surprise']
        prediction_error = data['prediction_error']
        entropy = data['entropy']
        
        if len(cp_prob) < window_size:
            return None
        
        window_cp = cp_prob[-window_size:]
        window_dev = deviation[-window_size:]
        window_surp = surprise[-window_size:]
        window_err = prediction_error[-window_size:]
        window_ent = entropy[-window_size:]
        
        features[f'cp_prob_mean_{window_size}h'] = np.mean(window_cp)
        features[f'cp_prob_max_{window_size}h'] = np.max(window_cp)
        features[f'cp_prob_std_{window_size}h'] = np.std(window_cp)
        
        if len(window_cp) >= 12:
            features[f'cp_prob_rise_rate_{window_size}h'] = (
                np.mean(window_cp[-6:]) - np.mean(window_cp[:-6])
            )
        else:
            features[f'cp_prob_rise_rate_{window_size}h'] = 0
        
        features[f'deviation_mean_{window_size}h'] = np.mean(window_dev)
        features[f'deviation_max_{window_size}h'] = np.max(window_dev)
        features[f'deviation_std_{window_size}h'] = np.std(window_dev)
        
        if len(window_dev) > 1:
            slope = np.polyfit(range(len(window_dev)), window_dev, 1)[0]
            features[f'deviation_trend_{window_size}h'] = slope
        else:
            features[f'deviation_trend_{window_size}h'] = 0
        
        features[f'surprise_mean_{window_size}h'] = np.mean(window_surp)
        features[f'surprise_max_{window_size}h'] = np.max(window_surp)
        
        features[f'prediction_error_mean_{window_size}h'] = np.mean(window_err)
        features[f'prediction_error_max_{window_size}h'] = np.max(window_err)
        
        features[f'entropy_mean_{window_size}h'] = np.mean(window_ent)
        features[f'entropy_trend_{window_size}h'] = (
            window_ent[-1] - window_ent[0]
        )
        
        return features

    def extract_price_features(self, prices, window_size):
        features = {}
        
        if len(prices) < window_size:
            return None
        
        window_prices = prices[-window_size:]
        returns = np.diff(np.log(window_prices))
        returns = returns[~np.isnan(returns)]
        returns = returns[~np.isinf(returns)]
        
        if len(returns) == 0:
            return None
        
        features[f'volatility_{window_size}h'] = np.std(returns)
        features[f'skewness_{window_size}h'] = skew(returns) if len(returns) > 2 else 0
        features[f'kurtosis_{window_size}h'] = kurtosis(returns) if len(returns) > 3 else 0
        
        features[f'cumreturn_{window_size}h'] = np.sum(returns)
        features[f'price_range_{window_size}h'] = (
            np.max(window_prices) - np.min(window_prices)
        ) / np.mean(window_prices)
        
        features[f'mean_reversion_{window_size}h'] = self._compute_mean_reversion(window_prices)
        
        return features

    def _compute_mean_reversion(self, prices):
        if len(prices) < 2:
            return 0
        mean_price = np.mean(prices)
        deviations = prices - mean_price
        return -np.corrcoef(deviations[:-1], deviations[1:])[0, 1] if len(deviations) > 1 else 0

    def extract_time_features(self, timestamp):
        features = {}
        features['hour_of_day'] = timestamp.hour
        features['day_of_week'] = timestamp.weekday()
        features['is_weekend'] = 1 if timestamp.weekday() >= 5 else 0
        features['is_market_hours'] = 1 if 9 <= timestamp.hour <= 16 else 0
        return features

    def extract_all_features(self, df, current_idx, metrics_df=None, 
                           curve_df=None, token_hour_df=None,
                           positions_df=None, mints_df=None, burns_df=None,
                           anchor=1.0):
        if current_idx < max(self.windows):
            return None
        
        all_features = {}
        
        prices = df['price'].values[:current_idx+1]
        returns = np.diff(np.log(prices))
        returns = returns[~np.isnan(returns)]
        returns = returns[~np.isinf(returns)]
        
        for window in self.windows:
            if len(returns) < window:
                continue
            
            window_returns = returns[-window:]
            
            all_features[f'volatility_{window}h'] = np.std(window_returns)
            all_features[f'cumreturn_{window}h'] = np.sum(window_returns)
            all_features[f'max_return_{window}h'] = np.max(window_returns)
            all_features[f'min_return_{window}h'] = np.min(window_returns)
            
            window_prices = prices[-window:]
            all_features[f'price_range_{window}h'] = (np.max(window_prices) - np.min(window_prices)) / np.mean(window_prices)
            
            all_features[f'anchor_deviation_{window}h'] = abs(window_prices[-1] - anchor)
        
        time_feats = self.extract_time_features(df.iloc[current_idx]['timestamp'])
        all_features.update(time_feats)
        
        # 仅对最大的窗口计算BOCD特征，避免计算量过大
        if metrics_df is not None and len(metrics_df) > 10:
            current_time = df.iloc[current_idx]['timestamp']
            current_ts = int(pd.Timestamp(current_time).timestamp())
            hour_ts = (current_ts // 3600) * 3600
            
            window = max(self.windows)  # 使用最大窗口
            
            window_data = metrics_df[
                (metrics_df['hour'] >= hour_ts - window * 3600) &
                (metrics_df['hour'] <= hour_ts)
            ]
            
            if len(window_data) >= 10:
                for metric_col, prefix in [('entropy', 'entropy'), ('flow', 'netflow'), ('markout', 'markout')]:
                    if metric_col in metrics_df.columns:
                        series = window_data[metric_col].dropna().values
                        if len(series) >= 10:
                            try:
                                bocd_features = self.compute_bocd_from_series_fast(series)
                                if bocd_features:
                                    all_features[f'{prefix}_cp_prob_mean'] = np.mean(bocd_features['cp_prob'][-window:])
                                    all_features[f'{prefix}_cp_prob_max'] = np.max(bocd_features['cp_prob'][-window:])
                                    all_features[f'{prefix}_deviation_mean'] = np.mean(bocd_features['deviation'][-window:])
                                    all_features[f'{prefix}_deviation_max'] = np.max(bocd_features['deviation'][-window:])
                            except Exception as e:
                                pass
        
        # Curve特征
        if curve_df is not None and len(curve_df) > 0:
            current_time = df.iloc[current_idx]['timestamp']
            current_ts = int(pd.Timestamp(current_time).timestamp())
            
            curve_idx = None
            if 'hour' in curve_df.columns:
                for i in range(len(curve_df) - 1, -1, -1):
                    if curve_df.iloc[i]['hour'] <= current_ts:
                        curve_idx = i
                        break
            
            if curve_idx is not None:
                curve_features = self.extract_curve_features(curve_df, curve_idx, anchor)
                all_features.update(curve_features)
        
        # TokenHourData特征
        if token_hour_df is not None and len(token_hour_df) > 0:
            current_time = df.iloc[current_idx]['timestamp']
            current_ts = int(pd.Timestamp(current_time).timestamp())
            
            token_idx = None
            if 'periodStartUnix' in token_hour_df.columns:
                for i in range(len(token_hour_df) - 1, -1, -1):
                    if token_hour_df.iloc[i]['periodStartUnix'] <= current_ts:
                        token_idx = i
                        break
            
            if token_idx is not None:
                token_hour_features = self.extract_token_hour_features(token_hour_df, token_idx)
                all_features.update(token_hour_features)
        
        # Position和Mint/Burn特征
        if positions_df is not None or mints_df is not None or burns_df is not None:
            current_time = df.iloc[current_idx]['timestamp']
            current_ts = int(pd.Timestamp(current_time).timestamp())
            
            position_features = self.extract_position_features(positions_df, mints_df, burns_df, current_ts)
            all_features.update(position_features)
        
        return all_features

    def generate_dataset(self, coin_name, data_dir='data', use_metrics=True, use_curve=False):
        price_file = os.path.join(data_dir, coin_name.lower(), 'prices.csv')
        label_file = os.path.join(data_dir, coin_name.lower(), 'depeg_labels.csv')
        
        if not os.path.exists(price_file):
            print(f"Price file not found for {coin_name}")
            return None, None
        
        df = pd.read_csv(price_file, parse_dates=['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        if os.path.exists(label_file):
            labels_df = pd.read_csv(label_file, parse_dates=['timestamp'])
            df = df.merge(
                labels_df[['timestamp', 'label_6h', 'label_24h', 'label_168h', 'anchor_deviation']],
                on='timestamp',
                how='left'
            )
        else:
            df['label_6h'] = 0
            df['label_24h'] = 0
            df['label_168h'] = 0
            df['anchor_deviation'] = abs(df['price'] - df.get('anchor', 1.0)) * 100
        
        anchor = df.get('anchor', [1.0] * len(df))[0] if 'anchor' in df.columns else 1.0
        
        metrics_df = None
        if use_metrics:
            metrics_df = self.load_metrics(coin_name, data_dir)
        
        curve_df = None
        token_hour_df = None
        positions_df = None
        mints_df = None
        burns_df = None
        
        if use_curve:
            curve_df = self.load_curve_data(coin_name, data_dir)
            token_hour_df = self.load_token_hour_data(coin_name, data_dir)
            positions_df = self.load_positions(coin_name, data_dir)
            mints_df, burns_df = self.load_mint_burn_events(coin_name, data_dir)
        
        all_samples = []
        
        for idx in range(max(self.windows), len(df) - max(self.horizons)):
            features = self.extract_all_features(
                df, idx, metrics_df, 
                curve_df, token_hour_df, 
                positions_df, mints_df, burns_df,
                anchor
            )
            if features is None:
                continue
            
            features['timestamp'] = df.iloc[idx]['timestamp']
            features['label_6h'] = df.iloc[idx]['label_6h']
            features['label_24h'] = df.iloc[idx]['label_24h']
            features['label_168h'] = df.iloc[idx]['label_168h']
            features['anchor_deviation'] = df.iloc[idx]['anchor_deviation']
            features['price'] = df.iloc[idx]['price']
            
            all_samples.append(features)
        
        features_df = pd.DataFrame(all_samples)
        
        output_file = os.path.join(data_dir, coin_name.lower(), 'features.csv')
        backup_path, output_file = save_csv_with_backup(features_df, output_file, index=False)
        if backup_path:
            print(f"{coin_name}: Existing features backed up to {backup_path}")
        print(f"{coin_name}: Generated {len(features_df)} samples, saved to {output_file}")
        
        X = features_df.drop(
            ['timestamp', 'label_6h', 'label_24h', 'label_168h'],
            axis=1, errors='ignore'
        )
        y_class = features_df[['label_6h', 'label_24h', 'label_168h']]
        
        return X, y_class

    def generate_hierarchical_dataset(self, coin_name, data_dir='data', use_pool_metrics=True):
        """
        生成分层特征数据集 (RQ1-RQ3)
        
        返回:
            DataFrame: 包含三层特征的样本
        """
        price_file = os.path.join(data_dir, coin_name.lower(), 'prices.csv')
        label_file = os.path.join(data_dir, coin_name.lower(), 'depeg_labels.csv')
        
        if not os.path.exists(price_file):
            print(f"Price file not found for {coin_name}")
            return None
        
        df = pd.read_csv(price_file, parse_dates=['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # 加载标签
        if os.path.exists(label_file):
            labels_df = pd.read_csv(label_file, parse_dates=['timestamp'])
            df = df.merge(
                labels_df[['timestamp', 'label_6h', 'label_24h', 'label_168h', 'anchor_deviation']],
                on='timestamp',
                how='left'
            )
        else:
            df['label_6h'] = 0
            df['label_24h'] = 0
            df['label_168h'] = 0
            df['anchor_deviation'] = abs(df['price'] - 1.0) * 100
        
        # 加载池指标
        pool_df = None
        if use_pool_metrics:
            pool_df = self.load_pool_metrics(coin_name, data_dir)
        
        all_samples = []
        
        for idx in range(max(self.windows), len(df) - max(self.horizons)):
            # 提取三层特征
            features = self.extract_coin_features(coin_name, df, pool_df, idx)
            
            if not features:
                continue
            
            # 添加时间戳和标签
            features['timestamp'] = df.iloc[idx]['timestamp']
            features['label_6h'] = df.iloc[idx].get('label_6h', 0)
            features['label_24h'] = df.iloc[idx].get('label_24h', 0)
            features['label_168h'] = df.iloc[idx].get('label_168h', 0)
            features['price'] = df.iloc[idx]['price']
            features['anchor_deviation'] = df.iloc[idx].get('anchor_deviation', 0)
            
            all_samples.append(features)
        
        if not all_samples:
            print(f"{coin_name}: No samples generated")
            return None
        
        features_df = pd.DataFrame(all_samples)
        
        output_file = os.path.join(data_dir, coin_name.lower(), 'features_hierarchical.csv')
        backup_path, output_file = save_csv_with_backup(features_df, output_file, index=False)
        if backup_path:
            print(f"{coin_name}: Existing hierarchical features backed up to {backup_path}")
        print(f"{coin_name}: Generated {len(features_df)} hierarchical samples, saved to {output_file}")
        
        return features_df


def main():
    # 从 config.yaml 读取 stablecoins 列表
    import yaml
    with open('config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    STABLECOINS = [coin['name'] for coin in config.get('stablecoins', [])]
    
    engineer = FeatureEngineer(windows=[6, 12, 24, 48, 72], horizons=[6, 24, 168])
    
    all_X = []
    all_y = []
    
    # 只处理有数据的币种
    for coin in STABLECOINS:
        price_file = os.path.join('data', coin.lower(), 'prices.csv')
        if not os.path.exists(price_file):
            print(f"Skipping {coin}: no price data")
            continue
            
        df = pd.read_csv(price_file, parse_dates=['timestamp'])
        if len(df) < 100:
            print(f"Skipping {coin}: insufficient data ({len(df)} rows)")
            continue
        
        print(f"\nProcessing {coin}...")
        
        # 生成标签
        labeler = DepegLabeler(threshold_pct=5, min_duration_hours=6)
        df = labeler.generate_future_labels(df)
        
        # 生成特征
        X, y = engineer.generate_dataset(coin, data_dir='data')
        
        if X is not None and len(X) > 0:
            X['coin'] = coin
            all_X.append(X)
            all_y.append(y)
    
    if not all_X:
        print("No data to process")
        return
    
    combined_X = pd.concat(all_X, ignore_index=True)
    combined_y = pd.concat(all_y, ignore_index=True)
    
    backup_path, combined_x_file = save_csv_with_backup(combined_X, 'data/combined_features.csv', index=False)
    if backup_path:
        print(f"Existing combined features backed up to {backup_path}")
    backup_path, combined_y_file = save_csv_with_backup(combined_y, 'data/combined_labels.csv', index=False)
    if backup_path:
        print(f"Existing combined labels backed up to {backup_path}")
    
    print(f"\nTotal samples: {len(combined_X)}")
    print(f"Combined features saved to: {combined_x_file}")
    print(f"Combined labels saved to: {combined_y_file}")
    print(f"Label distribution (label_24h):")
    print(combined_y['label_24h'].value_counts())


if __name__ == '__main__':
    main()
