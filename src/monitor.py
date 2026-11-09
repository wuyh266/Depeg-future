import numpy as np
import pandas as pd
import os
import json
import time
import subprocess
import sys
import yaml
from datetime import datetime, timedelta
import pickle

from fetch_price import PriceFetcher, STABLECOINS
from depeg_label import DepegLabeler
from bocd_extended import ExtendedBOCD
from feature_engineering import FeatureEngineer


class DepegMonitor:
    def __init__(self, model_dir='data/models', data_dir='data', 
                 update_metrics=True, metrics_interval=3600, metrics_days=7,
                 use_curve_monitor=False):
        self.model_dir = model_dir
        self.data_dir = data_dir
        self.stablecoins = list(STABLECOINS.keys())
        
        self.update_metrics = update_metrics
        self.metrics_interval = metrics_interval
        self.metrics_days = metrics_days
        self.last_metrics_update = 0
        
        self.use_curve_monitor = use_curve_monitor
        self.curve_monitors = {}
        
        self.predictor = None
        self.feature_engineer = FeatureEngineer(
            windows=[6, 12, 24, 48, 72],
            horizons=[6, 24, 168]
        )
        
        self.bocd_instances = {coin: ExtendedBOCD(hazard_rate=200) for coin in self.stablecoins}
        self.price_cache = {coin: None for coin in self.stablecoins}
        
        self.use_hierarchical = False
        self.load_models()
        
        if self.use_curve_monitor:
            self.init_curve_monitors()

    def should_update_metrics(self):
        """检查是否需要更新池数据"""
        if not self.update_metrics:
            return False
        
        current_time = time.time()
        return (current_time - self.last_metrics_update) >= self.metrics_interval

    def update_pool_metrics(self):
        """调用 main.py 更新池级别指标数据"""
        print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Updating pool metrics...")
        
        try:
            # 调用 main.py 获取最近 N 天的池数据
            main_script = os.path.join(os.path.dirname(__file__), 'main.py')
            
            for coin in self.stablecoins:
                result = subprocess.run(
                    [sys.executable, main_script, '--coin', coin, '--days', str(self.metrics_days)],
                    capture_output=True,
                    text=True,
                    timeout=300
                )
                
                if result.returncode == 0:
                    print(f"  {coin}: Updated successfully")
                else:
                    print(f"  {coin}: Update failed - {result.stderr[:100]}")
            
            self.last_metrics_update = time.time()
            print("Pool metrics update complete")
            
        except subprocess.TimeoutExpired:
            print("Warning: Pool metrics update timed out")
        except Exception as e:
            print(f"Warning: Failed to update pool metrics: {e}")

    def init_curve_monitors(self):
        """初始化Curve实时监控"""
        try:
            from fetch_curve import CurveWebsocketMonitor
            
            config_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.yaml')
            if os.path.exists(config_file):
                with open(config_file, 'r') as f:
                    cfg = yaml.safe_load(f)
                    curve_pools = cfg.get('curve_pools', {})
            else:
                curve_pools = {}
            
            for coin_name, pool_info in curve_pools.items():
                pool_addr = pool_info.get('pool_address', '')
                if pool_addr:
                    monitor = CurveWebsocketMonitor(coin_name, pool_addr, callback=self.on_curve_update)
                    monitor.start()
                    self.curve_monitors[coin_name] = monitor
                    print(f"Curve monitor started for {coin_name}")
            
            if self.curve_monitors:
                print(f"Initialized {len(self.curve_monitors)} Curve monitors")
        
        except Exception as e:
            print(f"Warning: Failed to initialize Curve monitors: {e}")

    def on_curve_update(self, data):
        """Curve数据更新回调"""
        coin = data.get('coin', 'unknown')
        vp = data.get('token0Price', 1.0)
        deviation = abs(vp - 1.0) * 100
        
        print(f"[Curve Update] {coin}: virtual_price={vp:.4f}, deviation={deviation:.2f}%")
        
        if deviation > 5:
            print(f"  ⚠️ WARNING: {coin} depeg deviation > 5%!")

    def update_curve_metrics(self):
        """更新Curve池数据"""
        if not self.curve_monitors:
            return
        
        print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Updating Curve metrics...")
        
        for coin_name, monitor in self.curve_monitors.items():
            try:
                monitor._fetch_latest()
                print(f"  {coin_name}: Updated")
            except Exception as e:
                print(f"  {coin_name}: Update failed - {e}")
        
        print("Curve metrics update complete")

    def stop_curve_monitors(self):
        """停止所有Curve监控"""
        for coin_name, monitor in self.curve_monitors.items():
            try:
                monitor.stop()
                print(f"Curve monitor stopped for {coin_name}")
            except Exception as e:
                print(f"Warning: Failed to stop {coin_name}: {e}")
        self.curve_monitors = {}

    def load_models(self):
        # 尝试加载简化模型
        simple_model_path = os.path.join(self.model_dir, 'rf_classifier_simple.pkl')
        if os.path.exists(simple_model_path):
            with open(simple_model_path, 'rb') as f:
                self.rf_classifier = pickle.load(f)
            # 加载特征列
            feature_cols_path = os.path.join(self.model_dir, 'feature_cols.pkl')
            with open(feature_cols_path, 'rb') as f:
                self.feature_cols = pickle.load(f)
            self.use_simple_model = True
            print("Simple model loaded successfully")
            return
        
        # 尝试加载分层模型
        global_model_path = os.path.join(self.model_dir, 'global_rf.pkl')
        if os.path.exists(global_model_path):
            self.use_hierarchical = True
            print("Hierarchical model (train_predict.py) not included in this pipeline; skipping.")
            return
        
        # 回退到标准模型
        model_file = os.path.join(self.model_dir, 'rf_classifier.pkl')
        if not os.path.exists(model_file):
            print("Warning: Models not found. Please run train_predict.py first.")
            return
        
        with open(model_file, 'rb') as f:
            self.rf_classifier = pickle.load(f)
        
        xgb_file = os.path.join(self.model_dir, 'xgb_classifier.pkl')
        if os.path.exists(xgb_file):
            with open(xgb_file, 'rb') as f:
                self.xgb_classifier = pickle.load(f)
            self.has_xgb = True
        else:
            self.has_xgb = False
        
        print("Standard models loaded successfully")

    def fetch_latest_prices(self):
        fetcher = PriceFetcher(self.data_dir)
        
        for coin in self.stablecoins:
            df = fetcher.update_realtime(coin)
            if df is not None:
                self.price_cache[coin] = df
        
        return self.price_cache

    def update_bocd(self, coin, returns):
        bocd = self.bocd_instances[coin]
        
        for x in returns:
            bocd.update(x)
        
        return bocd.get_features()

    def extract_features_single(self, coin):
        price_file = os.path.join(self.data_dir, coin.lower(), 'prices.csv')
        
        if not os.path.exists(price_file):
            return None
        
        df = pd.read_csv(price_file, parse_dates=['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        if len(df) < 72:
            return None
        
        # 加载池指标（如果有且时间范围匹配）
        pool_metrics_file = os.path.join(self.data_dir, coin.lower(), 'pool_metrics_full.csv')
        metrics_df = None
        if os.path.exists(pool_metrics_file):
            pool_metrics = pd.read_csv(pool_metrics_file)
            # 检查池数据时间范围是否与价格数据重叠
            if len(pool_metrics) > 0:
                # 使用最近的数据点进行时间范围检测
                recent_price = df.tail(100)
                price_start = int(recent_price['timestamp'].iloc[0].timestamp())
                price_end = int(recent_price['timestamp'].iloc[-1].timestamp())
                pool_start = pool_metrics['hour'].min()
                pool_end = pool_metrics['hour'].max()
                # 如果时间范围有重叠才使用
                if price_start <= pool_end and price_end >= pool_start:
                    # 只保留最近的数据
                    pool_metrics = pool_metrics[pool_metrics['hour'] >= price_start]
                    metrics_df = pool_metrics
                    print(f"Using pool metrics for {coin}: {len(metrics_df)} rows")
                else:
                    print(f"Pool metrics time range does not overlap with price data for {coin}")
        
        features = self.feature_engineer.extract_all_features(df, len(df)-1, metrics_df)
        
        # 添加锚定偏差和价格
        features['anchor_deviation'] = abs(df['price'].iloc[-1] - 1.0) * 100
        features['price'] = df['price'].iloc[-1]
        
        return features

    def predict_single(self, coin):
        features = self.extract_features_single(coin)
        
        if features is None:
            return None
        
        feature_df = pd.DataFrame([features])
        
        # 使用简化模型
        if hasattr(self, 'use_simple_model') and self.use_simple_model:
            feature_df = feature_df[self.feature_cols]
            rf_prob = self.rf_classifier.predict_proba(feature_df)[0, 1]
            return rf_prob
        
        # 分层预测
        if hasattr(self, 'use_hierarchical') and self.use_hierarchical:
            return self._predict_hierarchical_single(coin, feature_df)
        
        # 标准预测
        rf_prob = self.rf_classifier.predict_proba(feature_df)[0, 1]
        
        if hasattr(self, 'has_xgb') and self.has_xgb:
            xgb_prob = self.xgb_classifier.predict_proba(feature_df)[0, 1]
            prob = (rf_prob + xgb_prob) / 2
        else:
            prob = rf_prob
        
        return prob
        
        # 分层预测
        if self.use_hierarchical:
            return self._predict_hierarchical_single(coin, feature_df)
        
        # 标准预测
        rf_prob = self.rf_classifier.predict_proba(feature_df)[0, 1]
        
        if self.has_xgb:
            xgb_prob = self.xgb_classifier.predict_proba(feature_df)[0, 1]
            prob = (rf_prob + xgb_prob) / 2
        else:
            prob = rf_prob
        
        return prob

    def _predict_hierarchical_single(self, coin, feature_df):
        """分层预测: Layer 1 -> Layer 2 -> Layer 3"""
        try:
            # Layer 2: AMM 评分
            if coin in self.hierarchical_predictor.amm_models:
                X_amm = feature_df[self.hierarchical_predictor.amm_feature_cols].fillna(0)
                amm_score = self.hierarchical_predictor.amm_models[coin].predict_proba(X_amm)[0, 1]
            else:
                amm_score = 0.0
            
            # Layer 3: 全局预测
            feature_df['amm_score'] = amm_score
            X_global = feature_df[self.hierarchical_predictor.layer3_feature_cols].fillna(0)
            prob = self.hierarchical_predictor.global_model.predict_proba(X_global)[0, 1]
            
            return prob
        except Exception as e:
            print(f"Warning: Hierarchical prediction failed for {coin}: {e}")
            return 0.0

    def predict_all(self):
        results = {}
        
        for coin in self.stablecoins:
            prob = self.predict_single(coin)
            
            if prob is not None:
                results[coin] = {
                    'depeg_probability': float(prob),
                    'risk_level': self._get_risk_level(prob),
                    'timestamp': datetime.now().isoformat()
                }
        
        return results

    def _get_risk_level(self, prob):
        if prob >= 0.7:
            return 'CRITICAL'
        elif prob >= 0.5:
            return 'HIGH'
        elif prob >= 0.3:
            return 'MEDIUM'
        elif prob >= 0.1:
            return 'LOW'
        else:
            return 'SAFE'

    def run_continuous(self, interval_seconds=3600):
        print(f"Starting continuous monitoring (interval: {interval_seconds}s)")
        print(f"  Update metrics: {self.update_metrics}")
        print(f"  Metrics interval: {self.metrics_interval}s")
        print(f"  Hierarchical model: {self.use_hierarchical}")
        print(f"  Curve monitor: {self.use_curve_monitor}")
        print("=" * 60)
        
        # 初始更新
        if self.update_metrics:
            self.update_pool_metrics()
        
        while True:
            try:
                print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Updating prices...")
                
                self.fetch_latest_prices()
                
                # 检查是否需要更新池数据
                if self.should_update_metrics():
                    self.update_pool_metrics()
                
                # 更新Curve数据
                if self.use_curve_monitor and self.curve_monitors:
                    self.update_curve_metrics()
                
                print("Running predictions...")
                results = self.predict_all()
                
                print("\n" + "=" * 60)
                print("DEPEG RISK MONITORING RESULTS")
                print("=" * 60)
                
                for coin, data in sorted(results.items(), 
                                         key=lambda x: x[1]['depeg_probability'], 
                                         reverse=True):
                    print(f"\n{coin}:")
                    print(f"  Probability: {data['depeg_probability']:.2%}")
                    print(f"  Risk Level:  {data['risk_level']}")
                
                print("\n" + "=" * 60)
                
                with open('data/monitoring_results.json', 'w') as f:
                    json.dump(results, f, indent=2)
                
                print(f"Results saved to data/monitoring_results.json")
                
                time.sleep(interval_seconds)
                
            except KeyboardInterrupt:
                print("\nMonitoring stopped by user")
                if self.use_curve_monitor and self.curve_monitors:
                    self.stop_curve_monitors()
                break
            except Exception as e:
                print(f"Error: {e}")
                time.sleep(60)

    def get_status_report(self):
        results = self.predict_all()
        
        report = []
        report.append("=" * 50)
        report.append("STABLECOIN DEPEG RISK STATUS REPORT")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("=" * 50)
        
        sorted_results = sorted(
            results.items(),
            key=lambda x: x[1]['depeg_probability'],
            reverse=True
        )
        
        for coin, data in sorted_results:
            prob = data['depeg_probability']
            risk = data['risk_level']
            
            report.append(f"\n{coin}:")
            report.append(f"  Depeg Probability: {prob:.2%}")
            report.append(f"  Risk Level:        {risk}")
        
        report.append("\n" + "=" * 50)
        
        return "\n".join(report)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Stablecoin Depeg Monitor')
    parser.add_argument('--continuous', action='store_true', help='Run continuous monitoring')
    parser.add_argument('--interval', type=int, default=3600, help='Update interval in seconds')
    parser.add_argument('--once', action='store_true', help='Run once and exit')
    parser.add_argument('--update-metrics', action='store_true', default=True,
                        help='Update pool metrics before prediction (default: True)')
    parser.add_argument('--no-update-metrics', action='store_false', dest='update_metrics',
                        help='Skip pool metrics update')
    parser.add_argument('--metrics-interval', type=int, default=3600,
                        help='Interval for metrics update in seconds (default: 3600)')
    parser.add_argument('--metrics-days', type=int, default=7,
                        help='Number of days to fetch for metrics (default: 7)')
    parser.add_argument('--curve-monitor', action='store_true',
                        help='Enable Curve realtime monitoring')
    
    args = parser.parse_args()
    
    monitor = DepegMonitor(
        update_metrics=args.update_metrics,
        metrics_interval=args.metrics_interval,
        metrics_days=args.metrics_days,
        use_curve_monitor=args.curve_monitor
    )
    
    if args.continuous:
        monitor.run_continuous(interval_seconds=args.interval)
    elif args.once:
        results = monitor.predict_all()
        print(monitor.get_status_report())
    else:
        results = monitor.predict_all()
        
        print("\n" + "=" * 50)
        print("CURRENT DEPEG RISK STATUS")
        print("=" * 50)
        
        for coin, data in sorted(results.items(), 
                                 key=lambda x: x[1]['depeg_probability'], 
                                 reverse=True):
            print(f"\n{coin}:")
            print(f"  Probability: {data['depeg_probability']:.2%}")
            print(f"  Risk Level:  {data['risk_level']}")


if __name__ == '__main__':
    main()
