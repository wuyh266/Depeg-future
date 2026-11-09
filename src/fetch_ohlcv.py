import os
import pandas as pd
import ccxt
from datetime import datetime, timedelta
from dateutil import parser as dateparser
import time
import requests
import yaml
try:
    from tools import save_csv_with_backup
except ImportError:
    from src.tools import save_csv_with_backup

# 加载配置
_config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.yaml')
if not os.path.exists(_config_path):
    _config_path = 'config.yaml'
_cfg = yaml.safe_load(open(_config_path, encoding='utf-8'))

PROXIES = {
    'http': _cfg.get("proxy", {}).get("http", "http://127.0.0.1:7890"),
    'https': _cfg.get("proxy", {}).get("https", "http://127.0.0.1:7890")
}

RAW_DATA_DIR = _cfg.get("data_path", {}).get("raw", "data/raw data")

EXCHANGES = _cfg.get("ohlcv", {}).get("exchanges", ["bybit", "coinbase", "binance", "okx", "kucoin"])
SYMBOLS = _cfg.get("ohlcv", {}).get("symbols", {})


class OHLCVFetcher:
    def __init__(self):
        self.exchanges = EXCHANGES
        os.makedirs(RAW_DATA_DIR, exist_ok=True)
    
    def _get_symbol(self, coin_name, exchange):
        """获取交易对符号"""
        if coin_name not in SYMBOLS:
            return None
        if SYMBOLS[coin_name] is None:
            return None
        if exchange in SYMBOLS[coin_name]:
            return SYMBOLS[coin_name][exchange]
        return None
    
    def _get_existing_time_range(self, coin_name):
        """获取现有数据的时间范围"""
        filepath = self._get_filepath(coin_name)
        if not os.path.exists(filepath):
            return None, None
        
        df = pd.read_csv(filepath)
        if len(df) == 0:
            return None, None
        
        return df['timestamp'].min(), df['timestamp'].max()
    
    def _get_filepath(self, coin_name):
        """获取文件路径"""
        return os.path.join(RAW_DATA_DIR, f'ohlcv_{coin_name}.csv')
    
    def _parse_datetime(self, date_str):
        """解析日期字符串"""
        if isinstance(date_str, datetime):
            return date_str
        return dateparser.parse(date_str)
    
    def _fetch_from_exchange(self, exchange_id, symbol, since, until):
        """从单个交易所获取数据（同步版本）"""
        try:
            exchange_class = getattr(ccxt, exchange_id)
            exchange = exchange_class({
                'proxies': PROXIES,
                'enableRateLimit': True,
                'timeout': 30000,
            })
            
            all_data = []
            current_since = since * 1000
            until_ms = until * 1000
            
            while current_since < until_ms:
                try:
                    ohlcv = exchange.fetch_ohlcv(
                        symbol,
                        timeframe='1m',
                        since=current_since,
                        limit=1000
                    )
                    
                    if not ohlcv:
                        break
                    
                    first_ts = ohlcv[0][0]
                    last_ts = ohlcv[-1][0]
                    
                    if first_ts > until_ms:
                        if last_ts > until_ms:
                            break
                    
                    filtered_ohlcv = [k for k in ohlcv if since * 1000 <= k[0] <= until_ms]
                    
                    if filtered_ohlcv:
                        all_data.extend(filtered_ohlcv)
                    
                    if last_ts >= until_ms - 60000:
                        break
                    
                    if last_ts <= current_since - 60000:
                        break
                    
                    current_since = last_ts + 60000
                    time.sleep(0.5)
                    
                except Exception as e:
                    print(f"      错误: {type(e).__name__}: {e}")
                    break
            
            if all_data:
                df = pd.DataFrame([
                    {
                        'timestamp': k[0] // 1000,
                        'datetime': datetime.fromtimestamp(k[0] // 1000).strftime('%Y-%m-%d %H:%M:%S'),
                        'open': k[1],
                        'high': k[2],
                        'low': k[3],
                        'close': k[4],
                        'volume': k[5],
                        'exchange': exchange_id
                    }
                    for k in all_data
                ])
                df = df.sort_values('timestamp')
                return df
            
            return None
            
        except Exception as e:
            print(f"      错误: {type(e).__name__}: {e}")
            return None
    
    def fetch(self, coin_name, start_date, end_date):
        """
        获取OHLCV数据
        
        Args:
            coin_name: 币种名称
            start_date: 开始日期 (str or datetime)
            end_date: 结束日期 (str or datetime)
        
        Returns:
            DataFrame: OHLCV数据
        """
        yaml_start = self._parse_datetime(start_date)
        yaml_end = self._parse_datetime(end_date)
        
        yaml_start_ts = int(yaml_start.timestamp())
        yaml_end_ts = int(yaml_end.timestamp())
        
        existing_start_ts, existing_end_ts = self._get_existing_time_range(coin_name)
        
        fetch_ranges = []
        
        if existing_start_ts is None:
            fetch_ranges.append((yaml_start_ts, yaml_end_ts))
        else:
            if existing_start_ts > yaml_start_ts:
                fetch_ranges.append((yaml_start_ts, existing_start_ts))
            
            if existing_end_ts is not None and existing_end_ts < yaml_end_ts:
                fetch_ranges.append((existing_end_ts, yaml_end_ts))
        
        if not fetch_ranges:
            print(f"  {coin_name}: 数据已覆盖yaml时间范围，跳过")
            return None
        
        print(f"  {coin_name}: 需要获取 {len(fetch_ranges)} 个区间")
        
        for since, until in fetch_ranges:
            print(f"    获取区间: {datetime.fromtimestamp(since)} ~ {datetime.fromtimestamp(until)}")
            
            success = False
            for exchange in self.exchanges:
                symbol = self._get_symbol(coin_name, exchange)
                if not symbol:
                    continue
                
                print(f"    尝试交易所: {exchange}")
                
                df = self._fetch_from_exchange(exchange, symbol, since, until)
                
                if df is not None and len(df) > 0:
                    print(f"    成功: 获取 {len(df)} 条记录 (交易所: {exchange})")
                    
                    filepath = self._get_filepath(coin_name)
                    
                    if os.path.exists(filepath):
                        existing_df = pd.read_csv(filepath)
                        df = pd.concat([existing_df, df], ignore_index=True)
                        df = df.drop_duplicates(subset=['timestamp'], keep='last')
                        df = df.sort_values('timestamp')
                    
                    backup_path, filepath = save_csv_with_backup(df, filepath, index=False)
                    if backup_path:
                        print(f"    已备份旧 OHLCV 文件: {backup_path}")
                    print(f"    已保存到: {filepath}")
                    success = True
                    break
            
            if not success:
                print(f"    警告: 所有交易所获取失败")
        
        return None


def fetch_ohlcv_for_coin(coin_name, start_date, end_date):
    """便捷函数：获取单个币种的OHLCV数据"""
    fetcher = OHLCVFetcher()
    return fetcher.fetch(coin_name, start_date, end_date)


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        coin_name = sys.argv[1]
        start = sys.argv[2] if len(sys.argv) > 2 else '2023-02-01'
        end = sys.argv[3] if len(sys.argv) > 3 else '2023-04-01'
        
        print(f"获取 {coin_name} OHLCV 数据: {start} ~ {end}")
        df = fetch_ohlcv_for_coin(coin_name, start, end)
        
        if df is not None:
            print(f"成功: {len(df)} 条记录")
        else:
            print("无数据")
    else:
        print("用法: python fetch_ohlcv.py <币种名称> [开始日期] [结束日期]")
        print("示例: python fetch_ohlcv.py USDC 2023-02-01 2023-04-01")
