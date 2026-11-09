import requests
import pandas as pd
import time
import os
import yaml
from datetime import datetime, timedelta

try:
    from tools import backup_existing_file, save_csv_with_backup
except ImportError:
    from src.tools import backup_existing_file, save_csv_with_backup

PROXIES = {
    'http': 'http://127.0.0.1:7890',
    'https': 'http://127.0.0.1:7890'
}

DATA_DIR = 'data'
COINGECKO_RATE_LIMIT = 15


def load_config():
    """从config.yaml加载配置"""
    with open("config.yaml", 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


cfg = load_config()

THEGRAPH_V2_SUBGRAPH = cfg.get("thegraph", {}).get("subgraph_v2", 
    "https://gateway.thegraph.com/api/subgraphs/id/EYCKATKGBKLWvSfwvBjzfCBmGwYNdVkduYXVivCsLRFu")
THEGRAPH_API_KEY = cfg.get("thegraph", {}).get("api_key", 
    cfg.get("THE_GRAPH_API_KEY", ""))

STABLECOINS = {}
for coin in cfg.get("stablecoins", []):
    name = coin.get("name")
    STABLECOINS[name] = {
        'id': coin.get("coingecko_id"),
        'anchor': coin.get("anchor", 1.0),
        'binance': coin.get("binance"),
        'coinbase': coin.get("coinbase"),
    }


class PriceFetcher:
    def __init__(self, data_dir=DATA_DIR):
        self.data_dir = data_dir
        self.rate_limit = COINGECKO_RATE_LIMIT

    def fetch_from_coinbase(self, product_id, days=365):
        url = f"https://api.exchange.coinbase.com/products/{product_id}/candles"
        
        all_candles = []
        
        batch_days = 10
        batches = (days + batch_days - 1) // batch_days
        
        for i in range(batches):
            end_ts = int(datetime.now().timestamp()) - i * batch_days * 86400
            start_ts = end_ts - batch_days * 86400
            
            params = {
                'start': datetime.fromtimestamp(start_ts).isoformat(),
                'end': datetime.fromtimestamp(end_ts).isoformat(),
                'granularity': 3600
            }
            
            try:
                response = requests.get(url, params=params, timeout=30, proxies=PROXIES)
                
                if response.status_code != 200:
                    print(f"Coinbase error {response.status_code}")
                    break
                
                candles = response.json()
                if not candles:
                    continue
                
                all_candles.extend(candles)
                time.sleep(0.5)
                
            except Exception as e:
                print(f"Coinbase API error: {e}")
                break
        
        if not all_candles:
            return None
        
        df = pd.DataFrame([
            {
                'timestamp': datetime.fromtimestamp(c[0]),
                'price': (c[3] + c[4]) / 2,
                'open': c[3],
                'high': c[2],
                'low': c[1],
                'close': c[4],
                'volume': c[5],
                'market_cap': 0,
                'total_volume': 0,
            }
            for c in all_candles
        ])
        
        df = df.sort_values('timestamp').reset_index(drop=True)
        return df

    def fetch_from_thegraph(self, token_address, start_ts=None, end_ts=None):
        """从TheGraph获取价格数据 - 支持增量获取"""
        url = THEGRAPH_V2_SUBGRAPH
        headers = {"Authorization": f"Bearer {THEGRAPH_API_KEY}"}
        
        if end_ts is None:
            end_ts = int(datetime.now().timestamp())
        if start_ts is None:
            start_ts = int((datetime.now() - timedelta(days=365*3)).timestamp())
        
        all_data = []
        skip = 0
        batch_size = 1000
        
        query = """
        {{
          tokenDayDatas(
            first: {batch_size},
            skip: {skip},
            where: {{
              token: "{token}"
              date_gte: {start}
              date_lte: {end}
            }},
            orderBy: date,
            orderDirection: asc
          ) {{
            date
            priceUSD
          }}
        }}
        """
        
        print(f"TheGraph查询: token={token_address[:10]}..., start={start_ts}, end={end_ts}")
        
        while True:
            q = query.format(
                token=token_address,
                start=start_ts,
                end=end_ts,
                batch_size=batch_size,
                skip=skip
            )
            
            try:
                response = requests.post(url, headers=headers, json={"query": q}, timeout=60, proxies=PROXIES)
                
                if response.status_code != 200:
                    print(f"TheGraph error {response.status_code}")
                    break
                
                data = response.json()
                if 'errors' in data:
                    print(f"TheGraph errors: {data['errors']}")
                    break
                
                token_data = data.get('data', {}).get('tokenDayDatas', [])
                if not token_data:
                    break
                
                all_data.extend(token_data)
                skip += batch_size
                
                if len(token_data) < batch_size:
                    break
                    
            except Exception as e:
                print(f"TheGraph error: {e}")
                break
        
        if not all_data:
            return None
        
        df = pd.DataFrame([
            {
                'timestamp': datetime.fromtimestamp(d['date']),
                'price': float(d['priceUSD']) if d['priceUSD'] else 1.0,
                'volume': 0,
                'market_cap': 0,
                'total_volume': 0,
            }
            for d in all_data
        ])
        
        return df

    def fetch_from_binance(self, symbol, days=365):
        url = "https://api.binance.com/api/v3/klines"
        
        end_time = int(datetime.now().timestamp() * 1000)
        start_time = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
        
        all_klines = []
        limit = 1000
        
        while start_time < end_time:
            params = {
                'symbol': symbol,
                'interval': '1h',
                'startTime': start_time,
                'endTime': end_time,
                'limit': limit
            }
            
            try:
                response = requests.get(url, params=params, timeout=30, proxies=PROXIES)
                if response.status_code != 200:
                    print(f"Binance error {response.status_code}: {response.text[:100]}")
                    break
                    
                klines = response.json()
                if not klines:
                    break
                    
                all_klines.extend(klines)
                start_time = klines[-1][0] + 3600 * 1000
                
            except Exception as e:
                print(f"Binance API error: {e}")
                break
        
        if not all_klines:
            return None
            
        df = pd.DataFrame([
            {
                'timestamp': pd.to_datetime(k[0], unit='ms'),
                'price': (float(k[1]) + float(k[4])) / 2,
                'open': float(k[1]),
                'high': float(k[2]),
                'low': float(k[3]),
                'close': float(k[4]),
                'volume': float(k[5]),
                'market_cap': 0,
                'total_volume': float(k[7]) if len(k) > 7 else 0,
            }
            for k in all_klines
        ])
        
        return df

    def fetch_from_coingecko(self, coin_id, days=365*3, max_retries=5):
        url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
        
        for attempt in range(max_retries):
            try:
                params = {
                    'vs_currency': 'usd',
                    'days': days,
                }
                
                response = requests.get(url, params=params, timeout=60, proxies=PROXIES)
                
                if response.status_code == 429:
                    wait_time = (attempt + 1) * 30
                    print(f"Rate limited, waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                
                if response.status_code != 200:
                    print(f"CoinGecko error {response.status_code}")
                    break
                
                data = response.json()
                
                if 'prices' not in data or not data['prices']:
                    return None
                
                prices = data['prices']
                
                df = pd.DataFrame([
                    {
                        'timestamp': pd.to_datetime(p[0], unit='ms'),
                        'price': p[1],
                        'open': None,
                        'high': None,
                        'low': None,
                        'close': None,
                        'volume': 0,
                        'market_cap': 0,
                        'total_volume': 0,
                    }
                    for p in prices
                ])
                
                return df
                
            except requests.exceptions.Timeout:
                print(f"Timeout, attempt {attempt + 1}/{max_retries}")
                time.sleep(10)
            except requests.exceptions.ConnectionError as e:
                print(f"Connection error: {e}")
                time.sleep(10)
            except Exception as e:
                print(f"Error: {e}")
                break
        
        return None

    def fetch_historical(self, coin_name, force_refresh=False):
        """获取历史价格数据 - 支持增量更新"""
        from src.tools import parse_datetime_to_timestamp
        
        coin_dir = os.path.join(self.data_dir, coin_name.lower())
        os.makedirs(coin_dir, exist_ok=True)
        
        output_file = os.path.join(coin_dir, 'prices.csv')
        
        # 获取币种配置
        coin_config = None
        for c in cfg.get("stablecoins", []):
            if c.get("name") == coin_name:
                coin_config = c
                break
        
        # 检查现有数据
        if not force_refresh and os.path.exists(output_file):
            existing_df = pd.read_csv(output_file, parse_dates=['timestamp'])
            if len(existing_df) >= 100:
                last_date = existing_df['timestamp'].max()
                days_from_last = (datetime.now() - last_date).days
                if days_from_last < 1:
                    print(f"{coin_name}: 数据已是最新")
                    return existing_df
                
                # 增量更新：计算新的时间范围
                if coin_config:
                    start_str = coin_config.get("start", "")
                    end_str = coin_config.get("end", "")
                    
                    if start_str and end_str:
                        config_start = parse_datetime_to_timestamp(start_str)
                        config_end = parse_datetime_to_timestamp(end_str)
                        
                        # 获取现有数据的最大时间
                        existing_max_ts = int(existing_df['timestamp'].max().timestamp())
                        
                        if existing_max_ts < config_end:
                            print(f"{coin_name}: 增量更新现有数据...")
                            # 从现有数据之后继续获取
                            return self._fetch_incremental(coin_name, coin_config, existing_df, output_file)
        
        # 全量获取
        if os.path.exists(output_file):
            backup_path = backup_existing_file(output_file)
            if backup_path:
                print(f"{coin_name}: existing prices backed up to {backup_path}")
            print(f"{coin_name}: 重新获取全部数据")
        
        return self._full_fetch(coin_name, coin_config, output_file)

    def _fetch_incremental(self, coin_name, coin_config, existing_df, output_file):
        """增量获取数据"""
        # 优先从Binance获取
        binance_symbol = STABLECOINS.get(coin_name, {}).get('binance')
        if binance_symbol:
            print(f"从Binance增量获取 {coin_name}...")
            df = self.fetch_from_binance(binance_symbol, days=365)
            if df is not None and len(df) > 0:
                # 合并数据
                combined = pd.concat([existing_df, df], ignore_index=True)
                combined = combined.drop_duplicates(subset=['timestamp'], keep='last')
                combined = combined.sort_values('timestamp')
                combined['datetime'] = pd.to_datetime(combined['timestamp'], unit='s').dt.strftime('%Y-%m-%d %H:00')
                backup_path, output_file = save_csv_with_backup(combined, output_file, index=False)
                if backup_path:
                    print(f"{coin_name}: existing incremental prices backed up to {backup_path}")
                print(f"{coin_name}: 增量更新完成，共 {len(combined)} 条")
                return combined
        
        return existing_df

    def _full_fetch(self, coin_name, coin_config, output_file):
        """全量获取数据"""
        # 1. 优先从Binance获取小时级数据
        binance_symbol = STABLECOINS.get(coin_name, {}).get('binance')
        if binance_symbol:
            print(f"从Binance获取 {coin_name}...")
            df = self.fetch_from_binance(binance_symbol, days=365)
            if df is not None and len(df) > 0:
                df['datetime'] = pd.to_datetime(df['timestamp'], unit='s').dt.strftime('%Y-%m-%d %H:00')
                backup_path, output_file = save_csv_with_backup(df, output_file, index=False)
                if backup_path:
                    print(f"{coin_name}: existing Binance prices backed up to {backup_path}")
                print(f"{coin_name}: 保存 {len(df)} 条记录")
                time.sleep(1)
                return df
        
        # 2. Binance失败，尝试Coinbase
        coinbase_symbol = STABLECOINS.get(coin_name, {}).get('coinbase')
        if coinbase_symbol:
            print(f"从Coinbase获取 {coin_name}...")
            df = self.fetch_from_coinbase(coinbase_symbol, days=365)
            if df is not None and len(df) > 0:
                df['datetime'] = pd.to_datetime(df['timestamp'], unit='s').dt.strftime('%Y-%m-%d %H:00')
                backup_path, output_file = save_csv_with_backup(df, output_file, index=False)
                if backup_path:
                    print(f"{coin_name}: existing Coinbase prices backed up to {backup_path}")
                print(f"{coin_name}: 保存 {len(df)} 条记录")
                time.sleep(1)
                return df
        
        # 3. 尝试从TheGraph获取
        if coin_config and coin_config.get("token_id"):
            print(f"从TheGraph获取 {coin_name}...")
            token_id = coin_config.get("token_id")
            df = self.fetch_from_thegraph(token_id)
            if df is not None and len(df) > 0:
                df['datetime'] = pd.to_datetime(df['timestamp'], unit='s').dt.strftime('%Y-%m-%d %H:00')
                backup_path, output_file = save_csv_with_backup(df, output_file, index=False)
                if backup_path:
                    print(f"{coin_name}: existing TheGraph prices backed up to {backup_path}")
                print(f"{coin_name}: 保存 {len(df)} 条记录")
                time.sleep(1)
                return df
        
        # 没有可用数据源
        print(f"{coin_name}: 无可用数据源")
        return None

    def fetch_all_coins(self):
        """获取所有配置中稳定币的价格数据"""
        results = {}
        for coin_name in STABLECOINS:
            df = self.fetch_historical(coin_name)
            if df is not None:
                results[coin_name] = df
        return results

    def update_realtime(self, coin_name):
        """更新实时数据"""
        return self.fetch_historical(coin_name, force_refresh=False)


if __name__ == '__main__':
    fetcher = PriceFetcher()
    fetcher.fetch_all_coins()
