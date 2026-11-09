import requests
import pandas as pd
import os
import json
import time
from datetime import datetime, timedelta
from datetime import datetime as dt
import threading
import queue
import yaml
try:
    from tools import save_csv_with_backup
except ImportError:
    from src.tools import save_csv_with_backup

PROXIES = {
    'http': 'http://127.0.0.1:7890',
    'https': 'http://127.0.0.1:7890'
}

RAW_DATA_DIR = 'data/raw data'
CURVE_DATA_DIR = 'data'

CURVE_SUBGRAPH_URL = "https://gateway.thegraph.com/api/subgraphs/id/3fy93eAT56UJsRCEht8iFhfi6wjHWXtZ9dnnbQmvFopF"

def get_api_key():
    """从config.yaml获取The Graph API key"""
    try:
        with open('config.yaml', 'r') as f:
            cfg = yaml.safe_load(f)
            return cfg.get('THE_GRAPH_API_KEY', '')
    except:
        return ''


STABLECOIN_POOLS = {
    'UST': {
        'pool_address': '0x890f6cd2d6a93f3d2bf2b24b28fb7f3d5fb2d3c',
        'anchor': 1.0,
        'start': '2022-04-01',
        'end': '2022-07-01',
    },
    'USDN': {
        'pool_address': '0x4f3E8F405CF5aFC05D68142F3783bDfE13811522',
        'anchor': 1.0,
        'start': '2022-06-01',
        'end': '2022-08-01',
    },
    'sUSD': {
        'pool_address': '0xc25a3a3b969415c80451098fa907ec722572917f',
        'anchor': 1.0,
        'start': '2025-03-01',
        'end': '2025-05-01',
    },
    'DEI': {
        'pool_address': '0x5F890651b14b8d3334E84776A9f9dE1A95d42eA',
        'anchor': 1.0,
        'start': '2022-05-01',
        'end': '2022-06-15',
    },
}


def parse_datetime_to_timestamp(date_str):
    """解析日期字符串为时间戳"""
    if isinstance(date_str, datetime):
        return int(date_str.timestamp())
    try:
        dt_obj = dt.strptime(date_str, '%Y-%m-%d')
        return int(dt_obj.timestamp())
    except:
        return int(dt.now().timestamp())


def query_subgraph(query, variables=None):
    """执行GraphQL查询"""
    api_key = get_api_key()
    headers = {
        "Content-Type": "application/json",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    
    try:
        response = requests.post(
            CURVE_SUBGRAPH_URL,
            headers=headers,
            json={"query": query, "variables": variables} if variables else {"query": query},
            timeout=60,
            proxies=PROXIES
        )
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        print(f"Subgraph query error: {e}")
    return None


def fetch_pool_hour_data(pool_addr, start_ts, end_ts, batch_size=1000):
    """获取Curve池小时数据"""
    all_data = []
    skip = 0
    
    query = """
    query GetPoolHourData($pool: String!, $start: Int!, $end: Int!, $skip: Int!, $first: Int!) {
        poolHourDatas(
            first: $first
            skip: $skip
            where: {
                pool: $pool
                timestamp_gte: $start
                timestamp_lte: $end
            }
            orderBy: timestamp
            orderDirection: asc
        ) {
            timestamp
            pool {
                id
                symbol
                name
            }
            virtualPrice
            A
            gamma
            dxVolume
            dyVolume
            totalVolume
            totalLiquidity
            baseApr
            boostedApr
        }
    }
    """
    
    while True:
        variables = {
            "pool": pool_addr,
            "start": start_ts,
            "end": end_ts,
            "skip": skip,
            "first": batch_size
        }
        
        result = query_subgraph(query, variables)
        if not result:
            break
            
        data = result.get('data', {}).get('poolHourDatas', [])
        if not data:
            break
            
        all_data.extend(data)
        skip += batch_size
        
        if len(data) < batch_size:
            break
    
    return all_data


def fetch_liquidity_events(pool_addr, start_ts, end_ts, event_type='add', batch_size=1000):
    """获取流动性事件（添加/移除）"""
    all_data = []
    skip = 0
    
    if event_type == 'add':
        query_name = "AddLiquidityEvents"
        query = """
        query GetAddLiquidity($pool: String!, $start: Int!, $end: Int!, $skip: Int!, $first: Int!) {
            addLiquidityEvents(
                first: $first
                skip: $skip
                where: {
                    pool: $pool
                    timestamp_gte: $start
                    timestamp_lte: $end
                }
                orderBy: timestamp
                orderDirection: asc
            ) {
                timestamp
                pool {
                    id
                    symbol
                }
                provider
                tokenAmounts
                liquidityMinted
                amounts
            }
        }
        """
    else:
        query_name = "RemoveLiquidityEvents"
        query = """
        query GetRemoveLiquidity($pool: String!, $start: Int!, $end: Int!, $skip: Int!, $first: Int!) {
            removeLiquidityEvents(
                first: $first
                skip: $skip
                where: {
                    pool: $pool
                    timestamp_gte: $start
                    timestamp_lte: $end
                }
                orderBy: timestamp
                orderDirection: asc
            ) {
                timestamp
                pool {
                    id
                    symbol
                }
                provider
                tokenAmounts
                liquidityBurned
                amounts
            }
        }
        """
    
    while True:
        variables = {
            "pool": pool_addr,
            "start": start_ts,
            "end": end_ts,
            "skip": skip,
            "first": batch_size
        }
        
        result = query_subgraph(query, variables)
        if not result:
            break
            
        data = result.get('data', {}).get(query_name, [])
        if not data:
            break
            
        all_data.extend(data)
        skip += batch_size
        
        if len(data) < batch_size:
            break
    
    return all_data


def fetch_exchange_events(pool_addr, start_ts, end_ts, batch_size=1000):
    """获取兑换事件"""
    all_data = []
    skip = 0
    
    query = """
    query GetExchanges($pool: String!, $start: Int!, $end: Int!, $skip: Int!, $first: Int!) {
        exchangeEvents(
            first: $first
            skip: $skip
            where: {
                pool: $pool
                timestamp_gte: $start
                timestamp_lte: $end
            }
            orderBy: timestamp
            orderDirection: asc
        ) {
            timestamp
            pool {
                id
                symbol
            }
            buyer
            seller
            tokensSold
            tokensBought
            soldId
            boughtId
        }
    }
    """
    
    while True:
        variables = {
            "pool": pool_addr,
            "start": start_ts,
            "end": end_ts,
            "skip": skip,
            "first": batch_size
        }
        
        result = query_subgraph(query, variables)
        if not result:
            break
            
        data = result.get('data', {}).get('exchangeEvents', [])
        if not data:
            break
            
        all_data.extend(data)
        skip += batch_size
        
        if len(data) < batch_size:
            break
    
    return all_data


def compute_curve_features(pool_df, anchor=1.0):
    """计算Curve特征（对齐Uniswap格式）"""
    if pool_df is None or len(pool_df) == 0:
        return pool_df
    
    df = pool_df.copy()
    
    df['hour'] = df['timestamp']
    df['datetime'] = pd.to_datetime(df['hour'], unit='s').dt.strftime('%Y-%m-%d %H:%M')
    df['liquidity'] = pd.to_numeric(df.get('totalLiquidity', 0), errors='coerce').fillna(0)
    df['tvlUSD'] = df['liquidity']
    df['volumeUSD'] = pd.to_numeric(df.get('dxVolume', 0), errors='coerce').fillna(0)
    df['token0Price'] = pd.to_numeric(df.get('virtualPrice', 1.0), errors='coerce').fillna(1.0)
    df['token1Price'] = df['token0Price']
    
    df['virtual_price_deviation'] = abs(df['token0Price'] - anchor)
    df['A'] = pd.to_numeric(df.get('A', 0), errors='coerce').fillna(0)
    df['baseApr'] = pd.to_numeric(df.get('baseApr', 0), errors='coerce').fillna(0)
    df['boostedApr'] = pd.to_numeric(df.get('boostedApr', 0), errors='coerce').fillna(0)
    
    if len(df) > 1:
        df['virtual_price_change_1h'] = df['token0Price'].pct_change(1)
        df['liquidity_change_1h'] = df['liquidity'].pct_change(1)
        df['volume_change_1h'] = df['volumeUSD'].pct_change(1)
        
        df['virtual_price_change_6h'] = df['token0Price'].pct_change(6)
        df['liquidity_change_6h'] = df['liquidity'].pct_change(6)
        
        df['volume_ma_24h'] = df['volumeUSD'].rolling(24, min_periods=1).mean()
        df['volume_spike'] = df['volumeUSD'] / (df['volume_ma_24h'] + 1e-10)
    
    for col in ['entropy', 'flow', 'markout', 'volume']:
        if col not in df.columns:
            df[col] = 0
        df[f'{col}_deviation'] = 0
        df[f'{col}_cp_prob'] = 0
        df[f'{col}_surprise'] = 0
    
    df['entropy_ld'] = 0
    df['flow_ld'] = 0
    
    return df


class CurveFetcher:
    """Curve数据获取器"""
    
    SUBGRAPH_URL = CURVE_SUBGRAPH_URL
    STABLECOIN_POOLS = STABLECOIN_POOLS
    
    def __init__(self, data_dir='data'):
        self.data_dir = data_dir
        os.makedirs(os.path.join(data_dir, 'raw data'), exist_ok=True)
    
    def fetch(self, coin_name, start_date=None, end_date=None):
        """获取单个币种的Curve数据"""
        if coin_name not in self.STABLECOIN_POOLS:
            print(f"Warning: {coin_name} not in Curve pool mapping")
            return None
        
        pool_info = self.STABLECOIN_POOLS[coin_name]
        pool_addr = pool_info['pool_address']
        anchor = pool_info.get('anchor', 1.0)
        
        if start_date is None:
            start_date = pool_info.get('start', '2022-01-01')
        if end_date is None:
            end_date = pool_info.get('end', datetime.now().strftime('%Y-%m-%d'))
        
        start_ts = parse_datetime_to_timestamp(start_date)
        end_ts = parse_datetime_to_timestamp(end_date)
        
        print(f"\nFetching Curve data for {coin_name}")
        print(f"  Pool: {pool_addr}")
        print(f"  Time range: {start_date} ~ {end_date}")
        
        hour_data = fetch_pool_hour_data(pool_addr, start_ts, end_ts)
        print(f"  Pool hour data: {len(hour_data)} records")
        
        add_events = fetch_liquidity_events(pool_addr, start_ts, end_ts, 'add')
        remove_events = fetch_liquidity_events(pool_addr, start_ts, end_ts, 'remove')
        print(f"  Add liquidity events: {len(add_events)}")
        print(f"  Remove liquidity events: {len(remove_events)}")
        
        exchanges = fetch_exchange_events(pool_addr, start_ts, end_ts)
        print(f"  Exchange events: {len(exchanges)}")
        
        coin_dir = os.path.join(self.data_dir, coin_name.lower())
        os.makedirs(coin_dir, exist_ok=True)
        
        if hour_data:
            df = pd.DataFrame(hour_data)
            df = compute_curve_features(df, anchor)
            
            output_file = os.path.join(coin_dir, 'curve_pool_hour.csv')
            backup_path, output_file = save_csv_with_backup(df, output_file, index=False)
            if backup_path:
                print(f"  Backed up existing file to: {backup_path}")
            print(f"  Saved: {output_file}")
            
            metrics_file = os.path.join(coin_dir, 'curve_metrics.csv')
            columns = ['hour', 'datetime', 'liquidity', 'tvlUSD', 'volumeUSD', 
                      'token0Price', 'A', 'baseApr', 'boostedApr',
                      'virtual_price_deviation', 'virtual_price_change_1h', 'liquidity_change_1h',
                      'volume_spike', 'entropy', 'flow', 'markout', 'volume',
                      'entropy_deviation', 'entropy_cp_prob', 'entropy_surprise',
                      'markout_deviation', 'markout_cp_prob', 'markout_surprise',
                      'volume_deviation', 'volume_cp_prob', 'volume_surprise']
            available_cols = [c for c in columns if c in df.columns]
            backup_path, metrics_file = save_csv_with_backup(df[available_cols], metrics_file, index=False)
            if backup_path:
                print(f"  Backed up existing file to: {backup_path}")
            print(f"  Saved: {metrics_file}")
        
        if add_events:
            add_df = pd.DataFrame(add_events)
            add_file = os.path.join(coin_dir, 'curve_add_liquidity.csv')
            backup_path, add_file = save_csv_with_backup(add_df, add_file, index=False)
            if backup_path:
                print(f"  Backed up existing file to: {backup_path}")
            print(f"  Saved: {add_file}")
        
        if remove_events:
            remove_df = pd.DataFrame(remove_events)
            remove_file = os.path.join(coin_dir, 'curve_remove_liquidity.csv')
            backup_path, remove_file = save_csv_with_backup(remove_df, remove_file, index=False)
            if backup_path:
                print(f"  Backed up existing file to: {backup_path}")
            print(f"  Saved: {remove_file}")
        
        if exchanges:
            ex_df = pd.DataFrame(exchanges)
            ex_file = os.path.join(coin_dir, 'curve_exchanges.csv')
            backup_path, ex_file = save_csv_with_backup(ex_df, ex_file, index=False)
            if backup_path:
                print(f"  Backed up existing file to: {backup_path}")
            print(f"  Saved: {ex_file}")
        
        return {
            'coin': coin_name,
            'pool': pool_addr,
            'hour_data': len(hour_data),
            'add_events': len(add_events),
            'remove_events': len(remove_events),
            'exchanges': len(exchanges),
        }


class CurveWebsocketMonitor:
    """Curve实时监控器"""
    
    def __init__(self, coin_name, pool_addr, callback=None):
        self.coin_name = coin_name
        self.pool_addr = pool_addr
        self.callback = callback
        self.running = False
        self.ws = None
        self.thread = None
        self.data_queue = queue.Queue()
        self.last_update = None
        
        coin_dir = os.path.join(CURVE_DATA_DIR, coin_name.lower())
        os.makedirs(coin_dir, exist_ok=True)
        self.output_file = os.path.join(coin_dir, 'curve_realtime.csv')
    
    def start(self):
        """启动实时监控"""
        if self.running:
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
        print(f"Curve websocket monitor started for {self.coin_name}")
    
    def stop(self):
        """停止监控"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        print(f"Curve websocket monitor stopped for {self.coin_name}")
    
    def _monitor_loop(self):
        """监控循环 - 定时轮询而非WebSocket"""
        while self.running:
            try:
                self._fetch_latest()
                time.sleep(60)
            except Exception as e:
                print(f"Monitor error: {e}")
                time.sleep(60)
    
    def _fetch_latest(self):
        """获取最新数据"""
        now = int(time.time())
        hour_ts = (now // 3600) * 3600
        start_ts = hour_ts - 3600
        
        data = fetch_pool_hour_data(self.pool_addr, start_ts, now)
        
        if data:
            df = pd.DataFrame(data)
            df = compute_curve_features(df)
            
            if os.path.exists(self.output_file):
                existing = pd.read_csv(self.output_file)
                df = pd.concat([existing, df], ignore_index=True)
                df = df.drop_duplicates(subset=['hour'], keep='last')
            
            backup_path, self.output_file = save_csv_with_backup(df, self.output_file, index=False)
            if backup_path:
                print(f"Backed up existing curve monitor file to: {backup_path}")
            self.last_update = time.time()
            
            if self.callback:
                self.callback(df.iloc[-1].to_dict())
            
            print(f"[{datetime.now().strftime('%H:%M:%S')}] {self.coin_name} updated: {df.iloc[-1]['token0Price']:.4f}")
    
    def get_latest(self):
        """获取最新数据"""
        if os.path.exists(self.output_file):
            df = pd.read_csv(self.output_file)
            if len(df) > 0:
                return df.iloc[-1].to_dict()
        return None


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Curve数据获取')
    parser.add_argument('--coin', type=str, default=None, help='币种名称')
    parser.add_argument('--all', action='store_true', help='获取所有Curve池数据')
    parser.add_argument('--start', type=str, default=None, help='开始日期')
    parser.add_argument('--end', type=str, default=None, help='结束日期')
    parser.add_argument('--monitor', action='store_true', help='实时监控模式')
    
    args = parser.parse_args()
    
    fetcher = CurveFetcher()
    
    coins_to_fetch = []
    
    if args.all:
        coins_to_fetch = list(STABLECOIN_POOLS.keys())
    elif args.coin:
        coins_to_fetch = [args.coin]
    else:
        print("请指定 --coin <币种> 或 --all")
        return
    
    if args.monitor:
        monitors = []
        for coin in coins_to_fetch:
            if coin in STABLECOIN_POOLS:
                pool_addr = STABLECOIN_POOLS[coin]['pool_address']
                monitor = CurveWebsocketMonitor(coin, pool_addr)
                monitor.start()
                monitors.append(monitor)
        
        print("\n实时监控中... 按 Ctrl+C 退出")
        try:
            while True:
                time.sleep(60)
        except KeyboardInterrupt:
            print("\n停止监控...")
            for m in monitors:
                m.stop()
    else:
        for coin in coins_to_fetch:
            result = fetcher.fetch(coin, args.start, args.end)
            if result:
                print(f"\n{coin} 完成: {result}")


if __name__ == "__main__":
    main()
