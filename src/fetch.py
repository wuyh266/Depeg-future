import pandas as pd
import os
import sys
import time
import yaml

# 加载配置
_config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.yaml')
if not os.path.exists(_config_path):
    _config_path = 'config.yaml'
_cfg = yaml.safe_load(open(_config_path, encoding='utf-8'))

PROXIES = {
    'http': _cfg.get("proxy", {}).get("http", "http://127.0.0.1:7890"),
    'https': _cfg.get("proxy", {}).get("https", "http://127.0.0.1:7890")
}

# 添加路径支持
if __name__ == '__main__':
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if os.path.basename(os.path.dirname(os.path.abspath(__file__))) == 'src':
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from client import GraphClient
    from tools import parse_datetime_to_timestamp
except ImportError:
    from src.client import GraphClient
    from src.tools import parse_datetime_to_timestamp

# 从配置中获取数据路径
raw_data_path = _cfg.get("data_path", {}).get("raw", "data/raw data")
MAX_RETRIES = _cfg.get("network", {}).get("max_retries", 3)
RETRY_DELAY = _cfg.get("network", {}).get("retry_delay", 5)

# 从配置中获取 subgraph URLs
UNISWAP_V2_FACTORY = _cfg.get("subgraph", "https://gateway.thegraph.com/api/subgraphs/id/EYCKATKGBKLWvSfwvBjzfCBmGwYNdVkduYXVivCsLRFu")
UNISWAP_V3_FACTORY = _cfg.get("subgraph_uniswap_v3", "https://gateway.thegraph.com/api/subgraphs/id/5zvR82QoaXYFyDEKLZ9t6v9adgnptxYpKpSbxtgVENFV")
UNISWAP_V4_FACTORY = _cfg.get("subgraph_uniswap_v4", "https://gateway.thegraph.com/api/subgraphs/id/DiYPVdygkfjDWhbxGSqAQxwBKmfKnkWQojqeM2rkLb3G")

def is_v3_subgraph(url):
    """判断是否为 Uniswap V3 subgraph"""
    return 'uniswap-v3' in url.lower() or '5zvR82QoaXYFyDEKLZ9t6v9adgnptxYpKpSbxtgVENFV' in url

def is_v4_subgraph(url):
    """判断是否为 Uniswap V4 subgraph"""
    return 'uniswap-v4' in url.lower() or 'DiYPVdygkfjDWhbxGSqAQxwBKmfKnkWQojqeM2rkLb3G' in url


def get_token_pools(token_address, api_key, min_tvl=1000000, top_n=5):
    """
    从 TheGraph 查询代币池列表 (支持 V2 和 V3)
    
    Args:
        token_address: 代币地址 (如 0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48)
        api_key: TheGraph API 密钥
        min_tvl: 最小 TVL 阈值 (USD)
        top_n: 返回前 N 个池
    
    Returns:
        list: [{'address': pool地址, 'tvl': TVL, 'factory': 'v2/v3', 'tokens': [token0, token1]}]
    """
    import requests
    
    pools = []
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    # Uniswap V2 查询
    v2_query = """
    {{
      pairs(
        where: {{or: [
          {{token0: "{token}"}},
          {{token1: "{token}"}}
        ]}}
        orderBy: reserveUSD
        orderDirection: desc
        first: {top_n}
      ) {{
        id
        token0 {{
          symbol
          id
        }}
        token1 {{
          symbol
          id
        }}
        reserveUSD
      }}
    }}
    """.format(token=token_address.lower(), top_n=top_n)
    
    try:
        response = requests.post(
            UNISWAP_V2_FACTORY, 
            headers=headers, 
            json={"query": v2_query}, 
            timeout=30,
            proxies=PROXIES
        )
        if response.status_code == 200:
            data = response.json()
            pairs = data.get('data', {}).get('pairs', [])
            for p in pairs:
                tvl = float(p.get('reserveUSD', 0) or 0)
                if tvl >= min_tvl:
                    pools.append({
                        'address': p['id'],
                        'tvl': tvl,
                        'factory': 'v2',
                        'token0': p['token0']['symbol'],
                        'token1': p['token1']['symbol'],
                        'token0_addr': p['token0']['id'],
                        'token1_addr': p['token1']['id']
                    })
    except Exception as e:
        print(f"V2 query error: {e}")
    
    # Uniswap V3 查询
    v3_query = """
    {{
      pools(
        where: {{or: [
          {{token0: "{token}"}},
          {{token1: "{token}"}}
        ]}}
        orderBy: totalValueLockedUSD
        orderDirection: desc
        first: {top_n}
      ) {{
        id
        token0 {{
          symbol
          id
        }}
        token1 {{
          symbol
          id
        }}
        totalValueLockedUSD
      }}
    }}
    """.format(token=token_address.lower(), top_n=top_n)
    
    try:
        response = requests.post(
            UNISWAP_V3_FACTORY, 
            headers=headers, 
            json={"query": v3_query}, 
            timeout=30,
            proxies=PROXIES
        )
        if response.status_code == 200:
            data = response.json()
            pools_data = data.get('data', {}).get('pools', [])
            for p in pools_data:
                tvl = float(p.get('totalValueLockedUSD', 0) or 0)
                if tvl >= min_tvl:
                    pools.append({
                        'address': p['id'],
                        'tvl': tvl,
                        'factory': 'v3',
                        'token0': p['token0']['symbol'],
                        'token1': p['token1']['symbol'],
                        'token0_addr': p['token0']['id'],
                        'token1_addr': p['token1']['id']
                    })
    except Exception as e:
        print(f"V3 query error: {e}")
    
    # Uniswap V4 查询
    v4_query = """
    {{
      pools(
        where: {{or: [
          {{token0: "{token}"}},
          {{token1: "{token}"}}
        ]}}
        orderBy: totalValueLockedUSD
        orderDirection: desc
        first: {top_n}
      ) {{
        id
        token0 {{
          symbol
          id
        }}
        token1 {{
          symbol
          id
        }}
        totalValueLockedUSD
      }}
    }}
    """.format(token=token_address.lower(), top_n=top_n)
    
    try:
        response = requests.post(
            UNISWAP_V4_FACTORY, 
            headers=headers, 
            json={"query": v4_query}, 
            timeout=30,
            proxies=PROXIES
        )
        if response.status_code == 200:
            data = response.json()
            pools_data = data.get('data', {}).get('pools', [])
            for p in pools_data:
                tvl = float(p.get('totalValueLockedUSD', 0) or 0)
                if tvl >= min_tvl:
                    pools.append({
                        'address': p['id'],
                        'tvl': tvl,
                        'factory': 'v4',
                        'token0': p['token0']['symbol'],
                        'token1': p['token1']['symbol'],
                        'token0_addr': p['token0']['id'],
                        'token1_addr': p['token1']['id']
                    })
    except Exception as e:
        print(f"V4 query error: {e}")
    
    # 过滤异常 TVL 的池（如 > 1e10 是异常数据）
    max_tvl = 1e10  # 100亿 USD 是合理的上限
    pools = [p for p in pools if p['tvl'] <= max_tvl]
    
    # 过滤掉不包含目标 token 的交易对
    token_lower = token_address.lower()
    pools = [p for p in pools if p['token0_addr'].lower() == token_lower or p['token1_addr'].lower() == token_lower]
    
    # 版本优先级排序: V3 > V4 > V2 (V3历史数据更完整)
    version_priority = {'v3': 3, 'v4': 2, 'v2': 1}
    pools.sort(key=lambda x: (version_priority.get(x['factory'], 0), x['tvl']), reverse=True)
    return pools[:top_n]


def get_token_pools_at_time(token_address, api_key, timestamp, min_tvl=1000, top_n=3):
    """
    从 TheGraph 查询特定时间点的代币池列表 (按历史TVL排序)
    
    Args:
        token_address: 代币地址 (如 0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48)
        api_key: TheGraph API 密钥
        timestamp: 历史时间戳 (如 1739923200)
        min_tvl: 最小 TVL 阈值 (USD)
        top_n: 返回前 N 个池
    
    Returns:
        list: [{'address': pool地址, 'tvl': 历史TVL, 'factory': 'v3', ...}]
    """
    import requests
    
    # 先获取当前存在的候选池
    candidate_pools = get_token_pools(token_address, api_key, min_tvl=min_tvl, top_n=20)
    
    if not candidate_pools:
        return []
    
    pools_with_historical_tvl = []
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    # 查询每个池在指定时间点的TVL
    for pool in candidate_pools:
        pool_addr = pool['address']
        factory = pool['factory']
        
        if factory == 'v3':
            subgraph_url = UNISWAP_V3_FACTORY
        elif factory == 'v4':
            subgraph_url = UNISWAP_V4_FACTORY
        else:
            subgraph_url = UNISWAP_V2_FACTORY
        
        # 查询poolHourDatas获取历史TVL
        query = """
        {{
          poolHourDatas(
            first: 1
            where: {{
              pool: "{pool}"
              periodStartUnix_lte: {timestamp}
            }}
            orderBy: periodStartUnix
            orderDirection: desc
          ) {{
            periodStartUnix
            tvlUSD
          }}
        }}
        """.format(pool=pool_addr, timestamp=timestamp)
        
        try:
            response = requests.post(
                subgraph_url,
                headers=headers,
                json={"query": query},
                timeout=30,
                proxies=PROXIES
            )
            if response.status_code == 200:
                data = response.json()
                hour_data = data.get('data', {}).get('poolHourDatas', [])
                if hour_data and len(hour_data) > 0:
                    hist_tvl = float(hour_data[0].get('tvlUSD', 0) or 0)
                    if hist_tvl >= min_tvl:
                        pool_copy = pool.copy()
                        pool_copy['tvl'] = hist_tvl
                        pool_copy['historical_tvl'] = hist_tvl
                        pool_copy['tvl_timestamp'] = hour_data[0].get('periodStartUnix')
                        pools_with_historical_tvl.append(pool_copy)
        except Exception as e:
            continue
    
    # 按历史TVL排序
    pools_with_historical_tvl.sort(key=lambda x: x['tvl'], reverse=True)
    return pools_with_historical_tvl[:top_n]


class PoolFetcher:
    """专门用于获取池数据的 fetcher"""
    
    def __init__(self, pool_address, subgraph_url, api_key):
        self.pool = pool_address
        self.client = GraphClient(subgraph_url, api_key)
        self.is_v3 = is_v3_subgraph(subgraph_url)
        self.is_v4 = is_v4_subgraph(subgraph_url)
        self.version = 'v4' if self.is_v4 else ('v3' if self.is_v3 else 'v2')
    
    def fetch_swaps(self, start_ts, end_ts):
        """获取 Swap 数据"""
        if self.is_v4:
            tpl = """
            {{
              swaps(
                first:1000
                skip:{{skip}}
                where:{{
                  pool:"{pool}"
                  timestamp_gte:{start}
                  timestamp_lte:{end}
                }}
                orderBy: timestamp
                orderDirection: asc
              ){{
                timestamp
                amount0
                amount1
                origin
              }}
            }}
            """
        elif self.is_v3:
            tpl = """
            {{
              swaps(
                first:1000
                skip:{{skip}}
                where={{
                  pool:"{pool}"
                  timestamp_gte:{start}
                  timestamp_lte:{end}
                }}
                orderBy: timestamp
                orderDirection: asc
              ){{
                timestamp
                amount0
                amount1
                origin
              }}
            }}
            """
        else:
            tpl = """
            {{
              swaps(
                first:1000
                skip:{{skip}}
                where:{{
                  pair:"{pool}"
                  timestamp_gte:{start}
                  timestamp_lte:{end}
                }}
                orderBy: timestamp
                orderDirection: asc
              ){{
                timestamp
                amount0In
                amount1In
                amount0Out
                amount1Out
                sender
              }}
            }}
            """
        tpl = tpl.format(pool=self.pool, start=start_ts, end=end_ts, skip="{skip}")
        entity = "swaps"
        return self.client.paged_query(tpl, entity)
    
    def fetch_pool_hours(self, start_ts, end_ts):
        """获取池小时数据"""
        if self.is_v4:
            tpl = """
            {{
              poolHourDatas(
                first:1000
                skip:{{skip}}
                where={{
                  pool:"{pool}"
                  periodStartUnix_gte: {start}
                  periodStartUnix_lte: {end}
                }}
                orderBy:periodStartUnix
                orderDirection:asc
              ) {{
                periodStartUnix
                liquidity
                token0Price
                token1Price
                volumeUSD
                tvlUSD
                volumeToken0
                volumeToken1
                feesUSD
              }}
            }}
            """
            entity = "poolHourDatas"
        elif self.is_v3:
            tpl = """
            {{
              poolHourDatas(
                first:1000
                skip:{{skip}}
                where={{
                  pool:"{pool}"
                  periodStartUnix_gte: {start}
                  periodStartUnix_lte: {end}
                }}
                orderBy:periodStartUnix
                orderDirection:asc
              ) {{
                periodStartUnix
                liquidity
                token0Price
                token1Price
                volumeUSD
                tvlUSD
                volumeToken0
                volumeToken1
                feesUSD
              }}
            }}
            """
            entity = "poolHourDatas"
        else:
            tpl = """
            {{
              pairHourDatas(
                first:1000
                skip:{{skip}}
                where={{
                  pair:"{pool}"
                  hourStartUnix_gte: {start}
                  hourStartUnix_lte: {end}
                }}
                orderBy:hourStartUnix
                orderDirection:asc
              ) {{
                hourStartUnix
                reserve0
                reserve1
                volumeUSD
                reserveUSD
              }}
            }}
            """
            entity = "pairHourDatas"
        tpl = tpl.format(pool=self.pool, start=start_ts, end=end_ts, skip="{skip}")
        return self.client.paged_query(tpl, entity)

    def fetch_token_hour_data(self, start_ts, end_ts):
        """获取TokenHourData - 代币级别小时数据（跨池聚合）"""
        if not self.token_id:
            return pd.DataFrame()
        
        query = """
        {{
          tokenHourDatas(
            first: 1000
            skip: {{skip}}
            where: {{
              token: "{token}"
              periodStartUnix_gte: {start}
              periodStartUnix_lte: {end}
            }}
            orderBy: periodStartUnix
            orderDirection: asc
          ) {{
            periodStartUnix
            volume
            volumeUSD
            totalValueLocked
            totalValueLockedUSD
            priceUSD
            feesUSD
            open
            high
            low
            close
          }}
        }}
        """.format(token=self.token_id.lower(), start=start_ts, end=end_ts, skip="{skip}")
        
        entity = "tokenHourDatas"
        data = self.client.paged_query(query, entity)
        
        if data:
            df = pd.DataFrame(data)
            return df
        return pd.DataFrame()

    def fetch_positions(self, start_ts, end_ts):
        """获取Position数据 - 集中流动性头寸变化"""
        if not self.pool or not self.is_v3:
            return pd.DataFrame()
        
        query = """
        {{
          positions(
            first: 1000
            skip: {{skip}}
            where: {{
              pool: "{pool}"
              timestamp_gte: {start}
              timestamp_lte: {end}
            }}
            orderBy: timestamp
            orderDirection: asc
          ) {{
            id
            owner
            pool {{
              id
            }}
            tickLower {{
              tickIdx
            }}
            tickUpper {{
              tickIdx
            }}
            liquidity
            depositedToken0
            depositedToken1
            withdrawnToken0
            withdrawnToken1
            collectedFeesToken0
            collectedFeesToken1
            timestamp
          }}
        }}
        """.format(pool=self.pool, start=start_ts, end=end_ts, skip="{skip}")
        
        entity = "positions"
        data = self.client.paged_query(query, entity)
        
        if data:
            df = pd.DataFrame(data)
            return df
        return pd.DataFrame()

    def fetch_mints(self, start_ts, end_ts):
        """获取Mint事件 - 流动性创建"""
        if not self.pool:
            return pd.DataFrame()
        
        if self.is_v3:
            query = """
            {{
              mints(
                first: 1000
                skip: {{skip}}
                where: {{
                  pool: "{pool}"
                  timestamp_gte: {start}
                  timestamp_lte: {end}
                }}
                orderBy: timestamp
                orderDirection: asc
              ) {{
                id
                pool {{
                  id
                }}
                owner
                sender
                origin
                amount
                amount0
                amount1
                tickLower
                tickUpper
                timestamp
              }}
            }}
            """.format(pool=self.pool, start=start_ts, end=end_ts, skip="{skip}")
            entity = "mints"
        else:
            query = """
            {{
              mints(
                first: 1000
                skip: {{skip}}
                where: {{
                  pair: "{pool}"
                  timestamp_gte: {start}
                  timestamp_lte: {end}
                }}
                orderBy: timestamp
                orderDirection: asc
              ) {{
                id
                pair {{
                  id
                }}
                to
                liquidity
                amount0
                amount1
                timestamp
              }}
            }}
            """.format(pool=self.pool, start=start_ts, end=end_ts, skip="{skip}")
            entity = "mints"
        
        data = self.client.paged_query(query, entity)
        
        if data:
            df = pd.DataFrame(data)
            return df
        return pd.DataFrame()

    def fetch_burns(self, start_ts, end_ts):
        """获取Burn事件 - 流动性销毁"""
        if not self.pool:
            return pd.DataFrame()
        
        if self.is_v3:
            query = """
            {{
              burns(
                first: 1000
                skip: {{skip}}
                where: {{
                  pool: "{pool}"
                  timestamp_gte: {start}
                  timestamp_lte: {end}
                }}
                orderBy: timestamp
                orderDirection: asc
              ) {{
                id
                pool {{
                  id
                }}
                owner
                origin
                amount
                amount0
                amount1
                tickLower
                tickUpper
                timestamp
              }}
            }}
            """.format(pool=self.pool, start=start_ts, end=end_ts, skip="{skip}")
            entity = "burns"
        else:
            query = """
            {{
              burns(
                first: 1000
                skip: {{skip}}
                where: {{
                  pair: "{pool}"
                  timestamp_gte: {start}
                  timestamp_lte: {end}
                }}
                orderBy: timestamp
                orderDirection: asc
              ) {{
                id
                pair {{
                  id
                }}
                sender
                liquidity
                amount0
                amount1
                timestamp
              }}
            }}
            """.format(pool=self.pool, start=start_ts, end=end_ts, skip="{skip}")
            entity = "burns"
        
        data = self.client.paged_query(query, entity)
        
        if data:
            df = pd.DataFrame(data)
            return df
        return pd.DataFrame()


class Fetcher:

    def __init__(self, cfg, client=None):
        api_key = cfg.get("api_keys", {}).get("the_graph", cfg.get("THE_GRAPH_API_KEY", ""))
        if client is None:
            self.client = GraphClient(cfg["subgraph"], api_key)
            self.subgraph_url = cfg["subgraph"]
        else:
            self.client = GraphClient(client, api_key)
            self.subgraph_url = client
        
        self.pool = cfg["pools"][0]
        self.cfg = cfg
        self.is_v3 = is_v3_subgraph(self.subgraph_url)
        self.is_v4 = is_v4_subgraph(self.subgraph_url)
        self.version = 'v4' if self.is_v4 else ('v3' if self.is_v3 else 'v2')
        
        # 从配置中获取稳定币列表
        self.stablecoins = cfg.get("stablecoins", [])
        
        # 默认值
        self.token_id = cfg.get("token_id", "")
        self.token_name = cfg.get("token_name", "")
        
        # 解析时间范围
        self.start = cfg.get("start", "")
        self.end = cfg.get("end", "")
        
        # 如果配置的是日期字符串，转换为时间戳
        if self.start and isinstance(self.start, str):
            self.start = parse_datetime_to_timestamp(self.start)
        if self.end and isinstance(self.end, str):
            self.end = parse_datetime_to_timestamp(self.end)

    def set_coin_config(self, coin_config):
        """
        设置当前要获取的币种配置
        
        Args:
            coin_config: 包含 name, token_id, start, end 的字典
        """
        self.token_id = coin_config.get("token_id", "")
        
        start_str = coin_config.get("start", "")
        end_str = coin_config.get("end", "")
        
        if start_str:
            self.start = parse_datetime_to_timestamp(start_str)
        if end_str:
            self.end = parse_datetime_to_timestamp(end_str)
        
        print(f"设置币种: {coin_config.get('name')}, 时间范围: {self.start} - {self.end}")

    def _get_existing_data_range(self, filepath, time_field):
        """获取现有数据的起止时间范围"""
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            if len(df) > 0 and time_field in df.columns:
                return df[time_field].min(), df[time_field].max()
        return None, None

    def _get_incremental_params(self, filepath, time_field, config_start, config_end):
        """
        计算增量更新的起止时间
        
        Returns:
            tuple: (start_ts, end_ts, status_message)
        """
        existing_min, existing_max = self._get_existing_data_range(filepath, time_field)
        
        if existing_max is not None:
            # 已有数据，从最新时间之后开始获取
            start_ts = int(existing_max) + 1
            end_ts = config_end
            if start_ts >= config_end:
                return None, None, f"数据已是最新 (最后记录: {existing_max})"
            return start_ts, end_ts, f"增量更新: {existing_max} -> {config_end}"
        
        # 无现有数据，使用配置的时间范围
        return config_start, config_end, "全量获取"

    def _save_incremental_data(self, filepath, new_data, time_field, key_field=None):
        """
        增量保存数据：合并现有数据和新数据，去重，排序
        
        Args:
            filepath: 现有数据文件路径
            new_data: 新获取的数据DataFrame
            time_field: 时间字段名
            key_field: 可选，用于去重的关键字段
        """
        if new_data is None or len(new_data) == 0:
            if os.path.exists(filepath):
                return pd.read_csv(filepath)
            return pd.DataFrame()
        
        if os.path.exists(filepath):
            existing_df = pd.read_csv(filepath)
            if len(existing_df) > 0:
                combined = pd.concat([existing_df, new_data], ignore_index=True)
                
                if key_field and key_field in combined.columns:
                    combined = combined.drop_duplicates(subset=[key_field], keep='last')
                elif time_field in combined.columns:
                    combined = combined.drop_duplicates(subset=[time_field], keep='last')
                
                if time_field in combined.columns:
                    combined[time_field] = pd.to_numeric(combined[time_field], errors='coerce')
                    combined = combined.sort_values(time_field)
                    combined['datetime'] = pd.to_datetime(combined[time_field], unit='s').dt.strftime('%Y-%m-%d %H:00')
                
                combined.to_csv(filepath, index=False, encoding='utf-8')
                print(f"增量更新: 原有{len(existing_df)}条 + 新增{len(new_data)}条 = 共{len(combined)}条")
                return combined
        
        # 无现有数据，直接保存
        if time_field in new_data.columns:
            new_data['datetime'] = pd.to_datetime(new_data[time_field], unit='s').dt.strftime('%Y-%m-%d %H:00')
        new_data.to_csv(filepath, index=False, encoding='utf-8')
        print(f"保存新数据: {len(new_data)}条")
        return new_data

    def fetch_swaps(self):
        """获取Swap数据 - 支持增量更新"""
        existing_file = raw_data_path + '/swaps' + self.pool + '.csv'
        
        start_ts, end_ts, status = self._get_incremental_params(
            existing_file, 'timestamp', self.start, self.end
        )
        
        if start_ts is None:
            print(f"Swaps: {status}")
            return pd.read_csv(existing_file)
        
        print(f"Swaps: {status}, 范围: {start_ts} - {end_ts}")
        
        if self.is_v4 or self.is_v3:
            tpl = """{ swaps( first:1000 skip:{skip} where:{ pool:"__POOL__" timestamp_gte:__START__ timestamp_lte:__END__ } orderBy:timestamp orderDirection:asc ){ timestamp amount0 amount1 sender } }"""
        else:
            tpl = """{ swaps( first:1000 skip:{skip} where:{ pair:"__POOL__" timestamp_gte:__START__ timestamp_lte:__END__ } orderBy:timestamp orderDirection:asc ){ timestamp amount0In amount1In amount0Out amount1Out sender } }"""
        
        tpl = tpl.replace("__POOL__", self.pool)
        tpl = tpl.replace("__START__", str(start_ts))
        tpl = tpl.replace("__END__", str(end_ts))
        
        data = self.client.paged_query(tpl, "swaps")
        
        if data and len(data) > 0:
            print(f"Swaps fetched: {len(data)} records")
            new_data = pd.DataFrame(data)
            return self._save_incremental_data(existing_file, new_data, 'timestamp')
        else:
            print("Warning: No swaps data fetched, using existing file if available")
            if os.path.exists(existing_file):
                data = pd.read_csv(existing_file)
                print(f"Loaded existing swaps: {len(data)} records")
                return data
            return pd.DataFrame()

    def fetch_pool_hours(self):
        """获取池小时数据 - 支持增量更新"""
        existing_file = raw_data_path + '/poolHourDatas' + self.pool + '.csv'
        
        start_ts, end_ts, status = self._get_incremental_params(
            existing_file, 'periodStartUnix' if (self.is_v3 or self.is_v4) else 'hourStartUnix', self.start, self.end
        )
        
        if start_ts is None:
            print(f"Pool hours: {status}")
            return pd.read_csv(existing_file)
        
        print(f"Pool hours: {status}, 范围: {start_ts} - {end_ts}")
        
        if self.is_v4 or self.is_v3:
            tpl = """{ poolHourDatas( first:1000 skip:{skip} where:{ pool:"__POOL__" periodStartUnix_gte:__START__ periodStartUnix_lte:__END__ } orderBy:periodStartUnix orderDirection:asc ){ periodStartUnix liquidity token0Price token1Price volumeUSD tvlUSD } }"""
            time_field = 'periodStartUnix'
        else:
            tpl = """{ pairHourDatas( first:1000 skip:{skip} where:{ pair:"__POOL__" hourStartUnix_gte:__START__ hourStartUnix_lte:__END__ } orderBy:hourStartUnix orderDirection:asc ){ hourStartUnix reserve0 reserve1 } }"""
            time_field = 'hourStartUnix'
        
        tpl = tpl.replace("__POOL__", self.pool)
        tpl = tpl.replace("__START__", str(start_ts))
        tpl = tpl.replace("__END__", str(end_ts))
        
        entity = "poolHourDatas" if (self.is_v3 or self.is_v4) else "pairHourDatas"
        data = self.client.paged_query(tpl, entity)
        
        if data and len(data) > 0:
            print(f"Pool hours fetched: {len(data)} records")
            new_data = pd.DataFrame(data)
            return self._save_incremental_data(existing_file, new_data, time_field)
        else:
            print("Warning: No pool hours data fetched, using existing file if available")
            if os.path.exists(existing_file):
                data = pd.read_csv(existing_file)
                print(f"Loaded existing pool hours: {len(data)} records")
                return data
            return pd.DataFrame()

    def fetch_prices(self):
        """获取价格数据 - 支持增量更新"""
        import requests
        
        token_suffix = self.token_id[:10] if self.token_id else 'unknown'
        token_name = getattr(self, 'token_name', 'unknown')
        existing_file = raw_data_path + '/prices_' + token_name + '_' + token_suffix + '.csv'
        
        start_ts, end_ts, status = self._get_incremental_params(
            existing_file, 'date', self.start, self.end
        )
        
        if start_ts is None:
            print(f"Prices: {status}")
            return pd.read_csv(existing_file)
        
        print(f"Prices: {status}, 范围: {start_ts} - {end_ts}")
        
        all_data = []
        skip = 0
        batch_size = 1000
        
        url = self.client.url
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.client.api_key}"
        }
        
        query_template = """
{{
    tokenHourDatas(
        first: {batch_size}
        skip: {skip}
        where: {{
            token: "{token_id}"
            date_gte: {start}
            date_lte: {end}
        }}
        orderBy: date
        orderDirection: asc
    ) {{
        date
        priceUSD
    }}
}}
"""
        
        print(f"Fetching prices for token {self.token_id[:10]}... from {start_ts} to {end_ts}")
        
        while True:
            query = query_template.format(
                batch_size=batch_size,
                skip=skip,
                token_id=self.token_id,
                start=start_ts,
                end=end_ts
            )
            
            try:
                response = requests.post(url, headers=headers, json={"query": query}, timeout=60)
                
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
                print(f"  Fetched page at skip={skip}: +{len(token_data)} records (total: {len(all_data)})")
                skip += len(token_data)
                
                if len(token_data) < batch_size:
                    break
                
                time.sleep(0.2)
                
            except Exception as e:
                print(f"TheGraph error: {e}")
                break
        
        if len(all_data) > 0:
            print(f"Prices fetched: {len(all_data)} records")
            new_data = pd.DataFrame(all_data)
            
            # 过滤掉无效价格（0或null）
            original_len = len(new_data)
            new_data = new_data[new_data['priceUSD'].notna() & (new_data['priceUSD'] != '0') & (new_data['priceUSD'] != 0)]
            new_data['priceUSD'] = pd.to_numeric(new_data['priceUSD'], errors='coerce')
            new_data = new_data[new_data['priceUSD'] > 0]
            
            if original_len > len(new_data):
                print(f"  Filtered {original_len - len(new_data)} invalid price records (0 or null)")
            
            return self._save_incremental_data(existing_file, new_data, 'date')
        else:
            print("Warning: No prices data fetched, using existing file if available")
            if os.path.exists(existing_file):
                data = pd.read_csv(existing_file)
                print(f"Loaded existing prices: {len(data)} records")
                return data
            return pd.DataFrame()

    def fetch_all_for_coin(self, coin_config):
        """
        获取单个币种的所有数据
        
        Args:
            coin_config: 币种配置字典
        """
        self.set_coin_config(coin_config)
        
        results = {}
        
        print(f"\n=== 开始获取 {coin_config.get('name')} 数据 ===")
        
        # 获取价格数据
        print("\n[1/3] 获取价格数据...")
        prices = self.fetch_prices()
        if prices is not None and len(prices) > 0:
            results['prices'] = prices
            print(f"价格数据: {len(prices)} 条")
        
        # 获取Swap数据
        print("\n[2/3] 获取Swap数据...")
        swaps = self.fetch_swaps()
        if swaps is not None and len(swaps) > 0:
            results['swaps'] = swaps
            print(f"Swap数据: {len(swaps)} 条")
        
        # 获取池小时数据
        print("\n[3/3] 获取池小时数据...")
        pool_hours = self.fetch_pool_hours()
        if pool_hours is not None and len(pool_hours) > 0:
            results['pool_hours'] = pool_hours
            print(f"池小时数据: {len(pool_hours)} 条")
        
        print(f"\n=== {coin_config.get('name')} 数据获取完成 ===")
        return results

    def fetch_all_coins(self):
        """
        获取配置中所有稳定币的数据
        """
        all_results = {}
        
        for coin in self.stablecoins:
            try:
                results = self.fetch_all_for_coin(coin)
                all_results[coin['name']] = results
                time.sleep(1)  # 避免请求过快
            except Exception as e:
                print(f"获取 {coin.get('name')} 数据失败: {e}")
                continue
        
        return all_results
