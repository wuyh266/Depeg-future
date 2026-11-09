import pandas as pd
import numpy as np


def compute_markout(swaps, prices, h):
    """原有的小时级markout计算"""
    if 'amount0In' in swaps.columns and 'amount0Out' in swaps.columns:
        price_map = {}

        for _, r in prices.iterrows():
            if 'token' in r and isinstance(r.get('token'), dict):
                price_map[(r['token']["symbol"], r["hour"])] = float(r["priceUSD"])
            else:
                price_map[("USD", r["hour"])] = float(r["priceUSD"])

        out = []

        for _, s in swaps.iterrows():
            t0 = s["hour"]
            t1 = t0 + h

            try:
                p0 = price_map.get(("USDC", t1)) or price_map.get(("USD", t1), 1.0)
                p1 = price_map.get(("USDT", t1)) or price_map.get(("USD", t1), 1.0)
            except:
                continue

            y0 = float(s.amount0Out) if 'amount0Out' in s else 0
            y1 = float(s.amount1In) if 'amount1In' in s else 0

            n = y0 * p0 - y1 * p1
            out.append([t0, n])

        df = pd.DataFrame(out, columns=["hour","markout"])
        return df.groupby("hour")["markout"].mean().reset_index()
    elif 'amount0' in swaps.columns:
        if 'volumeUSD' in prices.columns or (len(prices.columns) > 0 and 'priceUSD' in prices.columns):
            out = []
            for _, s in swaps.iterrows():
                t0 = s["hour"]
                amount = float(s.amount0) if 'amount0' in s else 0
                out.append([t0, amount])
            
            df = pd.DataFrame(out, columns=["hour","markout"])
            return df.groupby("hour")["markout"].sum().reset_index()
    
    return pd.DataFrame(columns=["hour","markout"])


def compute_markout_5min(swaps, pool_hour_data=None, h=300, anchor_price=1.0):
    """
    计算5分钟markout及价格统计指标
    
    Args:
        swaps: 交易数据
            V2格式: timestamp, amount0In, amount0Out, amount1In, amount1Out
            V3格式: timestamp, amount0, amount1
        pool_hour_data: 可选，池小时数据，包含token0Price, token1Price (V3格式用)
        h: 时间窗口(秒)，默认300秒(5分钟)
        anchor_price: 稳定币的锚定价格（默认1.0）
    
    Returns:
        DataFrame with columns:
        - window: 5分钟窗口时间戳
        - markout: 5分钟markout (PnL)
        - price_mean: 价格均值
        - price_std: 价格标准差
        - price_var: 价格方差
        - price_max: 最高价
        - price_min: 最低价
        - price_range: 极差
        - price_skew: 偏度
        - price_kurt: 峰度
        - price_pct_change: 涨跌幅
        - vwap: 成交量加权价格
        - trade_count: 交易次数
        - total_volume: 总成交量(amount0)
    """
    if swaps is None or len(swaps) == 0:
        return pd.DataFrame(columns=[
            'window', 'markout', 'price_mean', 'price_std', 'price_var',
            'price_max', 'price_min', 'price_range', 'price_skew', 'price_kurt',
            'price_pct_change', 'vwap', 'trade_count', 'total_volume'
        ])
    
    swaps = swaps.copy()
    
    if 'timestamp' not in swaps.columns:
        return pd.DataFrame(columns=[
            'window', 'markout', 'price_mean', 'price_std', 'price_var',
            'price_max', 'price_min', 'price_range', 'price_skew', 'price_kurt',
            'price_pct_change', 'vwap', 'trade_count', 'total_volume'
        ])
    
    swaps['timestamp'] = pd.to_numeric(swaps['timestamp'], errors='coerce')
    swaps = swaps.dropna(subset=['timestamp'])
    
    swaps['window'] = (swaps['timestamp'] // h) * h
    
    is_v2 = 'amount0In' in swaps.columns and 'amount0Out' in swaps.columns
    is_v3 = 'amount0' in swaps.columns and 'amount1' in swaps.columns
    
    if not is_v2 and not is_v3:
        return pd.DataFrame(columns=[
            'window', 'markout', 'price_mean', 'price_std', 'price_var',
            'price_max', 'price_min', 'price_range', 'price_skew', 'price_kurt',
            'price_pct_change', 'vwap', 'trade_count', 'total_volume'
        ])
    
    if is_v2:
        def calc_price(row):
            amount0_in = float(row.get('amount0In', 0)) if pd.notna(row.get('amount0In')) else 0
            amount0_out = float(row.get('amount0Out', 0)) if pd.notna(row.get('amount0Out')) else 0
            amount1_in = float(row.get('amount1In', 0)) if pd.notna(row.get('amount1In')) else 0
            amount1_out = float(row.get('amount1Out', 0)) if pd.notna(row.get('amount1Out')) else 0
            
            amount0 = amount0_in + amount0_out
            amount1 = amount1_in + amount1_out
            
            if amount0 > 0:
                return amount1 / amount0
            return np.nan
        
        swaps['price'] = swaps.apply(calc_price, axis=1)
        
        def calc_markout_row(row):
            amount0_out = float(row.get('amount0Out', 0)) if pd.notna(row.get('amount0Out')) else 0
            amount1_in = float(row.get('amount1In', 0)) if pd.notna(row.get('amount1In')) else 0
            price = row.get('price', 1.0)
            if pd.isna(price) or price <= 0:
                price = 1.0
            return amount0_out * price - amount1_in
        
        swaps['markout_row'] = swaps.apply(calc_markout_row, axis=1)
        
        swaps['amount0_in'] = pd.to_numeric(swaps['amount0In'], errors='coerce').fillna(0)
        swaps['amount0_out'] = pd.to_numeric(swaps['amount0Out'], errors='coerce').fillna(0)
        swaps['volume'] = swaps['amount0_in'] + swaps['amount0_out']
    
    elif is_v3:
        swaps['amount0'] = pd.to_numeric(swaps['amount0'], errors='coerce').fillna(0)
        swaps['amount1'] = pd.to_numeric(swaps['amount1'], errors='coerce').fillna(0)
        
        swaps['volume'] = swaps['amount0'].abs()
        
        swaps['markout_row'] = swaps['amount1'] * anchor_price
        
        if pool_hour_data is not None and len(pool_hour_data) > 0:
            price_map = {}
            for _, r in pool_hour_data.iterrows():
                hour = r.get('periodStartUnix') or r.get('hourStartUnix')
                if hour:
                    token1_price = float(r.get('token1Price', 1.0))
                    if token1_price > 0:
                        usd_price = token1_price
                    else:
                        usd_price = anchor_price
                    price_map[int(hour)] = usd_price
            
            swaps['hour'] = (swaps['window'] // 3600) * 3600
            swaps['pool_price'] = swaps['hour'].map(price_map)
            swaps['price'] = swaps['pool_price'].fillna(anchor_price)
        else:
            swaps['price'] = anchor_price
    
    grouped = swaps.groupby('window')
    
    result = pd.DataFrame()
    result['window'] = grouped.size().index
    
    result['markout'] = grouped['markout_row'].sum().values
    result['price_mean'] = grouped['price'].mean().values
    result['price_std'] = grouped['price'].std().values
    result['price_var'] = grouped['price'].var().values
    result['price_max'] = grouped['price'].max().values
    result['price_min'] = grouped['price'].min().values
    result['price_range'] = result['price_max'] - result['price_min']
    result['price_skew'] = grouped['price'].skew().values
    
    def kurt(x):
        if len(x) < 4:
            return np.nan
        return x.kurtosis()
    result['price_kurt'] = grouped['price'].apply(kurt).values
    
    result['price_pct_change'] = result['price_mean'].pct_change().fillna(0).values
    
    def vwap_calc(group):
        valid = group['price'].notna() & (group['volume'] > 0)
        if valid.sum() == 0:
            return np.nan
        return (group.loc[valid, 'price'] * group.loc[valid, 'volume']).sum() / group.loc[valid, 'volume'].sum()
    result['vwap'] = grouped.apply(vwap_calc).values
    
    result['trade_count'] = grouped.size().values
    result['total_volume'] = grouped['volume'].sum().values
    
    result = result.fillna(0)
    
    return result
