import yaml
import pandas as pd
import numpy as np
import os
import sys
import argparse
from datetime import datetime, timedelta

# 添加路径以支持从不同目录运行
if __name__ == '__main__':
    if os.path.dirname(__file__):
        sys.path.insert(0, os.path.dirname(__file__))

from fetch import Fetcher, get_token_pools, get_token_pools_at_time
from align import align_hour
from entropyV2 import compute_entropy_v2
from netflow import compute_netflow
from markout import compute_markout, compute_markout_5min
from preprocess import logdiff
from bocd_extended import ExtendedBOCD
from fetch_ohlcv import OHLCVFetcher
try:
    from tools import save_csv_with_backup
except ImportError:
    from src.tools import save_csv_with_backup


def compute_pool_bocd_features(pool_metrics_df, hazard_rate=200):
    """
    对池指标运行 BOCD，计算偏离特征 (增量计算)
    
    Args:
        pool_metrics_df: 包含池指标的 DataFrame
        hazard_rate: BOCD 风险率
    
    Returns:
        DataFrame: 添加了 BOCD 特征的 DataFrame
    """
    if pool_metrics_df is None or len(pool_metrics_df) < 10:
        return pool_metrics_df
    
    indicator_cols = ['entropy', 'netflow', 'markout', 'volume']
    
    for col in indicator_cols:
        if col not in pool_metrics_df.columns:
            continue
        
        values = pool_metrics_df[col].fillna(0).values
        
        try:
            log_returns = np.diff(np.log(values + 1e-10))
            log_returns = log_returns[~np.isnan(log_returns)]
            log_returns = log_returns[~np.isinf(log_returns)]
            
            if len(log_returns) < 10:
                continue
            
            n = len(pool_metrics_df)
            deviations = np.zeros(n)
            cp_probs = np.zeros(n)
            surprises = np.zeros(n)
            
            bocd = ExtendedBOCD(hazard_rate=hazard_rate)
            
            for i in range(len(log_returns)):
                bocd.update(log_returns[i])
                features = bocd.get_latest_features()
                deviations[i + 1] = features.get('deviation', 0)
                cp_probs[i + 1] = features.get('cp_prob', 0)
                surprises[i + 1] = features.get('surprise', 0)
            
            pool_metrics_df[f'{col}_deviation'] = deviations
            pool_metrics_df[f'{col}_cp_prob'] = cp_probs
            pool_metrics_df[f'{col}_surprise'] = surprises
            
        except Exception as e:
            print(f"  Warning: BOCD failed for {col}: {e}")
            continue
    
    return pool_metrics_df
    
    indicator_cols = ['entropy', 'netflow', 'markout', 'volume']
    for col in indicator_cols:
        if col not in pool_metrics_df.columns:
            continue
        
        values = pool_metrics_df[col].fillna(0).values
        
        # 计算 log returns
        try:
            returns = pd.Series(values).pct_change().dropna().values
            returns = returns[~pd.isna(returns)]
            returns = returns[~np.isinf(returns)]
            
            if len(returns) < 10:
                continue
            
            # 运行 BOCD
            bocd = ExtendedBOCD(hazard_rate=hazard_rate)
            for x in returns:
                bocd.update(x)
            
            features = bocd.get_latest_features()
            
            # 添加偏离特征
            pool_metrics_df[f'{col}_deviation'] = features.get('deviation', 0)
            pool_metrics_df[f'{col}_cp_prob'] = features.get('cp_prob', 0)
            pool_metrics_df[f'{col}_surprise'] = features.get('surprise', 0)
            
        except Exception as e:
            print(f"  Warning: BOCD failed for {col}: {e}")
            continue
    
    return pool_metrics_df


def get_time_range(coin_config, days=None):
    """
    计算数据获取的时间范围
    
    Args:
        coin_config: 币种配置字典，包含 start 和 end
        days: 如果指定，只获取最近 N 天的数据
    
    Returns:
        tuple: (start_ts, end_ts, description)
    """
    from src.tools import parse_datetime_to_timestamp
    
    if days is not None:
        # 只获取最近 N 天
        end_ts = int(datetime.now().timestamp())
        start_ts = end_ts - days * 24 * 3600
        return start_ts, end_ts, f"最近 {days} 天"
    
    # 使用配置文件的时间范围
    start_str = coin_config.get('start', '')
    end_str = coin_config.get('end', '')
    
    if start_str and end_str:
        start_ts = parse_datetime_to_timestamp(start_str)
        end_ts = parse_datetime_to_timestamp(end_str)
        return start_ts, end_ts, f"{start_str} 至 {end_str}"
    
    # 默认：最近 90 天
    end_ts = int(datetime.now().timestamp())
    start_ts = end_ts - 90 * 24 * 3600
    return start_ts, end_ts, "最近 90 天"


def process_single_pool(pool, coin_name, coin_config, start_ts, end_ts, pool_index, total_pools):
    """
    处理单个池的数据获取和计算
    
    Args:
        pool: 池信息字典
        coin_name: 币种名称
        coin_config: 币种配置
        start_ts: 开始时间戳
        end_ts: 结束时间戳
        pool_index: 池索引
        total_pools: 总池数
    
    Returns:
        dict: 处理结果
    """
    cfg = yaml.safe_load(open("config.yaml", encoding='utf-8'))
    api_key = cfg.get("api_keys", {}).get("the_graph", cfg.get("THE_GRAPH_API_KEY", ""))
    
    pool_address = pool['address']
    pool_factory = pool['factory']
    pool_name = f"{pool['token0']}/{pool['token1']}"
    
    print(f"\n  [{pool_index+1}/{total_pools}] 处理池: {pool_name} (TVL: ${pool.get('tvl', 0):,.0f}) [{pool_factory}]")
    
    # 根据版本选择subgraph
    if pool_factory == "v3":
        subgraph = cfg.get("subgraph_uniswap_v3", "https://gateway.thegraph.com/api/subgraphs/id/5zvR82QoaXYFyDEKLZ9t6v9adgnptxYpKpSbxtgVENFV")
    elif pool_factory == "v4":
        subgraph = cfg.get("subgraph_uniswap_v4", "https://gateway.thegraph.com/api/subgraphs/id/DiYPVdygkfjDWhbxGSqAQxwBKmfKnkWQojqeM2rkLb3G")
    else:
        subgraph = cfg.get("subgraph", "https://gateway.thegraph.com/api/subgraphs/id/EYCKATKGBKLWvSfwvBjzfCBmGwYNdVkduYXVivCsLRFu")
    
    # 创建池专属目录
    pool_dir = f"data/{coin_name.lower()}/pool_{pool_index+1}_{pool_address[:8]}"
    os.makedirs(pool_dir, exist_ok=True)
    
    # 获取数据
    f = Fetcher(cfg, subgraph)
    f.pool = pool_address
    f.token_id = coin_config.get('token_id', '')
    f.token_name = coin_name
    f.start = start_ts
    f.end = end_ts
    
    # 单独获取这个池的数据
    print(f"    获取池数据...")
    swaps = f.fetch_swaps()
    pools_data = f.fetch_pool_hours()
    swaps_file = f"{pool_dir}/swaps.csv"
    pool_hours_file = f"{pool_dir}/pool_hours.csv"
    
    # 保存原始数据
    if len(swaps) > 0:
        backup_path, swaps_file = save_csv_with_backup(swaps, swaps_file, index=False)
        if backup_path:
            print(f"    Existing swaps backed up to: {backup_path}")
        print(f"    Swaps: {len(swaps)} records")
    if len(pools_data) > 0:
        pools_data['version'] = pool_factory
        backup_path, pool_hours_file = save_csv_with_backup(pools_data, pool_hours_file, index=False)
        if backup_path:
            print(f"    Existing pool hours backed up to: {backup_path}")
        print(f"    Pool hours: {len(pools_data)} records")
    
    return {
        'pool_dir': pool_dir,
        'swaps_file': swaps_file,
        'pool_hours_file': pool_hours_file,
        'swaps': swaps,
        'pools_data': pools_data,
        'pool_factory': pool_factory
    }


def process_single_coin(coin_name, coin_config, days=None):
    """
    处理单个币种的所有池数据
    
    Args:
        coin_name: 币种名称 (如 USDC)
        coin_config: 币种配置字典
        days: 只获取最近 N 天数据 (可选)
    
    Returns:
        dict: 处理结果
    """
    cfg = yaml.safe_load(open("config.yaml", encoding='utf-8'))
    api_key = cfg.get("api_keys", {}).get("the_graph", cfg.get("THE_GRAPH_API_KEY", ""))
    token_address = coin_config.get('token_id', '')
    
    print(f"\n{'='*60}")
    print(f"处理币种: {coin_name}")
    print(f"代币地址: {token_address}")
    
    # 获取时间范围
    start_ts, end_ts, time_desc = get_time_range(coin_config, days)
    print(f"时间范围: {time_desc}")
    
    # Step 1: 查询池地址
    print(f"\n[1/5] 查询 {coin_name} 的池地址...")
    
    # 优先使用本地已有的池列表
    coin_dir = f"data/{coin_name.lower()}"
    local_pools_file = f"{coin_dir}/pools_list.csv"
    pools = []
    
    if os.path.exists(local_pools_file):
        local_pools_df = pd.read_csv(local_pools_file)
        if len(local_pools_df) > 0:
            pools = local_pools_df.to_dict('records')
            print(f"  使用本地池列表: {len(pools)} 个池")
    
    # 如果没有本地数据，查询 API
    if not pools:
        # 检查是否配置了历史TVL参考时间
        tvl_ref_time = coin_config.get('tvl_reference_time', '')
        if tvl_ref_time:
            # 解析时间并查询历史时间点的TVL
            from src.tools import parse_datetime_to_timestamp
            tvl_ts = parse_datetime_to_timestamp(tvl_ref_time)
            print(f"  使用历史TVL参考时间: {tvl_ref_time} (timestamp: {tvl_ts})")
            pools = get_token_pools_at_time(token_address, api_key, tvl_ts, min_tvl=1000, top_n=3)
            if pools:
                print(f"  按历史TVL排序找到 {len(pools)} 个池")
        else:
            pools = get_token_pools(token_address, api_key, min_tvl=10000, top_n=3)
    
    if not pools:
        print(f"  Warning: 未找到 {coin_name} 的有效池")
        return {'status': 'error', 'message': 'No pools found'}
    
    print(f"  找到 {len(pools)} 个池:")
    for i, p in enumerate(pools):
        tvl_info = f"TVL: ${p['tvl']:,.0f}"
        if 'historical_tvl' in p:
            tvl_info = f"历史TVL: ${p['tvl']:,.0f} (参考时间: {p.get('tvl_timestamp', 'N/A')})"
        print(f"    {i+1}. {p['token0']}/{p['token1']} ({tvl_info}) [{p['factory']}]")
    
    # 保存池列表
    pools_df = pd.DataFrame(pools)
    os.makedirs(coin_dir, exist_ok=True)
    backup_path, pools_list_file = save_csv_with_backup(pools_df, f"{coin_dir}/pools_list.csv", index=False)
    if backup_path:
        print(f"  Existing pools list backed up to: {backup_path}")
    
    # Step 2-5: 遍历所有池，获取并处理数据
    print(f"\n[2/5] 获取所有池的数据...")
    
    # 获取价格数据（所有池共享）
    token_name = coin_name
    token_suffix = token_address[:10] if token_address else 'unknown'
    prices_file = f"data/raw data/prices_{token_name}_{token_suffix}.csv"
    prices = pd.DataFrame()
    
    # 检查是否有现有价格数据
    if os.path.exists(prices_file):
        prices = pd.read_csv(prices_file)
        print(f"  加载价格数据: {len(prices)} records")
    else:
        # 获取价格数据
        print(f"  获取价格数据...")
        # 使用V3 subgraph获取价格
        subgraph = cfg.get("subgraph_uniswap_v3", "https://gateway.thegraph.com/api/subgraphs/id/5zvR82QoaXYFyDEKLZ9t6v9adgnptxYpKpSbxtgVENFV")
        f = Fetcher(cfg, subgraph)
        f.token_id = token_address
        f.token_name = coin_name
        f.start = start_ts
        f.end = end_ts
        prices = f.fetch_prices()
        if len(prices) > 0:
            backup_path, prices_file = save_csv_with_backup(prices, prices_file, index=False)
            if backup_path:
                print(f"  Existing prices file backed up to: {backup_path}")
            print(f"  价格数据: {len(prices)} records -> {prices_file}")
    
    all_pool_metrics = []
    
    for pool_idx, pool in enumerate(pools):
        pool_result = process_single_pool(pool, coin_name, coin_config, start_ts, end_ts, pool_idx, len(pools))
        
        # 读取刚获取的池数据
        pool_dir = pool_result['pool_dir']
        swaps_file = pool_result.get('swaps_file', f"{pool_dir}/swaps.csv")
        pool_hours_file = pool_result.get('pool_hours_file', f"{pool_dir}/pool_hours.csv")
        
        if os.path.exists(swaps_file):
            swaps = pd.read_csv(swaps_file)
        else:
            swaps = pd.DataFrame()
        
        if os.path.exists(pool_hours_file):
            pools_data = pd.read_csv(pool_hours_file)
        else:
            pools_data = pd.DataFrame()
        
        pool_factory = pool_result['pool_factory']
        
        # 计算池指标
        time_field = "periodStartUnix" if pool_factory == "v3" else "hourStartUnix"
        
        if len(swaps) > 0 and "timestamp" in swaps.columns:
            swaps = align_hour(swaps, "timestamp")
        else:
            swaps = pd.DataFrame()
        
        if len(pools_data) > 0 and time_field in pools_data.columns:
            pools_data = align_hour(pools_data, time_field)
        else:
            pools_data = pd.DataFrame()
        
        # 计算指标
        pools_data = compute_entropy_v2(pools_data)
        net = compute_netflow(swaps, coin_config.get('anchor', 'USDC')) if len(swaps) > 0 and "hour" in swaps.columns else pd.DataFrame(columns=['hour', 'flow'])
        mark = compute_markout(swaps, prices, cfg.get("markout_window", 3600)) if len(prices) > 0 and "hour" in prices.columns else pd.DataFrame(columns=['hour', 'markout'])
        
        # 计算交易量
        if len(swaps) > 0:
            if 'amount0In' in swaps.columns and 'amount0Out' in swaps.columns:
                swaps['volume'] = swaps['amount0In'].astype(float) + swaps['amount0Out'].astype(float)
            elif 'amount0' in swaps.columns:
                swaps['volume'] = abs(swaps['amount0'].astype(float))
            else:
                swaps['volume'] = 0
            
            volume = swaps.groupby('hour')['volume'].sum().reset_index()
            volume.columns = ['hour', 'volume']
        else:
            volume = pd.DataFrame(columns=['hour', 'volume'])
        
        # 添加 TVL
        if len(pools_data) > 0:
            if 'reserve0' in pools_data.columns and 'reserve1' in pools_data.columns:
                pools_data['tvl'] = pools_data['reserve0'].astype(float) + pools_data['reserve1'].astype(float)
            elif 'tvlUSD' in pools_data.columns:
                pools_data['tvl'] = pools_data['tvlUSD'].astype(float)
        
        # 合并指标
        if len(pools_data) > 0 and "hour" in pools_data.columns:
            pool_metrics = pools_data.merge(net, on="hour", how="left")
            pool_metrics = pool_metrics.merge(mark, on="hour", how="left")
            pool_metrics = pool_metrics.merge(volume, on="hour", how="left")
        else:
            pool_metrics = pd.DataFrame()
        
        # 对数差分
        if len(pool_metrics) > 0 and "entropy" in pool_metrics.columns:
            pool_metrics["entropy_ld"] = logdiff(pool_metrics["entropy"])
            if "flow" in pool_metrics.columns:
                pool_metrics["flow_ld"] = logdiff(pool_metrics["flow"])
        
        # 添加池信息
        if len(pool_metrics) > 0:
            pool_metrics['pool_address'] = pool['address']
            pool_metrics['pool_name'] = f"{pool['token0']}/{pool['token1']}"
            pool_metrics['pool_index'] = pool_idx + 1
        
        # BOCD处理
        if len(pool_metrics) > 0:
            pool_metrics = compute_pool_bocd_features(pool_metrics)
        
        # 添加日期时间
        if len(pool_metrics) > 0 and 'hour' in pool_metrics.columns:
            pool_metrics['datetime'] = pd.to_datetime(pool_metrics['hour'], unit='s').dt.strftime('%Y-%m-%d %H:00')
        
        # 保存单个池的指标
        if len(pool_metrics) > 0:
            pool_metrics_file = f"{pool_dir}/pool_metrics.csv"
            backup_path, pool_metrics_file = save_csv_with_backup(pool_metrics, pool_metrics_file, index=False)
            if backup_path:
                print(f"    Existing pool metrics backed up to: {backup_path}")
            print(f"    池指标: {len(pool_metrics)} records -> {pool_dir}/pool_metrics.csv")
            all_pool_metrics.append(pool_metrics)
    
    # 合并所有池的指标
    if len(all_pool_metrics) > 0:
        combined_metrics = pd.concat(all_pool_metrics, ignore_index=True)
        
        # 保存完整数据
        full_file = f"{coin_dir}/pool_metrics_full.csv"
        backup_path, full_file = save_csv_with_backup(combined_metrics, full_file, index=False)
        if backup_path:
            print(f"\n  Existing combined metrics backed up to: {backup_path}")
        print(f"\n  所有池完整数据: {len(combined_metrics)} records -> {full_file}")
        
        # 保存5分钟markout（只对第一个池）
        first_pool_dir = all_pool_metrics[0].get('pool_address', '')
        mark_5min_file = f"{coin_dir}/markout_5min.csv"
        if os.path.exists(f"{coin_dir}/pool_1_{all_pool_metrics[0].get('pool_address', '')[:8]}/swaps.csv"):
            print(f"  5分钟markout: 已保存到各池目录")
    else:
        print(f"\n  Warning: 没有获取到任何池的指标数据")
    
    print(f"\n{'='*60}")
    print(f"{coin_name} 处理完成! 共处理 {len(pools)} 个池")
    print(f"{'='*60}")
    
    return {
        'status': 'success',
        'coin': coin_name,
        'pools_count': len(pools),
        'records': len(combined_metrics) if len(all_pool_metrics) > 0 else 0
    }


def main():
    parser = argparse.ArgumentParser(description='获取稳定币池数据')
    parser.add_argument('--coin', type=str, default=None, 
                        help='指定币种 (如 USDC, USDT, DAI, UST)')
    parser.add_argument('--days', type=int, default=None,
                        help='只获取最近 N 天数据 (用于持续监控)')
    parser.add_argument('--all', action='store_true',
                        help='处理配置中所有币种')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='配置文件路径')
    parser.add_argument('--ohlcv', action='store_true',
                        help='获取分钟级OHLCV数据')
    parser.add_argument('--curve', action='store_true',
                        help='获取Curve池数据')
    parser.add_argument('--monitor', action='store_true',
                        help='实时监控模式')
    
    args = parser.parse_args()
    
    # 加载配置
    cfg = yaml.safe_load(open(args.config, encoding='utf-8'))
    stablecoins = cfg.get("stablecoins", [])
    
    # 确定要处理的币种
    coins_to_process = []
    
    if args.all:
        coins_to_process = stablecoins
    elif args.coin:
        for coin in stablecoins:
            if coin.get('name', '').upper() == args.coin.upper():
                coins_to_process = [coin]
                break
        if not coins_to_process:
            print(f"Error: 未在配置中找到币种 {args.coin}")
            return
    else:
        print("Error: 请指定 --coin <币种> 或 --all")
        parser.print_help()
        return
    
    # 如果是 OHLCV 模式
    if args.ohlcv:
        print(f"\n{'='*60}")
        print("获取分钟级 OHLCV 数据")
        print(f"{'='*60}")
        
        ohlcv_fetcher = OHLCVFetcher()
        
        for coin in coins_to_process:
            coin_name = coin.get('name', 'UNKNOWN')
            start_date = coin.get('start', '2023-01-01')
            end_date = coin.get('end', '2024-01-01')
            
            print(f"\n处理 {coin_name}: {start_date} ~ {end_date}")
            ohlcv_fetcher.fetch(coin_name, start_date, end_date)
        
        print(f"\n{'='*60}")
        print("OHLCV 数据获取完成!")
        print(f"{'='*60}")
        return
    
    # 如果是 Curve 模式
    if args.curve:
        from fetch_curve import CurveFetcher, CurveWebsocketMonitor
        import time
        
        print(f"\n{'='*60}")
        print("获取 Curve 池数据")
        print(f"{'='*60}")
        
        cfg = yaml.safe_load(open(args.config, encoding='utf-8'))
        curve_pools = cfg.get("curve_pools", {})
        
        if args.monitor:
            print("实时监控模式")
            monitors = []
            
            for coin in coins_to_process:
                coin_name = coin.get('name', 'UNKNOWN')
                if coin_name in curve_pools:
                    pool_info = curve_pools[coin_name]
                    pool_addr = pool_info.get('pool_address', '')
                    monitor = CurveWebsocketMonitor(coin_name, pool_addr)
                    monitor.start()
                    monitors.append(monitor)
            
            print(f"\n实时监控中... 按 Ctrl+C 退出")
            try:
                while True:
                    time.sleep(60)
            except KeyboardInterrupt:
                print("\n停止监控...")
                for m in monitors:
                    m.stop()
            return
        
        curve_fetcher = CurveFetcher()

        for coin in coins_to_process:
            coin_name = coin.get('name', 'UNKNOWN')
            if coin_name in curve_pools:
                pool_info = curve_pools[coin_name]
                pool_addr = pool_info.get('pool_address', '')
                monitor = CurveWebsocketMonitor(coin_name, pool_addr)
                monitor.start()
                monitors.append(monitor)
            
            print(f"\n实时监控中... 按 Ctrl+C 退出")
            try:
                while True:
                    time.sleep(60)
            except KeyboardInterrupt:
                print("\n停止监控...")
                for m in monitors:
                    m.stop()
            return

        curve_fetcher = CurveFetcher()

        for coin in coins_to_process:
            coin_name = coin.get('name', 'UNKNOWN')
            if coin_name in curve_pools:
                pool_info = curve_pools[coin_name]
                start_date = coin.get('start', pool_info.get('start'))
                end_date = coin.get('end', pool_info.get('end'))
                
                print(f"\n处理 {coin_name}: {start_date} ~ {end_date}")
                curve_fetcher.fetch(coin_name, start_date, end_date)
            else:
                print(f"\n{coin_name} 不在 Curve 池配置中")
        
        print(f"\n{'='*60}")
        print("Curve 数据获取完成!")
        print(f"{'='*60}")
        return
    
    # 处理每个币种
    results = []
    for coin in coins_to_process:
        coin_name = coin.get('name', 'UNKNOWN')
        result = process_single_coin(coin_name, coin, args.days)
        results.append(result)
    
    # 汇总结果
    print(f"\n{'='*60}")
    print("处理汇总")
    print(f"{'='*60}")
    for r in results:
        if r['status'] == 'success':
            print(f"  {r['coin']}: {r['records']} 条记录, 池数: {r['pools_count']}")
        else:
            print(f"  {r.get('coin', 'UNKNOWN')}: {r.get('message', 'Error')}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
