"""Reusable CCXT pagination for one-minute spot candles."""

import ccxt


MINUTE_MS = 60_000


def make_exchange(exchange_id, proxies=None, *, spot_only=False):
    options = {"enableRateLimit": True, "timeout": 15_000}
    if proxies:
        options["proxies"] = proxies
    if exchange_id == "okx" and spot_only:
        options["options"] = {"fetchMarkets": {"types": ["spot"]}}
    return getattr(ccxt, exchange_id)(options)


def iter_minute_rows(exchange, symbol, start_ms, end_ms, *, confirmed=False):
    """Yield (opening_ms, OHLCV values) in the half-open time range.

    The caller sets the cutoff for closed candles. On OKX, confirmed mode
    additionally checks the exchange's close flag, which unified CCXT drops.
    """
    if confirmed:
        if exchange.id != "okx":
            raise ValueError("目前只支持 OKX 的收盘状态校验")
        yield from _iter_okx_confirmed(exchange, symbol, start_ms, end_ms)
        return

    cursor = start_ms
    while cursor < end_ms:
        page = exchange.fetch_ohlcv(symbol, timeframe="1m", since=cursor, limit=300)
        if not page:
            return
        for row in sorted(page, key=lambda row: row[0]):
            if cursor <= row[0] < end_ms:
                yield row[0], row[1:6]
        last = max(row[0] for row in page)
        if last < cursor:
            return
        cursor = last + MINUTE_MS


def _iter_okx_confirmed(exchange, symbol, start_ms, end_ms):
    exchange.load_markets()
    market = exchange.market(symbol)
    base, quote = symbol.split("/")
    if not market.get("spot") or (market.get("base"), market.get("quote")) != (base, quote):
        raise ValueError(f"OKX 现货交易对不匹配: {symbol}")

    # With at most 100 possible opens per window, requesting the preceding
    # 100 rows covers it even if the exchange skips minutes with no data.
    page_minutes = 100
    step = page_minutes * MINUTE_MS
    recent_cutoff = exchange.milliseconds() - 20 * 60 * MINUTE_MS
    for first in range(start_ms, end_ms, step):
        stop = min(first + step, end_ms)
        fetch_page = (exchange.public_get_market_candles if first >= recent_cutoff
                      else exchange.public_get_market_history_candles)
        response = fetch_page({
            "instId": market["id"], "bar": "1m", "after": str(stop - 1), "limit": str(page_minutes),
        })
        if response.get("code") != "0" or not isinstance(response.get("data"), list):
            raise ValueError("OKX 返回了无效的 K 线响应")
        rows = {}
        for row in response["data"]:
            if not isinstance(row, list) or len(row) != 9 or row[8] not in ("0", "1"):
                raise ValueError("OKX 返回了无效的 K 线格式或收盘状态")
            if not isinstance(row[0], str) or not row[0].isascii() or not row[0].isdigit():
                raise ValueError("OKX 返回了无效的 K 线开盘时间")
            stamp = int(row[0])
            if first <= stamp < stop and row[8] == "1":
                values = row[1:6]
                if stamp in rows and rows[stamp] != values:
                    raise ValueError(f"OKX 返回了相同开盘时间的冲突记录: {stamp}")
                rows[stamp] = values
        yield from sorted(rows.items())
