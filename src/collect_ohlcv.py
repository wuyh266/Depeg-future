"""Collect closed OKX USDC/USDT one-minute candles."""

import argparse
import csv
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import logging
import os
from pathlib import Path
import re
import subprocess
from tempfile import NamedTemporaryFile
import time
from uuid import uuid4

import ccxt

if __package__:
    from .ohlcv_pages import MINUTE_MS, iter_minute_rows, make_exchange
else:
    from ohlcv_pages import MINUTE_MS, iter_minute_rows, make_exchange


EXCHANGE, SYMBOL, BASE, QUOTE = "okx", "USDC/USDT", "USDC", "USDT"
ASSET_ID, SOURCE, INTERVAL = "ccxt.usdc-usdt", "ccxt:okx", "1m"
FILE_PREFIX = "okx_usdc-usdt_1m_"
MAX_BATCH_MS = 1_440 * MINUTE_MS
LOOKBACK_MS = 2 * MINUTE_MS
CLOSE_DELAY_SECONDS = 5
COLUMNS = ("timestamp", "datetime", "open", "high", "low", "close", "volume", "exchange")
EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)
LOG = logging.getLogger("ohlcv")


def utc_label(stamp_ms):
    return (EPOCH + timedelta(milliseconds=stamp_ms)).isoformat(timespec="seconds").replace("+00:00", "Z")


def parse_utc(value):
    try:
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
            value += "T00:00:00+00:00"
        stamp = datetime.fromisoformat(value[:-1] + "+00:00" if value.endswith("Z") else value)
        if stamp.tzinfo is None or stamp < EPOCH or stamp.second or stamp.microsecond:
            raise ValueError
        millis = int((stamp - EPOCH).total_seconds()) * 1000
        if millis % MINUTE_MS:
            raise ValueError
        return millis
    except (ValueError, OverflowError):
        raise argparse.ArgumentTypeError("请填写带时区且对齐整分钟的时间，如 2026-11-10T00:00:00Z") from None


def decimal_text(value):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9]{1,20}(?:\.[0-9]{1,40})?", value):
        raise ValueError("价格和成交量必须是非负十进制数字")
    value = value.rstrip("0").rstrip(".") if "." in value else value
    if "." in value and len(value.split(".")[1]) > 18:
        raise ValueError("价格和成交量最多保留 18 位小数")
    return format(Decimal(value), "f")


def fetch_batch(exchange, start_ms, end_ms):
    rows = []
    next_expected = start_ms
    for stamp, values in iter_minute_rows(exchange, SYMBOL, start_ms, end_ms, confirmed=True):
        try:
            if stamp % MINUTE_MS:
                raise ValueError("开盘时间未对齐整分钟")
            cleaned = tuple(decimal_text(value) for value in values)
            op, high, low, close = map(Decimal, cleaned[:4])
            if not (0 < low <= min(op, close) <= max(op, close) <= high):
                raise ValueError("开盘价、最高价、最低价和收盘价不一致")
        except ValueError as error:
            raise ValueError(f"{utc_label(stamp)} 的 K 线校验失败: {error}") from error
        log_gap(next_expected, stamp)
        next_expected = stamp + MINUTE_MS
        rows.append((stamp // 1000, utc_label(stamp), *cleaned, EXCHANGE))
    log_gap(next_expected, end_ms)
    LOG.info("%s %s %s [%s, %s) 收到 %d 根已收盘 K 线",
             EXCHANGE, SYMBOL, INTERVAL, utc_label(start_ms), utc_label(end_ms), len(rows))
    return rows


def log_gap(start_ms, end_ms):
    if start_ms < end_ms:
        LOG.warning("缺少 %d 根 K 线 [%s, %s)，保留缺口，可用 --start/--end 补抓",
                    (end_ms - start_ms) // MINUTE_MS, utc_label(start_ms), utc_label(end_ms))


def sync_directory(directory):
    if os.name == "posix":
        fd = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def save_batch(directory, rows):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{FILE_PREFIX}{rows[0][0]:012d}_{rows[-1][0]:012d}_{uuid4().hex}.csv"
    temporary = None
    try:
        with NamedTemporaryFile(mode="w", encoding="utf-8", newline="",
                                dir=directory, prefix=FILE_PREFIX, suffix=".tmp",
                                delete=False) as output:
            temporary = Path(output.name)
            writer = csv.writer(output)
            writer.writerow(COLUMNS)
            writer.writerows(rows)
            output.flush()
            os.fsync(output.fileno())
        temporary.replace(path)
        sync_directory(directory)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    LOG.info("已保存 %d 根 K 线: %s", len(rows), path)


class Importer:
    """CLI boundary for the next step: a precompiled Go importer."""

    def __init__(self, binary, pending):
        if not binary.is_file():
            raise ValueError(f"找不到 Go 导入工具: {binary}")
        self.binary = str(binary.absolute())
        self.pending = pending

    def latest(self):
        result = subprocess.run(
            [self.binary, "-latest-open-time", "-asset-id", ASSET_ID, "-source", SOURCE, "-interval", INTERVAL],
            check=True, stdout=subprocess.PIPE, text=True, encoding="utf-8", timeout=30,
        ).stdout.strip()
        if not result:
            return None
        try:
            return parse_utc(result)
        except argparse.ArgumentTypeError as error:
            raise ValueError("导入工具返回的最新开盘时间无效") from error

    def flush(self):
        for path in sorted(self.pending.glob(f"{FILE_PREFIX}*.csv")):
            try:
                subprocess.run(
                    [self.binary, "-file", str(path), "-asset-id", ASSET_ID, "-symbol", BASE,
                     "-quote", QUOTE, "-source", "ccxt", "-interval", INTERVAL, "-timezone", "UTC", "-anchor="],
                    check=True, timeout=120,
                )
            except (OSError, subprocess.SubprocessError):
                LOG.error("入库失败，待处理批次已保留: %s", path)
                raise
            path.unlink()
            sync_directory(self.pending)
            LOG.info("入库成功: %s", path.name)


def run_round(exchange, args, importer=None, *, now_ms=None):
    if importer is not None:
        importer.flush()
    now_ms = time.time_ns() // 1_000_000 if now_ms is None else now_ms
    cutoff = now_ms // MINUTE_MS * MINUTE_MS
    fixed = args.start is not None
    if fixed:
        start, end = args.start, min(args.end, cutoff)
    elif importer is None:
        start, end = cutoff - 10 * MINUTE_MS, cutoff
    else:
        latest = importer.latest()
        if latest is None:
            if args.bootstrap_from is None:
                raise ValueError("数据库没有这条行情，请用 --bootstrap-from 指定首次采集起点")
            start = args.bootstrap_from
        else:
            if latest >= cutoff:
                raise ValueError("数据库最新开盘时间尚未收盘，请核对数据库记录和服务器时钟")
            floor = args.bootstrap_from if args.bootstrap_from is not None and args.bootstrap_from <= latest else 0
            start = max(floor, latest - LOOKBACK_MS)
            LOG.info("数据库最新开盘时间: %s", utc_label(latest))
        end = min(cutoff, start + MAX_BATCH_MS)
    if start >= end:
        LOG.info("指定起点还没有已收盘的 K 线: %s", utc_label(start))
        return False

    rows = []
    for first in range(start, end, MAX_BATCH_MS):
        rows = fetch_batch(exchange, first, min(first + MAX_BATCH_MS, end))
        if rows:
            save_batch(importer.pending if importer is not None else args.output_dir, rows)
            if importer is not None:
                importer.flush()
    # If the batch advanced the database and backlog remains, catch up now.
    return bool(importer is not None and not fixed and end < cutoff and rows
                and (latest is None or rows[-1][0] * 1000 > latest))


def main(argv=None):
    parser = argparse.ArgumentParser(description="采集 OKX USDC/USDT 已收盘的 1 分钟 K 线")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--once", action="store_true", help="运行一轮；未指定导入工具时只保存 CSV")
    mode.add_argument("--loop", action="store_true", help="持续采集；需要新版 Go 导入工具")
    parser.add_argument("--start", type=parse_utc, help="固定区间起点（包含），例如 2026-11-09T00:00:00Z")
    parser.add_argument("--end", type=parse_utc, help="固定区间终点（不包含），仅用于 --once")
    parser.add_argument("--bootstrap-from", type=parse_utc, help="数据库没有记录时的首次采集起点")
    parser.add_argument("--output-dir", type=Path, default=Path("data/collector-preview"), help="只抓取时的 CSV 目录")
    parser.add_argument("--pending-dir", type=Path, default=Path("data/collector-pending"), help="待入库 CSV 目录")
    parser.add_argument("--import-bin", type=Path, help="新版 Go 导入工具路径；省略则不连接数据库")
    parser.add_argument("--proxy", help="可选的 HTTP 代理，如 http://127.0.0.1:7890")
    args = parser.parse_args(argv)
    if ((args.start is None) != (args.end is None)
            or (args.start is not None and (args.start >= args.end or args.loop))):
        parser.error("--start 和 --end 需同时给出，起点须早于终点，仅用于 --once")
    if (args.loop or args.bootstrap_from is not None) and args.import_bin is None:
        parser.error("--loop 和 --bootstrap-from 需要新版 Go 导入工具（--import-bin）")
    if args.start is not None and args.bootstrap_from is not None:
        parser.error("指定固定时间区间时不使用 --bootstrap-from")
    if args.output_dir.resolve() == args.pending_dir.resolve():
        parser.error("预览 CSV 和待入库 CSV 必须使用不同的目录")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%Y-%m-%dT%H:%M:%SZ")
    logging.Formatter.converter = time.gmtime
    try:
        importer = Importer(args.import_bin, args.pending_dir.resolve()) if args.import_bin else None
        proxies = {"http": args.proxy, "https": args.proxy} if args.proxy else None
        exchange = make_exchange(EXCHANGE, proxies, spot_only=True)
        if args.once:
            run_round(exchange, args, importer)
            return 0
        delay = 5
        while True:
            try:
                catch_up = run_round(exchange, args, importer)
            except (ccxt.NetworkError, subprocess.SubprocessError) as error:
                LOG.warning("采集或导入暂时失败，%d 秒后重试 (%s)", delay, type(error).__name__)
                time.sleep(delay)
                delay = min(delay * 2, 60)
                continue
            delay = 5
            if not catch_up:
                time.sleep(60 - time.time() % 60 + CLOSE_DELAY_SECONDS)
    except KeyboardInterrupt:
        LOG.info("采集已停止")
        return 0
    except (ccxt.BaseError, ValueError, OSError, subprocess.SubprocessError) as error:
        LOG.error("采集失败: %s", error)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
