import csv
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import ccxt

from src import collect_ohlcv as collector


START = collector.parse_utc("2026-11-07T08:35:00Z")
MINUTE = collector.MINUTE_MS


def candle(minute, *, confirm="1", volume="628.557467", offset_ms=0):
    return [str(START + minute * MINUTE + offset_ms), "1.00106", "1.00106", "1.00105",
            "1.00105", volume, "629", "629", confirm]


class OKXFixture:
    id = "okx"

    def __init__(self, rows, *, now_ms=START + 210 * MINUTE):
        self.rows = rows
        self.now_ms = now_ms
        self.requests = []

    def load_markets(self):
        pass

    def market(self, symbol):
        if symbol != "USDC/USDT":
            raise AssertionError(symbol)
        return {"id": "USDC-USDT", "spot": True, "base": "USDC", "quote": "USDT"}

    def milliseconds(self):
        return self.now_ms

    def public_get_market_candles(self, params):
        self.requests.append(("recent", params))
        return self._page(params)

    def public_get_market_history_candles(self, params):
        self.requests.append(("history", params))
        return self._page(params)

    def _page(self, params):
        assert (params["instId"], params["bar"]) == ("USDC-USDT", "1m")
        rows = sorted((r for r in self.rows if int(r[0]) < int(params["after"])),
                      key=lambda r: int(r[0]), reverse=True)
        return {"code": "0", "data": rows[:int(params["limit"])]}


class CollectorTest(unittest.TestCase):
    def test_fixed_range_keeps_gaps_and_ignores_open_candle(self):
        now = START + 202 * MINUTE
        with TemporaryDirectory() as folder:
            out = Path(folder)
            args = SimpleNamespace(start=START, end=now + MINUTE, output_dir=out)
            exchange = OKXFixture([candle(0), candle(0), candle(99), candle(200),
                                   candle(201, confirm="0"), candle(202)], now_ms=now)
            with self.assertLogs(collector.LOG, level="INFO") as log:
                collector.run_round(exchange, args, now_ms=now)
            self.assertIn("缺少 100 根 K 线", "\n".join(log.output))
            self.assertEqual(len(exchange.requests), 3)
            with next(out.glob("*.csv")).open(encoding="utf-8", newline="") as saved:
                rows = list(csv.DictReader(saved))
            self.assertEqual([int(r["timestamp"]) for r in rows],
                             [(START + n * MINUTE) // 1000 for n in (0, 99, 200)])
            self.assertEqual(rows[0]["datetime"], "2026-11-07T08:35:00Z")
            self.assertEqual((rows[0]["volume"], rows[0]["exchange"]), ("628.557467", "okx"))

    def test_unaligned_time_and_invalid_volume_are_rejected(self):
        for row, error in [(candle(0, offset_ms=1), "整分钟"), (candle(0, volume="-5"), "非负")]:
            with self.subTest(error=error), self.assertRaisesRegex(ValueError, error):
                collector.fetch_batch(OKXFixture([row]), START, START + MINUTE)

    def test_failed_page_leaves_no_partial_csv(self):
        with TemporaryDirectory() as folder:
            args = SimpleNamespace(start=START, end=START + 101 * MINUTE, output_dir=Path(folder))
            exchange = OKXFixture([candle(0)], now_ms=START + 2 * 24 * 60 * MINUTE)
            get_page = exchange.public_get_market_history_candles

            def fail_second_page(params):
                if exchange.requests:
                    raise ccxt.NetworkError("test interruption")
                return get_page(params)

            exchange.public_get_market_history_candles = fail_second_page
            with self.assertRaises(ccxt.NetworkError):
                collector.run_round(exchange, args, now_ms=exchange.now_ms)
            self.assertFalse(list(args.output_dir.iterdir()))

    def test_import_failure_retains_file_and_retry_precedes_progress_lookup(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            pending = root / "pending"
            binary = root / "importer"
            binary.touch()
            importer = collector.Importer(binary, pending)
            args = SimpleNamespace(start=START, end=START + MINUTE, bootstrap_from=None)
            exchange = OKXFixture([candle(0)])
            original = []

            def fail(command, **_):
                file = Path(command[command.index("-file") + 1])
                original.append(file.read_bytes())
                self.assertIn("-anchor=", command)
                raise subprocess.CalledProcessError(1, command)

            with patch.object(collector.subprocess, "run", side_effect=fail), self.assertLogs(collector.LOG):
                with self.assertRaises(subprocess.CalledProcessError):
                    collector.run_round(exchange, args, importer, now_ms=exchange.now_ms)
            self.assertEqual(len(list(pending.iterdir())), 1)

            calls = []

            def recover(command, **_):
                calls.append(command[1])
                if command[1] == "-file":
                    self.assertEqual(Path(command[2]).read_bytes(), original[0])
                    return subprocess.CompletedProcess(command, 0)
                self.assertEqual(command[command.index("-source") + 1], "ccxt:okx")
                return subprocess.CompletedProcess(command, 0, stdout="2026-11-07T08:35:00Z\n")

            args.start = args.end = None
            with patch.object(collector.subprocess, "run", side_effect=recover), self.assertLogs(collector.LOG) as log:
                collector.run_round(OKXFixture([]), args, collector.Importer(binary, pending),
                                    now_ms=START + MINUTE)
            self.assertEqual(calls, ["-file", "-latest-open-time"])
            self.assertIn("2026-11-07T08:33:00Z", "\n".join(log.output))
            self.assertFalse(list(pending.iterdir()))


if __name__ == "__main__":
    unittest.main()
