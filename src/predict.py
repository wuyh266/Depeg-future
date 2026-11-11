"""Offline DG inference. Imports the training model, never computes future labels."""
import argparse
import csv
from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path
import sys

SCHEMA = "dg-inference/v1"
FEATURE_VERSION = "dg-build-features-v3-window10/v1"
LABEL_VERSION = "absolute-close-minus-one-ge-0.05/v1"
STATS = ["volatility_long", "dev_mean", "dev_max", "return_cum", "rolling_std", "rolling_skew"]


class InferenceError(Exception):
    def __init__(self, code, message):
        self.code, self.message = code, message
        super().__init__(message)


def fail(code, message):
    raise InferenceError(code, message)


def utc(value):
    try:
        stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if stamp.tzinfo is None or stamp.second or stamp.microsecond:
            raise ValueError()
        return stamp.astimezone(timezone.utc)
    except (ValueError, TypeError, AttributeError):
        fail("INVALID_INPUT", "datetime 必须带时区并对齐整分钟")


def iso(stamp):
    return stamp.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def read_input(path, now=None):
    if not path.is_file():
        fail("INPUT_NOT_FOUND", f"输入文件不存在: {path}")
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"timestamp", "datetime", "open", "high", "low", "close", "volume", "exchange"}.issubset(reader.fieldnames or []):
            fail("INVALID_INPUT", "缺少 OKX 分钟 K 线字段")
        rows = list(reader)
    if not rows:
        fail("INSUFFICIENT_DATA", "输入为空")
    result = []
    for row in rows:
        stamp = utc(row["datetime"])
        try:
            epoch = int(row["timestamp"])
            values = {key: float(row[key]) for key in ("open", "high", "low", "close", "volume")}
        except (ValueError, TypeError, OverflowError):
            fail("INVALID_INPUT", "timestamp 必须为秒，行情字段必须为有限数值")
        if epoch != int(stamp.timestamp()) or row["exchange"] != "okx":
            fail("INVALID_INPUT", "时间戳不一致或 exchange 不是 okx")
        if not all(math.isfinite(x) for x in values.values()):
            fail("INVALID_INPUT", "行情包含 NaN 或无穷值")
        if not (0 < values["low"] <= min(values["open"], values["close"]) <= max(values["open"], values["close"]) <= values["high"]) or values["volume"] < 0:
            fail("INVALID_INPUT", "OHLC 或成交量不合法")
        if result and stamp != result[-1]["datetime"] + timedelta(minutes=1):
            fail("NON_CONTIGUOUS_DATA", "K 线必须升序、无重复且逐分钟连续；不自动补齐")
        result.append({"datetime": stamp, **values})
    if result[-1]["datetime"] + timedelta(minutes=1) > (now or datetime.now(timezone.utc)):
        fail("UNCLOSED_CANDLE", "最后一根 K 线尚未收盘")
    return result


def artifact(folder, name):
    if not isinstance(name, str):
        fail("INVALID_MANIFEST", "模型文件名必须为字符串")
    path = (folder / name).resolve()
    if not path.is_relative_to(folder.resolve()):
        fail("INVALID_MANIFEST", "模型文件必须位于交付目录内")
    if not path.is_file():
        fail("MODEL_FILE_MISSING", f"模型文件不存在: {path}")
    return path


def load_manifest(folder):
    try:
        m = json.loads(artifact(folder, "manifest.json").read_text(encoding="utf-8"))
        if m["schema_version"] != SCHEMA or m["feature_version"] != FEATURE_VERSION or m["label_version"] != LABEL_VERSION:
            raise ValueError("不支持的版本")
        if m["symbol"] != "USDC/USDT" or m["source"] != "ccxt:okx" or m["asset_id"] != "ccxt.usdc-usdt":
            raise ValueError("当前入口只接受 OKX USDC/USDT")
        if m["price_unit"] != "USDT" or m["volume_unit"] != "USDC" or m["interval"] != "1m":
            raise ValueError("行情单位不匹配")
        if type(m["seq_len"]) is not int or m["seq_len"] < 1 or m["raw_candles_required"] != m["seq_len"] + 61:
            raise ValueError("序列长度或预热数量错误")
        if type(m["include_deviation"]) is not bool or type(m["include_volume"]) is not bool:
            raise ValueError("特征开关必须为布尔值")
        if m["alignment"] != "sequence-ends-reference-minus-one/v1":
            raise ValueError("时间对齐版本不匹配")
        if m["quality"] not in ("integration_only", "verified") or not m["model_version"]:
            raise ValueError("模型版本或质量标识缺失")
        if m["quality"] == "verified" and not m["data_compatibility_note"].strip():
            raise ValueError("正式模型必须记录训练数据口径核对依据")
        for key in ("checkpoint", "preprocessing"):
            artifact(folder, m[key])
        return m
    except (KeyError, ValueError, TypeError) as exc:
        fail("INVALID_MANIFEST", f"模型约定不合法: {exc}")


def infer(rows, folder, device="cpu"):
    m = load_manifest(folder)
    needed = m["raw_candles_required"]
    if len(rows) < needed:
        fail("INSUFFICIENT_DATA", f"至少需要 {needed} 根连续已收盘 K 线，收到 {len(rows)} 根")
    try:
        import numpy as np
        import pandas as pd
        import torch
        from dg_final import DGModelFinal, HORIZON_MAP, build_features_v3, predict_batch
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        fail("DEPENDENCY_MISSING", f"请安装 requirements-inference.txt: {exc}")
    if device == "cuda" and not torch.cuda.is_available():
        fail("DEVICE_UNAVAILABLE", "当前环境没有可用 CUDA")
    # The training code's X[i-seq_len:i] excludes reference candle i.
    frame = pd.DataFrame(rows[-needed:-1])
    if not m["include_volume"]:
        frame = frame.drop(columns="volume")
    features, columns, indices, _ = build_features_v3(frame, window=10, include_deviation=m["include_deviation"])
    if columns != m["feature_columns"]:
        fail("MODEL_INCOMPATIBLE", "特征顺序与交付约定不一致")
    if indices != list(range(60, len(frame))) or not np.isfinite(features).all():
        fail("INVALID_FEATURES", "滚动特征无效（例如窗口内成交量全为零）；不删除分钟或填零")
    stats_columns = [c for c in STATS if c in columns] if m["stats_columns"] else []
    if stats_columns != m["stats_columns"]:
        fail("MODEL_INCOMPATIBLE", "统计特征顺序不匹配")
    checkpoint = torch.load(artifact(folder, m["checkpoint"]), map_location="cpu", weights_only=True)
    for key in ("seq_len", "horizon", "d_model", "num_env", "stats_dim", "temperature", "gate_floor"):
        if checkpoint[key] != m[key]:
            fail("MODEL_INCOMPATIBLE", f"权重与 manifest 的 {key} 不一致")
    if m["horizon"] not in HORIZON_MAP or list(HORIZON_MAP[m["horizon"]]) != m["label_offsets_minutes"]:
        fail("MODEL_INCOMPATIBLE", "预测窗口与训练实现不匹配")
    prep = json.loads(artifact(folder, m["preprocessing"]).read_text(encoding="utf-8"))

    def scaler(name, size):
        spec = prep[name]
        mean, scale = np.asarray(spec["mean"], dtype=float), np.asarray(spec["scale"], dtype=float)
        if mean.shape != (size,) or scale.shape != (size,) or not np.isfinite(mean).all() or not np.isfinite(scale).all() or (scale <= 0).any():
            fail("MODEL_INCOMPATIBLE", "标准化参数维度或数值无效")
        obj = StandardScaler()
        obj.mean_, obj.scale_, obj.n_features_in_ = mean, scale, size
        return obj

    model = DGModelFinal(input_dim=len(columns), d_model=m["d_model"], num_env=m["num_env"],
                         stats_dim=m["stats_dim"], temperature=m["temperature"], gate_floor=m["gate_floor"])
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)
    stats = features[-1:, [columns.index(c) for c in stats_columns]] if stats_columns else None
    prediction = float(predict_batch(model, features[None, :, :], stats,
                       scaler("features", len(columns)), scaler("stats", len(stats_columns)) if stats_columns else None, device)[0])
    if not math.isfinite(prediction) or not 0 <= prediction <= 1:
        fail("INVALID_PREDICTION", "模型输出不在 [0,1] 内")
    reference = rows[-1]["datetime"]
    start, end = m["label_offsets_minutes"]
    return {
        "asset_id": m["asset_id"], "symbol": m["symbol"], "source": m["source"],
        "observed_at": iso(reference + timedelta(minutes=1)),
        "reference_open_time": iso(reference), "last_input_open_time": iso(reference - timedelta(minutes=1)),
        "horizon": m["horizon"], "horizon_start_minutes": start - 1, "horizon_end_minutes": end,
        "prediction_start": iso(reference + timedelta(minutes=start)),
        "prediction_end": iso(reference + timedelta(minutes=end + 1)),
        "depeg_minute_ratio": prediction, "model_version": m["model_version"],
        "feature_version": m["feature_version"], "label_version": m["label_version"], "quality": m["quality"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--model-dir", required=True, type=Path, action="append", help="可重复指定，每个目录交付一个窗口")
    parser.add_argument("--output", type=Path, help="可选；stdout 始终输出同一 JSON")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    try:
        rows = read_input(args.input)
        predictions = [infer(rows, folder, args.device) for folder in args.model_dir]
        if len({p["horizon"] for p in predictions}) != len(predictions):
            fail("INVALID_MANIFEST", "同一请求不允许重复窗口")
        result = {"schema_version": SCHEMA, "ok": True, "generated_at": iso(datetime.now(timezone.utc)), "predictions": predictions}
        status = 0
    except InferenceError as exc:
        result, status = {"schema_version": SCHEMA, "ok": False, "error": {"code": exc.code, "message": exc.message}}, 2
    except Exception as exc:
        result, status = {"schema_version": SCHEMA, "ok": False, "error": {"code": "INFERENCE_FAILED", "message": str(exc)}}, 2
    payload = json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2)
    if args.output:
        try:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            temp = args.output.with_suffix(args.output.suffix + ".tmp")
            temp.write_text(payload + "\n", encoding="utf-8")
            temp.replace(args.output)
        except OSError as exc:
            print(json.dumps({"schema_version": SCHEMA, "ok": False, "error": {"code": "OUTPUT_WRITE_FAILED", "message": str(exc)}}))
            return 2
    print(payload)
    return status


if __name__ == "__main__":
    sys.exit(main())
