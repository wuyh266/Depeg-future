"""Package one trained DG checkpoint and its own fitted scalers for offline inference."""
import argparse
import json
from pathlib import Path
import pickle
import shutil

import numpy as np
import pandas as pd
import torch

from dg_final import DGModelFinal, HORIZON_MAP, build_features_v3
from predict import FEATURE_VERSION, LABEL_VERSION, SCHEMA, STATS


def package(checkpoint_path, scaler_path, stats_scaler_path, output, model_version,
            include_deviation=True, include_volume=True, compatibility_note="", verified=False):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    required = ["seq_len", "horizon", "d_model", "num_env", "stats_dim", "temperature", "gate_floor"]
    for key in required:
        if key not in checkpoint:
            raise ValueError(f"checkpoint 缺少 {key}；请由模型交付方补充真实训练参数")
    if checkpoint["horizon"] not in HORIZON_MAP:
        raise ValueError("未知预测窗口")
    if type(checkpoint["seq_len"]) is not int or checkpoint["seq_len"] < 1:
        raise ValueError("seq_len 必须为正整数")
    if not model_version.strip() or (verified and not compatibility_note.strip()):
        raise ValueError("必须提供模型版本；正式交付还需训练数据口径核对依据")
    # Obtain column ordering from the authoritative implementation, not a duplicate list.
    sample = pd.DataFrame({"close": 1 + np.sin(np.arange(200) / 7) * .001})
    if include_volume:
        sample["volume"] = 10 + np.arange(200)
    _, columns, _, _ = build_features_v3(sample, include_deviation=include_deviation)
    stats_columns = [c for c in STATS if c in columns] if checkpoint["stats_dim"] else []
    if len(stats_columns) != checkpoint["stats_dim"]:
        raise ValueError("统计维度与 dg_final.py 的特征配置不一致")
    model = DGModelFinal(input_dim=len(columns), d_model=checkpoint["d_model"], num_env=checkpoint["num_env"],
                         stats_dim=checkpoint["stats_dim"], temperature=checkpoint["temperature"], gate_floor=checkpoint["gate_floor"])
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)

    def export_scaler(path, size):
        # Only package the trusted fitted scaler supplied by the model owner.
        with Path(path).open("rb") as stream:
            obj = pickle.load(stream)
        if getattr(obj, "n_features_in_", None) != size or not obj.with_mean or not obj.with_std:
            raise ValueError("仅支持训练脚本使用的完整 StandardScaler；维度必须匹配")
        mean, scale = np.asarray(obj.mean_), np.asarray(obj.scale_)
        if not np.isfinite(mean).all() or not np.isfinite(scale).all() or (scale <= 0).any():
            raise ValueError("标准化参数无效")
        return {"mean": mean.tolist(), "scale": scale.tolist()}

    prep = {"features": export_scaler(scaler_path, len(columns))}
    if stats_columns:
        if not stats_scaler_path:
            raise ValueError("此权重需要配套 stats scaler")
        prep["stats"] = export_scaler(stats_scaler_path, len(stats_columns))
    manifest = {
        "schema_version": SCHEMA, "model_version": model_version,
        "feature_version": FEATURE_VERSION, "label_version": LABEL_VERSION,
        "asset_id": "ccxt.usdc-usdt", "symbol": "USDC/USDT", "source": "ccxt:okx",
        "interval": "1m", "price_unit": "USDT", "volume_unit": "USDC",
        "include_deviation": include_deviation, "include_volume": include_volume,
        "feature_columns": columns, "stats_columns": stats_columns,
        "raw_candles_required": checkpoint["seq_len"] + 61,
        "alignment": "sequence-ends-reference-minus-one/v1",
        "label_offsets_minutes": list(HORIZON_MAP[checkpoint["horizon"]]),
        "quality": "verified" if verified else "integration_only",
        "data_compatibility_note": compatibility_note,
        "checkpoint": "model.pt", "preprocessing": "preprocessing.json",
        **{key: checkpoint[key] for key in required},
    }
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("交付目录必须为空，避免覆盖其他窗口的文件")
    output.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), **{key: checkpoint[key] for key in required}}, output / "model.pt")
    (output / "preprocessing.json").write_text(json.dumps(prep, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    for name in ("dg_final.py", "predict.py"):
        shutil.copyfile(Path(__file__).with_name(name), output / name)
    shutil.copyfile(Path(__file__).resolve().parents[1] / "requirements-inference.txt", output / "requirements-inference.txt")
    tested = Path(__file__).resolve().parents[1] / "requirements-inference-tested.txt"
    if tested.is_file():
        shutil.copyfile(tested, output / tested.name)
    (output / "README.txt").write_text(
        "DG Transformer offline bundle\nPython 3.11\n"
        "Install: python -m pip install -r requirements-inference.txt\n"
        "Run: python predict.py --input sample_input.csv --model-dir . --output prediction.json\n"
        "Read manifest.json for exact window, required candles, units and quality.\n"
        "integration_only means training/OKX compatibility is NOT verified.\n", encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--scaler", required=True, type=Path)
    parser.add_argument("--stats-scaler", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model-version", required=True)
    parser.add_argument("--include-deviation", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-volume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--verified", action="store_true", help="仅在交付方完成训练与 OKX 数据口径核对后使用")
    parser.add_argument("--data-compatibility-note", default="")
    parser.add_argument("--sample-input", type=Path, help="可选：复制验收输入并实际生成参考输出")
    args = parser.parse_args()
    try:
        manifest = package(args.checkpoint, args.scaler, args.stats_scaler, args.output_dir, args.model_version,
                           args.include_deviation, args.include_volume, args.data_compatibility_note, args.verified)
        if args.sample_input:
            from datetime import datetime, timezone
            from predict import infer, iso, read_input
            rows = read_input(args.sample_input)
            prediction = infer(rows, args.output_dir)
            shutil.copyfile(args.sample_input, args.output_dir / "sample_input.csv")
            result = {"schema_version": SCHEMA, "ok": True, "generated_at": iso(datetime.now(timezone.utc)), "predictions": [prediction]}
            (args.output_dir / "reference_output.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    except Exception as exc:
        parser.exit(2, f"打包失败: {exc}\n")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
