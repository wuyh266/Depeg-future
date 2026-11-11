"""Create clearly marked synthetic test artifacts; NEVER a trained model delivery."""
import csv
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import pickle
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
from dg_final import DGModelFinal, build_features_v3
from package_dg import package
from predict import STATS, infer, iso, read_input, SCHEMA


def make_bundle(root, seq_len=120, horizon="0-3h", include_volume=True, include_deviation=True):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    rows = []
    for i in range(seq_len + 61):
        price = 1 + .002 * np.sin(i / 9) + .0002 * np.cos(i / 3)
        rows.append({"timestamp": int((start + timedelta(minutes=i)).timestamp()), "datetime": iso(start + timedelta(minutes=i)),
                     "open": price, "high": price + .001, "low": price - .001, "close": price, "volume": 100 + i % 19, "exchange": "okx"})
    input_file = root / "sample_input.csv"
    with input_file.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    frame = pd.DataFrame(rows)
    if not include_volume:
        frame = frame.drop(columns="volume")
    features, columns, _, _ = build_features_v3(frame, include_deviation=include_deviation)
    stats_columns = [c for c in STATS if c in columns]
    scaler = StandardScaler().fit(features)
    stats_scaler = StandardScaler().fit(features[:, [columns.index(c) for c in stats_columns]])
    torch.manual_seed(42)
    model = DGModelFinal(input_dim=len(columns), d_model=16, num_env=2, stats_dim=len(stats_columns), temperature=.1)
    checkpoint = {"model_state_dict": model.state_dict(), "seq_len":seq_len, "horizon":horizon, "d_model":16, "num_env":2, "stats_dim":len(stats_columns), "temperature":.1, "gate_floor":0.0}
    torch.save(checkpoint, root / "untrained_test_checkpoint.pt")
    for name, obj in (("scaler.pkl",scaler), ("stats_scaler.pkl",stats_scaler)):
        with (root / name).open("wb") as stream:
            pickle.dump(obj,stream)
    manifest = package(root / "untrained_test_checkpoint.pt", root / "scaler.pkl", root / "stats_scaler.pkl", root / "bundle",
                       "UNTRAINED-SMOKE-ONLY", include_deviation, include_volume)
    prediction = infer(read_input(input_file), root / "bundle")
    result = {"schema_version":SCHEMA, "ok":True, "generated_at":iso(datetime.now(timezone.utc)), "predictions":[prediction]}
    (root / "reference_output.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    (root / "READ_ME.txt").write_text("SYNTHETIC INPUT + UNTRAINED RANDOM WEIGHTS. Interface smoke test only. Not a financial prediction or trained model.\n", encoding="utf-8")
    for name in ("sample_input.csv", "reference_output.json", "READ_ME.txt"):
        shutil.copyfile(root / name, root / "bundle" / name)
    return input_file, manifest, model, scaler, stats_scaler


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parents[1] / "data" / "inference-smoke")
    destination = parser.parse_args().output_dir
    if not (destination / "bundle" / "manifest.json").is_file():
        make_bundle(destination)
    else:
        print("Reusing existing artifacts; use --output-dir with a new directory to regenerate.")
    print(f"UNTRAINED test artifacts: {destination}")
