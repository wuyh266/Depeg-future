import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pandas as pd
from dg_final import build_features_v3, create_sequences, predict_batch
from predict import InferenceError, infer, read_input, STATS
from make_smoke_bundle import make_bundle


class InferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.input, cls.manifest, cls.model, cls.scaler, cls.stats_scaler = make_bundle(cls.root)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_matches_training_sequence_and_scalers(self):
        frame = pd.read_csv(self.input)
        features, columns, _, _ = build_features_v3(frame)
        sequences, _ = create_sequences(features, np.zeros(len(features)), 120)
        stats = features[-2:-1, [columns.index(c) for c in STATS]]
        expected = predict_batch(self.model, sequences, stats, self.scaler, self.stats_scaler, "cpu")[0]
        result = infer(read_input(self.input), self.root / "bundle")
        self.assertAlmostEqual(float(expected), result["depeg_minute_ratio"], places=7)
        self.assertEqual(result["prediction_start"], "2026-01-01T03:01:00Z")
        self.assertEqual(result["prediction_end"], "2026-01-01T06:01:00Z")
        self.assertEqual(result["last_input_open_time"], "2026-01-01T02:59:00Z")
        self.assertEqual(result["quality"], "integration_only")

    def test_cli_success_and_missing_file_replace_stale_result(self):
        output = self.root / "out.json"
        command = [sys.executable, "src/predict.py", "--input", str(self.input), "--model-dir", str(self.root / "bundle"), "--output", str(output)]
        success = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(success.returncode, 0, success.stdout + success.stderr)
        self.assertTrue(json.loads(success.stdout)["ok"])
        command[3] = str(self.root / "missing.csv")
        failure = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(failure.returncode, 2)
        self.assertEqual(json.loads(output.read_text(encoding="utf-8"))["error"]["code"], "INPUT_NOT_FOUND")

    def test_bad_data_and_missing_artifact(self):
        rows = read_input(self.input)
        with self.assertRaisesRegex(InferenceError, "181"):
            infer(rows[:-1], self.root / "bundle")
        for column, value in (("volume", 0), ("close", float("nan"))):
            broken = [dict(row, **{column: value}) for row in rows]
            with self.assertRaises(InferenceError):
                infer(broken, self.root / "bundle")
        with self.assertRaises(InferenceError) as caught:
            infer(rows, self.root / "missing-bundle")
        self.assertEqual(caught.exception.code, "MODEL_FILE_MISSING")

    def test_csv_validation(self):
        frame = pd.read_csv(self.input)
        cases = [frame.drop(index=40), frame.iloc[::-1], pd.concat([frame.iloc[:40], frame.iloc[39:]])]
        for frame_bad in cases:
            path = self.root / "bad.csv"
            frame_bad.to_csv(path, index=False)
            with self.assertRaises(InferenceError) as caught:
                read_input(path)
            self.assertEqual(caught.exception.code, "NON_CONTIGUOUS_DATA")

    def test_other_feature_configuration_and_window(self):
        with TemporaryDirectory() as folder:
            path, manifest, _, _, _ = make_bundle(folder, seq_len=60, horizon="6-9h", include_volume=False, include_deviation=False)
            result = infer(read_input(path), Path(folder) / "bundle")
            self.assertEqual(manifest["raw_candles_required"], 121)
            self.assertEqual(result["horizon_start_minutes"], 360)
            self.assertEqual(result["horizon_end_minutes"], 540)

    def test_standalone_bundle(self):
        command = [sys.executable, "predict.py", "--input", str(self.input), "--model-dir", "."]
        result = subprocess.run(command, cwd=self.root / "bundle", capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertTrue(json.loads(result.stdout)["ok"])

    def test_rejects_milliseconds_wrong_source_and_open_candle(self):
        frame = pd.read_csv(self.input)
        from datetime import datetime, timezone
        for column, value in (("timestamp", frame.iloc[0]["timestamp"] * 1000), ("exchange", "binance")):
            broken = frame.copy()
            broken.loc[0, column] = value
            path = self.root / "invalid.csv"
            broken.to_csv(path, index=False)
            with self.assertRaises(InferenceError) as caught:
                read_input(path)
            self.assertEqual(caught.exception.code, "INVALID_INPUT")
        with self.assertRaises(InferenceError) as caught:
            read_input(self.input, now=datetime(2026, 1, 1, tzinfo=timezone.utc))
        self.assertEqual(caught.exception.code, "UNCLOSED_CANDLE")

    def test_manifest_and_checkpoint_disagreement(self):
        path = self.root / "bundle" / "manifest.json"
        original = path.read_text(encoding="utf-8")
        try:
            manifest = json.loads(original)
            manifest["temperature"] = .2
            path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaises(InferenceError) as caught:
                infer(read_input(self.input), path.parent)
            self.assertEqual(caught.exception.code, "MODEL_INCOMPATIBLE")
        finally:
            path.write_text(original, encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
