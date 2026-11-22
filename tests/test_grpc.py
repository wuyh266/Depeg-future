from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import threading
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import grpc
from grpc_health.v1 import health_pb2, health_pb2_grpc
from depeg_rpc import inference_pb2 as pb, inference_pb2_grpc as rpc
import grpc_server
from make_smoke_bundle import make_bundle
from predict import SCHEMA, infer, read_input


class GRPCTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.input, _, _, _, _ = make_bundle(cls.root)
        cls.rows = read_input(cls.input)
        cls.expected = infer(cls.rows, cls.root / "bundle")
        cls.server, cls.port = grpc_server.create_server([cls.root / "bundle"], "127.0.0.1:0")
        cls.server.start()
        cls.channel = grpc.insecure_channel(f"127.0.0.1:{cls.port}")
        grpc.channel_ready_future(cls.channel).result(timeout=5)
        cls.stub = rpc.InferenceServiceStub(cls.channel)

    @classmethod
    def tearDownClass(cls):
        cls.channel.close()
        cls.server.stop(1).wait()
        cls.temp.cleanup()

    def request(self):
        return pb.PredictRequest(schema_version=SCHEMA, asset_id="ccxt.usdc-usdt", symbol="USDC/USDT", source="ccxt:okx", horizons=["0-3h"],
                                 candles=[pb.Candle(open_time_seconds=int(r["datetime"].timestamp()), **{k:str(r[k]) for k in ("open","high","low","close","volume")}) for r in self.rows])

    def test_health_and_prediction_match_offline_without_reloading_weights(self):
        result = health_pb2_grpc.HealthStub(self.channel).Check(health_pb2.HealthCheckRequest(service=grpc_server.SERVICE), timeout=3)
        self.assertEqual(result.status, health_pb2.HealthCheckResponse.SERVING)
        with patch("torch.load", side_effect=AssertionError("must reuse loaded model")):
            response = self.stub.Predict(self.request(), timeout=5)
        prediction = response.predictions[0]
        self.assertAlmostEqual(prediction.depeg_minute_ratio, self.expected["depeg_minute_ratio"], places=7)
        self.assertEqual(prediction.prediction_start, self.expected["prediction_start"])
        self.assertEqual(prediction.prediction_end, self.expected["prediction_end"])
        self.assertEqual(prediction.quality, "integration_only")
        self.assertTrue(prediction.HasField("depeg_minute_ratio"))

    def test_invalid_market_gap_short_input_and_missing_window(self):
        cases = []
        request = self.request(); request.symbol = "BTC/USDT"; cases.append((request, grpc.StatusCode.INVALID_ARGUMENT))
        request = self.request(); del request.candles[30]; cases.append((request, grpc.StatusCode.INVALID_ARGUMENT))
        request = self.request(); del request.candles[-1]; cases.append((request, grpc.StatusCode.INVALID_ARGUMENT))
        request = self.request(); request.horizons[0] = "3-6h"; cases.append((request, grpc.StatusCode.FAILED_PRECONDITION))
        request = self.request(); del request.horizons[:]; cases.append((request, grpc.StatusCode.INVALID_ARGUMENT))
        for request, expected in cases:
            with self.assertRaises(grpc.RpcError) as caught:
                self.stub.Predict(request, timeout=5)
            self.assertEqual(caught.exception.code(), expected)

    def test_deadline_busy_and_recovery(self):
        entered, release = threading.Event(), threading.Event()
        original = grpc_server.infer
        def blocked(*args, **kwargs):
            entered.set()
            release.wait(2)
            return original(*args, **kwargs)
        with patch("grpc_server.infer", side_effect=blocked):
            first = self.stub.Predict.future(self.request(), timeout=.15)
            self.assertTrue(entered.wait(1))
            with self.assertRaises(grpc.RpcError) as caught:
                self.stub.Predict(self.request(), timeout=1)
            self.assertEqual(caught.exception.code(), grpc.StatusCode.RESOURCE_EXHAUSTED)
            with self.assertRaises(grpc.RpcError) as caught:
                first.result()
            self.assertEqual(caught.exception.code(), grpc.StatusCode.DEADLINE_EXCEEDED)
            release.set()
        time.sleep(.2)
        self.assertTrue(self.stub.Predict(self.request(), timeout=5).predictions)

    def test_cancellation(self):
        future = self.stub.Predict.future(self.request(), timeout=5)
        future.cancel()
        self.assertEqual(future.code(), grpc.StatusCode.CANCELLED)

    def test_remote_binding_is_disabled(self):
        with self.assertRaisesRegex(ValueError, "loopback"):
            grpc_server.create_server([self.root / "bundle"], "0.0.0.0:0")


if __name__ == "__main__":
    unittest.main()
