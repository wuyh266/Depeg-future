"""Opt-in loopback gRPC inference service. Does not collect data or train models."""
import argparse
from concurrent import futures
from datetime import datetime, timezone
import ipaddress
import logging
from pathlib import Path
import signal
import threading

import grpc
from grpc_health.v1 import health, health_pb2, health_pb2_grpc

from depeg_rpc import inference_pb2 as pb, inference_pb2_grpc as rpc
from predict import InferenceError, SCHEMA, infer, iso, load_runtime, validate_rows

SERVICE = "depeg.inference.v1.InferenceService"
MAX_CANDLES = 4096


class InferenceService(rpc.InferenceServiceServicer):
    def __init__(self, folders, device="cpu"):
        self.device = device
        self.models = {}
        self.slot = threading.BoundedSemaphore(1)
        for folder in folders:
            runtime = load_runtime(folder, device)
            horizon = runtime["manifest"]["horizon"]
            if horizon in self.models:
                raise ValueError("Duplicate model horizon")
            self.models[horizon] = (folder, runtime)
        if not self.models:
            raise ValueError("At least one model is required")

    def Predict(self, request, context):
        if request.schema_version != SCHEMA or (request.asset_id, request.symbol, request.source) != ("ccxt.usdc-usdt", "USDC/USDT", "ccxt:okx"):
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "INVALID_INPUT: unsupported schema or market")
        if not request.horizons or len(set(request.horizons)) != len(request.horizons):
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "INVALID_INPUT: explicit unique horizons required")
        if any(h not in self.models for h in request.horizons):
            context.abort(grpc.StatusCode.FAILED_PRECONDITION, "MODEL_NOT_AVAILABLE: requested window not loaded")
        if len(request.candles) > MAX_CANDLES:
            context.abort(grpc.StatusCode.RESOURCE_EXHAUSTED, "TOO_MANY_CANDLES: maximum 4096")
        if not self.slot.acquire(blocking=False):
            context.abort(grpc.StatusCode.RESOURCE_EXHAUSTED, "INFERENCE_BUSY: one inference at a time")
        try:
            self.check_active(context)
            rows = []
            for candle in request.candles:
                rows.append({"timestamp": str(candle.open_time_seconds),
                             "datetime": iso(datetime.fromtimestamp(candle.open_time_seconds, timezone.utc)),
                             "exchange": "okx", **{key: getattr(candle, key) for key in ("open", "high", "low", "close", "volume")}})
            rows = validate_rows(rows)
            predictions = []
            for horizon in request.horizons:
                self.check_active(context)
                folder, runtime = self.models[horizon]
                result = infer(rows, folder, self.device, runtime=runtime)
                self.check_active(context)
                predictions.append(pb.Prediction(**result))
            return pb.PredictResponse(schema_version=SCHEMA, generated_at=iso(datetime.now(timezone.utc)), predictions=predictions)
        except InferenceError as exc:
            status = grpc.StatusCode.INVALID_ARGUMENT
            if exc.code in ("MODEL_INCOMPATIBLE", "INVALID_PREDICTION"):
                status = grpc.StatusCode.FAILED_PRECONDITION
            context.abort(status, f"{exc.code}: {exc.message}")
        except (ValueError, OverflowError, OSError) as exc:
            logging.error("gRPC input rejected: %s", type(exc).__name__)
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "INVALID_INPUT: invalid timestamp or candle")
        except Exception as exc:
            if not context.is_active():
                raise
            logging.error("gRPC inference failed: %s", type(exc).__name__)
            context.abort(grpc.StatusCode.INTERNAL, "INFERENCE_FAILED: see server diagnostics")
        finally:
            self.slot.release()

    @staticmethod
    def check_active(context):
        if not context.is_active():
            context.abort(grpc.StatusCode.CANCELLED, "REQUEST_CANCELED")


def create_server(folders, address, device="cpu"):
    # This proof uses local transport credentials. Remote deployment needs TLS/mTLS first.
    host, _, _ = address.rpartition(":")
    if not ipaddress.ip_address(host.strip("[]")).is_loopback:
        raise ValueError("Only loopback binding is enabled in this prototype")
    service = InferenceService(folders, device)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4), options=[("grpc.max_receive_message_length", 4 * 1024 * 1024)])
    rpc.add_InferenceServiceServicer_to_server(service, server)
    readiness = health.HealthServicer()
    health_pb2_grpc.add_HealthServicer_to_server(readiness, server)
    readiness.set("", health_pb2.HealthCheckResponse.SERVING)
    readiness.set(SERVICE, health_pb2.HealthCheckResponse.SERVING)
    port = server.add_insecure_port(address)
    if not port:
        raise RuntimeError("Cannot bind gRPC address")
    return server, port


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, action="append", type=Path)
    parser.add_argument("--bind", default="127.0.0.1:50051")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    try:
        server, port = create_server(args.model_dir, args.bind, args.device)
    except Exception as exc:
        parser.exit(2, f"gRPC startup failed: {type(exc).__name__}: {exc}\n")
    def stop(*_):
        server.stop(grace=5)
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    server.start()
    logging.info("DG gRPC ready on %s (port %s)", args.bind, port)
    server.wait_for_termination()


if __name__ == "__main__":
    main()
