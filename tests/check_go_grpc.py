"""Actual Go executable -> Python gRPC integration test with UNTRAINED fixtures."""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import grpc
from make_smoke_bundle import make_bundle
from predict import infer, read_input


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--go-client", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    executable = args.go_client.resolve()
    repo = Path(__file__).resolve().parents[1]
    with TemporaryDirectory() as directory:
        root = Path(directory)
        input_file, _, _, _, _ = make_bundle(root)
        expected = infer(read_input(input_file), root / "bundle")
        with socket.socket() as port_picker:
            port_picker.bind(("127.0.0.1", 0))
            port = port_picker.getsockname()[1]
        address = f"127.0.0.1:{port}"
        environment = dict(os.environ, PYTHONUTF8="1")
        with (root / "server.log").open("w", encoding="utf-8") as logs:
            server = subprocess.Popen([sys.executable, "src/grpc_server.py", "--model-dir", str(root / "bundle"), "--bind", address], cwd=repo, env=environment,
                                      stdout=logs, stderr=logs, creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            channel = grpc.insecure_channel(address)
            try:
                grpc.channel_ready_future(channel).result(timeout=20)
                command = [str(executable), "-address", address, "-input", str(input_file), "-timeout", "5s"]
                success = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", timeout=15)
                if success.returncode:
                    raise AssertionError(success.stderr)
                predictions = json.loads(success.stdout)
                error = abs(predictions[0]["DepegMinuteRatio"] - expected["depeg_minute_ratio"])
                assert error <= 1e-6, "Go/Python result differs from offline inference"
                assert predictions[0]["Quality"] == "integration_only"
                assert predictions[0]["ObservedAt"] == expected["observed_at"]
                missing = subprocess.run(command + ["-horizons", "3-6h"], capture_output=True, text=True, encoding="utf-8", timeout=15)
                assert missing.returncode == 2 and "FailedPrecondition" in missing.stderr, "Missing model status was lost"
                result = {"ok": True, "transport": "Go -> Python gRPC", "absolute_error_vs_offline": error,
                          "quality": "integration_only", "weights": "UNTRAINED-SMOKE-ONLY", "missing_window": "FailedPrecondition"}
                if args.output:
                    args.output.parent.mkdir(parents=True, exist_ok=True)
                    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
                print(json.dumps(result, indent=2))
            finally:
                channel.close()
                if os.name == "nt":
                    # The Windows venv launcher can have an interpreter child holding the log.
                    subprocess.run(["taskkill", "/PID", str(server.pid), "/T", "/F"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=subprocess.CREATE_NO_WINDOW)
                else:
                    server.terminate()
                server.wait(timeout=10)


if __name__ == "__main__":
    main()
