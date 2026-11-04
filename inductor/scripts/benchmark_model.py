"""Measure an immutable AOT package using the compiler's saved boundary inputs."""

import argparse
import datetime
import importlib
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import statistics
import sys
import time
import traceback
import uuid

from compile_aot import file_hash, source_identity, tensor_metadata, write_json, validate_reference_contract
from gpu_guard import admit, supervise


def validate_runtime_source(actual, expected):
    # An immutable worktree may relocate the same source contents.
    for key in ("commit", "status", "diff_sha256", "untracked_python_sha256", "untracked_template_sha256", "package_source_sha256", "kind"):
        if actual.get(key, {}) != expected.get(key, {}):
            raise RuntimeError(f"Source differs from package compilation: {key}")


def summarize(samples):
    result = {}
    for mode in ("ordinary", "graph"):
        rows = [row for row in samples if row["mode"] == mode]
        result[mode] = {"samples": len(rows), "forwards": sum(row["k"] for row in rows)}
        for key in ("event_ms", "wall_ms"):
            values = sorted(row[key] for row in rows)
            result[mode][key] = {
                "median": statistics.median(values),
                "p95_nearest_rank": values[math.ceil(0.95 * len(values)) - 1],
                "round_medians": {
                    str(index): statistics.median(row[key] for row in rows if row["round"] == index)
                    for index in sorted({row["round"] for row in rows})
                },
            }
        result[mode]["loop_forwards_per_second"] = result[mode]["forwards"] / (
            sum(row["wall_ms"] * row["k"] for row in rows) / 1000
        )
    return result


def worker(request_path):
    request = json.loads(Path(request_path).read_text())
    run = Path(request["run_dir"])
    compile_run = Path(request["compile_run"])
    compilation = json.loads((compile_run / "manifest.json").read_text())
    contract = compilation["request"]
    result = {"status": "running", "request": request, "pid": os.getpid()}
    try:
        # Editable-install .pth hooks can otherwise override PYTHONPATH.
        sys.path.insert(0, request["runtime_source"]["root"])
        import torch
        from torch.utils import _pytree

        torch.set_num_threads(1)
        torch.cuda.set_device(0)
        source_root = Path(torch.__file__).resolve().parents[1]
        if source_root != Path(request["runtime_source"]["root"]):
            raise RuntimeError(f"Unexpected imported Torch root: {source_root}")
        # Git identity is checked by the outer, non-injected supervisor.
        result["source"] = request["runtime_source"]
        result["runtime"] = {
            "python": sys.executable, "torch": torch.__version__, "torch_file": torch.__file__,
            "torch_c_extension": str(Path(torch._C.__file__).resolve()),
            "torch_c_extension_sha256": file_hash(torch._C.__file__),
            "cuda_build": torch.version.cuda, "cpu_threads": torch.get_num_threads(),
        }
        validate_runtime_source(result["source"], compilation["source"])
        if result["runtime"]["torch_c_extension_sha256"] != compilation["runtime"]["torch_c_extension_sha256"]:
            raise RuntimeError("Native Torch runtime differs from package compilation")
        for key, value in request["runtime_backends"].items():
            target = torch.backends
            parts = key.split(".")
            for part in parts[:-1]:
                target = getattr(target, part)
            setattr(target, parts[-1], value)
        if contract["loader"] == "torchfourier":
            fourier = Path(contract["fourier_root"])
            sys.path[:0] = [str(fourier), str(fourier / "fourier")]
            runtime = importlib.import_module("fourier.aot_runtime")
            if contract["allow_triton_version_override"]:
                runtime.AOT_TRITON_VERSION = runtime.installed_triton_version()
            ops = importlib.import_module("fourier.aot_ops")
            config = json.loads((Path(contract["input"]).parent / "aotconfig.json").read_text())
            ops.register_required_ops(config, artifact_path=Path(contract["input"]), strict=True)
        artifacts = {
            "package": compilation["package"], "inputs": compilation["saved_inputs"],
            "reference": compilation["reference"],
        }
        if request.get("validation_contract"):
            artifacts["reference"] = request["validation_contract"]["reference"]
        result["validation_basis"] = request.get("validation_contract") or compilation.get("validation_basis") or {
            "kind": "prepared_eager", "model_accuracy_status": "passed_on_original_example",
        }
        result["runtime_backends"] = request["runtime_backends"]
        for name, info in artifacts.items():
            if file_hash(info["path"]) != info["sha256"]:
                raise RuntimeError(f"Changed {name} artifact")
        result["artifacts"] = artifacts
        model = torch._inductor.aoti_load_package(artifacts["package"]["path"], run_single_threaded=True)
        args, kwargs = torch.load(artifacts["inputs"]["path"], map_location="cuda:0", weights_only=False)
        reference = torch.load(artifacts["reference"]["path"], map_location="cuda:0", weights_only=False)
        result["inputs"] = tensor_metadata((args, kwargs))

        def clone(value):
            return _pytree.tree_map(lambda x: x.detach().clone() if isinstance(x, torch.Tensor) else x, value)

        def compare(actual, expected):
            torch.testing.assert_close(actual, expected, rtol=(request.get("validation_contract") or contract)["rtol"], atol=(request.get("validation_contract") or contract)["atol"], equal_nan=False)
            for leaf in _pytree.tree_leaves(actual):
                if isinstance(leaf, torch.Tensor) and not bool(torch.isfinite(leaf).all()):
                    raise RuntimeError("Non-finite output")

        with torch.inference_mode():
            ordinary = clone(model(*args, **kwargs))
            torch.cuda.synchronize()
            compare(ordinary, reference)
            if request["mode"] == "capture":
                capture = run / "capture/capture.json"
                if not capture.is_file():
                    raise RuntimeError("Requested driver launch was not captured")
                result["capture"] = json.loads(capture.read_text())
                expected = request["capture_spec"].get("expected_cases")
                if expected is not None and len(result["capture"].get("cases", [])) != expected:
                    raise RuntimeError("Captured BMM case count differs from the reviewed contract")
                result["capture_sha256"] = {str(path.relative_to(capture.parent)): file_hash(path)
                                           for path in capture.parent.rglob("*") if path.is_file()}
                result["validation"] = {"instrumented_full_model_vs_reference": "passed"}
                result["status"] = "succeeded"
                return
            for _ in range(30):
                outputs = model(*args, **kwargs)
            torch.cuda.synchronize()
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(5):
                    outputs = model(*args, **kwargs)
            torch.cuda.current_stream().wait_stream(side)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            captured_outputs = []
            with torch.cuda.graph(graph, stream=side):
                for _ in range(request["k"]):
                    captured_outputs.append(model(*args, **kwargs))
            graph.replay()
            torch.cuda.synchronize()
            replay_output = clone(captured_outputs[-1])
            compare(replay_output, ordinary)
            compare(replay_output, reference)
            for _ in range(5):
                graph.replay()
            torch.cuda.synchronize()
            result["validation"] = {
                "ordinary_vs_reference": "passed", "graph_vs_ordinary": "passed",
                "graph_vs_reference": "passed", "graph_output_coverage": "last forward of each replay",
                "rtol": (request.get("validation_contract") or contract)["rtol"], "atol": (request.get("validation_contract") or contract)["atol"],
                "outputs": tensor_metadata(ordinary),
            }
            write_json(run / "result.json", result)
            samples = []
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            end.record()
            end.synchronize()

            def block(mode):
                if mode == "ordinary":
                    for _ in range(request["k"]):
                        model(*args, **kwargs)
                else:
                    graph.replay()

            if request["mode"] == "benchmark":
                for round_index in range(request["rounds"]):
                    order = ("ordinary", "graph") if round_index % 2 == 0 else ("graph", "ordinary")
                    for mode in order:
                        for sample in range(request["samples"]):
                            torch.cuda.synchronize()
                            wall = time.perf_counter()
                            start.record()
                            block(mode)
                            end.record()
                            end.synchronize()
                            samples.append({
                                "round": round_index, "sample": sample, "mode": mode,
                                "k": request["k"], "order": len(samples),
                                "wall_ms": (time.perf_counter() - wall) * 1000 / request["k"],
                                "event_ms": start.elapsed_time(end) / request["k"],
                            })
                write_json(run / "samples.json", samples)
                result["summary"] = summarize(samples)
            else:
                torch.cuda.profiler.start()
                counts = {}
                for mode in ("ordinary", "graph"):
                    deadline = time.monotonic() + request["capture_seconds"]
                    index = 0
                    with torch.cuda.nvtx.range("kda/" + mode):
                        while time.monotonic() < deadline or index < 3:
                            with torch.cuda.nvtx.range(f"kda/{mode}/block/{index}/K{request['k']}"):
                                block(mode)
                                torch.cuda.synchronize()
                            index += 1
                    counts[mode] = index
                torch.cuda.profiler.stop()
                result["profile_blocks"] = counts
            result["status"] = "succeeded"
    except BaseException:
        result["status"] = "failed"
        result["error"] = traceback.format_exc()
        raise
    finally:
        write_json(run / "result.json", result)


def validate_contract(compilation, validation_contract):
    if validation_contract is not None:
        validate_reference_contract(compilation, validation_contract)
    elif compilation["status"] != "succeeded" or not compilation["validation"]["passed"]:
        raise RuntimeError("Formal measurement requires successful compilation and numerical validation")


def launch(args):
    if min(args.k, args.rounds, args.samples) <= 0 or not 0 < args.capture_seconds < float("inf"):
        raise ValueError("K, rounds, samples and capture seconds must be positive and finite")
    compilation = json.loads((args.compile_run / "manifest.json").read_text())
    validation_contract = json.loads(args.validation_contract.read_text()) if args.validation_contract else None
    validate_contract(compilation, validation_contract)
    source_root = (args.source_root or Path(compilation["source"]["root"])).resolve()
    runtime_source = source_identity(source_root)
    validate_runtime_source(runtime_source, compilation["source"])
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    run = args.output_root.resolve() / f"{stamp}_{args.mode}_{uuid.uuid4().hex[:8]}"
    run.mkdir(parents=True)
    for name in ("tmp", "cache", "triton_cache", "scripts"):
        (run / name).mkdir()
    sample = admit(args.gpu, run / "gpu_admission.json", args.max_idle_memory_mib)
    gpu = sample["device"]["uuid"]
    env = dict(os.environ)
    for key, value in compilation["request"]["environment"].items():
        if value is None:
            env.pop(key, None)
        else:
            env[key] = value
    env.update(CUDA_VISIBLE_DEVICES=gpu, TMPDIR=str(run / "tmp"),
               TORCHINDUCTOR_CACHE_DIR=str(run / "cache"), TRITON_CACHE_DIR=str(run / "triton_cache"),
               PYTHONPATH=str(source_root))
    request = {"compile_run": str(args.compile_run.resolve()), "run_dir": str(run),
               "mode": args.mode, "k": args.k, "rounds": args.rounds, "samples": args.samples,
               "capture_seconds": args.capture_seconds, "gpu": sample["device"],
               "protocol": "GPU-resident warm inputs; K-forward block means; not request latency"}
    request["runtime_source"] = runtime_source
    request["validation_contract"] = validation_contract
    request["runtime_backends"] = (validation_contract or {}).get("torch_backends", compilation["request"]["torch_backends"])
    if args.mode == "capture":
        if args.capture_spec is None:
            raise ValueError("--capture-spec is required for capture mode")
        specification = json.loads(args.capture_spec.read_text())
        library = Path(specification["library"]).resolve(strict=True)
        request["capture_spec"] = specification
        request["capture_library_sha256"] = file_hash(library)
        env.update(LD_PRELOAD=str(library), KDA_CAPTURE_DIR=str(run / "capture"))
        if specification.get("kind", "kernel") == "aten_bmm":
            shapes = specification["shapes"]
            if not shapes or any(len(row) != 4 or any(type(v) is not int or v <= 0 for v in row) for row in shapes):
                raise ValueError("BMM shapes must be explicit positive [batch, M, K, N] rows")
            for key in ("expected_cases", "max_tensor_bytes"):
                if type(specification[key]) is not int or specification[key] <= 0:
                    raise ValueError(f"Capture {key} must be positive")
            env.update(KDA_CAPTURE_BMM_SHAPES=";".join(",".join(map(str, row)) for row in shapes),
                       KDA_CAPTURE_MAX_CASES=str(specification["expected_cases"]),
                       KDA_CAPTURE_MAX_BYTES=str(specification["max_tensor_bytes"]))
        elif specification.get("kind", "kernel") == "kernel":
            env.update(KDA_CAPTURE_KERNEL=specification["kernel"], KDA_CAPTURE_ARGS=specification["arguments"])
        else:
            raise ValueError("Unknown capture kind")
    for name in ("benchmark_model.py", "compile_aot.py", "gpu_guard.py"):
        shutil.copy2(Path(__file__).parent / name, run / "scripts" / name)
    command = [sys.executable, "-B", str(run / "scripts/benchmark_model.py"), "--worker", str(run / "request.json")]
    if args.mode == "profile":
        command = [args.nsys, "profile", "--trace=cuda,nvtx", "--cuda-graph-trace=node",
                   "--capture-range=cudaProfilerApi", "--capture-range-end=stop",
                   "--force-overwrite=false", "--wait=primary", "--kill=none",
                   "--output=" + str(run / "trace"), *command]
    request["command"] = command
    request["environment"] = {
        key: value for key, value in env.items()
        if key.startswith(("TORCH", "TRITON", "CUDA", "CUBLAS", "NVIDIA_TF32"))
        or key in compilation["request"]["environment"]
        or key.startswith("KDA_CAPTURE_") or key == "LD_PRELOAD"
    }
    write_json(run / "request.json", request)
    replay_command = ["env", *(f"{key}={value}" for key, value in request["environment"].items()), *command]
    (run / "command.sh").write_text("#!/bin/sh\n" + shlex.join(replay_command) + "\n")
    print(json.dumps({"run_dir": str(run), "command": command}), flush=True)
    result = supervise(command, env=env, cwd=run, gpu=gpu,
                       log_path=run / "worker.log", telemetry_path=run / "gpu_telemetry.jsonl")
    result["runtime_source_after"] = source_identity(source_root)
    if result["runtime_source_after"] != runtime_source:
        result.update(returncode=1, error="Runtime source changed during measurement")
    write_json(run / "exit.json", result)
    print(json.dumps({"run_dir": str(run), **result}), flush=True)
    return result["returncode"]


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--worker":
        worker(sys.argv[2])
    else:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--compile-run", required=True, type=Path)
        parser.add_argument("--output-root", required=True, type=Path)
        parser.add_argument("--gpu", required=True)
        parser.add_argument("--max-idle-memory-mib", type=int, required=True)
        parser.add_argument("--source-root", type=Path,
                            help="Relocated immutable compilation source; contents must match the package")
        parser.add_argument("--mode", choices=("benchmark", "profile", "capture"), default="benchmark")
        parser.add_argument("--capture-spec", type=Path, help="Reviewed kernel name, argument roles/byte sizes, and capture library")
        parser.add_argument("--validation-contract", type=Path,
                            help="Explicit independent reference or user-authorized baseline-equivalence diagnostic contract")
        parser.add_argument("--k", type=int, default=10)
        parser.add_argument("--rounds", type=int, default=3)
        parser.add_argument("--samples", type=int, default=20)
        parser.add_argument("--capture-seconds", type=float, default=1.0, help="Seconds per mode")
        parser.add_argument("--nsys", default="nsys")
        sys.exit(launch(parser.parse_args()))
