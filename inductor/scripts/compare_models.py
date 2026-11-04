"""Paired ordinary/Graph timing of immutable baseline and candidate AOT packages."""

import argparse
import importlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time

from compile_aot import file_hash, source_identity, tensor_metadata, write_json, validate_reference_contract


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-run", type=Path, required=True)
    parser.add_argument("--candidate-run", type=Path, required=True)
    parser.add_argument("--incumbent-run", type=Path,
                        help="Optional prior accepted model; retain original baseline as the contract/runtime anchor")
    parser.add_argument("--validation-contract", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()
    if min(args.rounds, args.samples, args.k) <= 0:
        parser.error("rounds, samples and K must be positive")
    run = Path(os.environ["KDA_RUN_DIR"])
    baseline = json.loads((args.baseline_run / "manifest.json").read_text())
    candidate = json.loads((args.candidate_run / "manifest.json").read_text())
    incumbent = json.loads((args.incumbent_run / "manifest.json").read_text()) if args.incumbent_run else None
    contract = json.loads(args.validation_contract.read_text())
    validate_reference_contract(baseline, contract)
    if (candidate["status"] != "succeeded" or not candidate["validation"]["passed"]
            or candidate["saved_inputs"]["sha256"] != baseline["saved_inputs"]["sha256"]
            or contract["inputs_sha256"] != baseline["saved_inputs"]["sha256"]
            or candidate["reference"]["sha256"] != contract["reference"]["sha256"]):
        raise RuntimeError("Baseline/candidate validation contract differs")
    if incumbent is not None and (
            incumbent["status"] != "succeeded" or not incumbent["validation"]["passed"]
            or incumbent["saved_inputs"]["sha256"] != contract["inputs_sha256"]
            or incumbent["reference"]["sha256"] != contract["reference"]["sha256"]):
        raise RuntimeError("Incumbent does not satisfy the original baseline contract")
    # The installed environment has a .pth hook that prepends its editable tree.
    # Restore the explicitly selected immutable tree before importing Torch.
    if os.environ.get("PYTHONPATH"):
        sys.path[:0] = os.environ["PYTHONPATH"].split(os.pathsep)
    import torch
    from torch.utils import _pytree

    torch.set_num_threads(1)
    torch.cuda.set_device(0)
    current_source = source_identity(Path(torch.__file__).resolve().parents[1])
    # Both binaries run in the baseline's clean source checkout and identical wheel.
    # A relocated, immutable worktree may have another root path.
    for key in ("commit", "status", "diff_sha256", "untracked_python_sha256", "untracked_template_sha256", "package_source_sha256", "kind"):
        if current_source.get(key, {}) != baseline["source"].get(key, {}):
            raise RuntimeError(f"Runtime source differs from baseline: {key}")
    runtime = {"source": current_source, "torch": torch.__version__, "torch_file": torch.__file__,
               "c_extension_sha256": file_hash(torch._C.__file__)}
    for manifest in ([baseline, candidate] + ([incumbent] if incumbent is not None else [])):
        if runtime["c_extension_sha256"] != manifest["runtime"]["torch_c_extension_sha256"]:
            raise RuntimeError("Torch native runtime differs")
    for key, value in contract["torch_backends"].items():
        target = torch.backends
        parts = key.split(".")
        for part in parts[:-1]:
            target = getattr(target, part)
        setattr(target, parts[-1], value)
    request = baseline["request"]
    if request["loader"] == "torchfourier":
        fourier = Path(request["fourier_root"])
        sys.path[:0] = [str(fourier), str(fourier / "fourier")]
        adapter = importlib.import_module("fourier.aot_runtime")
        if request["allow_triton_version_override"]:
            adapter.AOT_TRITON_VERSION = adapter.installed_triton_version()
        ops = importlib.import_module("fourier.aot_ops")
        model_path = Path(request["input"])
        ops.register_required_ops(json.loads((model_path.parent / "aotconfig.json").read_text()),
                                  artifact_path=model_path, strict=True)
    artifacts = {"baseline": baseline["package"], "candidate": candidate["package"],
                 "inputs": baseline["saved_inputs"], "reference": contract["reference"]}
    if incumbent is not None:
        artifacts["incumbent"] = incumbent["package"]
    for item in artifacts.values():
        if file_hash(item["path"]) != item["sha256"]:
            raise RuntimeError("Immutable artifact changed")
    models = {name: torch._inductor.aoti_load_package(artifacts[name]["path"], run_single_threaded=True)
              for name in ("baseline", "incumbent", "candidate") if name in artifacts}
    inputs, kwargs = torch.load(artifacts["inputs"]["path"], map_location="cuda:0", weights_only=False)
    reference = torch.load(artifacts["reference"]["path"], map_location="cuda:0", weights_only=False)

    def clone(value):
        return _pytree.tree_map(lambda x: x.detach().clone() if isinstance(x, torch.Tensor) else x, value)

    def compare(value):
        torch.testing.assert_close(value, reference, rtol=contract["rtol"], atol=contract["atol"], equal_nan=False)
        errors = []
        for actual, expected in zip(_pytree.tree_leaves(value), _pytree.tree_leaves(reference)):
            if isinstance(actual, torch.Tensor):
                if not bool(torch.isfinite(actual).all()):
                    raise RuntimeError("Non-finite output")
                errors.append((actual.double() - expected.double()).abs().max().item() if actual.numel() else 0.0)
        return {"heads": len(errors), "max_abs_error": max(errors, default=0.0)}

    result = {"status": "running", "runtime": runtime, "artifacts": artifacts,
              "validation_basis": contract, "inputs": tensor_metadata((inputs, kwargs)),
              "protocol": {"rounds":args.rounds,"samples_per_round":args.samples,"k":args.k,
                  "scope":"GPU resident repeated inputs; paired K-forward block averages, not request latency; no profiler"},
              "validation": {}}
    graphs, output_lifetimes = {}, {}
    with torch.inference_mode():
        for name, model in models.items():
            ordinary = clone(model(*inputs, **kwargs))
            torch.cuda.synchronize()
            result["validation"][name] = {"ordinary": compare(ordinary)}
            for _ in range(30):
                model(*inputs, **kwargs)
            torch.cuda.synchronize()
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(5):
                    model(*inputs, **kwargs)
            torch.cuda.current_stream().wait_stream(side)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=side):
                outputs = [model(*inputs, **kwargs) for _ in range(args.k)]
            graph.replay()
            torch.cuda.synchronize()
            result["validation"][name]["graph"] = compare(clone(outputs[-1]))
            graphs[name], output_lifetimes[name] = graph, outputs
            for _ in range(5):
                graph.replay()
            torch.cuda.synchronize()
        write_json(run / "result.json", result)
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record(); end.record(); end.synchronize()
        samples = []
        for round_index in range(args.rounds):
            modes = ("ordinary", "graph") if round_index % 2 == 0 else ("graph", "ordinary")
            for mode in modes:
                for sample_index in range(args.samples):
                    names = tuple(models)
                    if (round_index + sample_index) % 2:
                        names = names[::-1]
                    if incumbent is not None:
                        shift = (round_index + sample_index) % len(names)
                        names = names[shift:] + names[:shift]
                    for name in names:
                        torch.cuda.synchronize()
                        wall = time.perf_counter()
                        start.record()
                        if mode == "ordinary":
                            for _ in range(args.k):
                                models[name](*inputs, **kwargs)
                        else:
                            graphs[name].replay()
                        end.record(); end.synchronize()
                        samples.append({"round":round_index,"sample":sample_index,"order":len(samples),
                            "mode":mode,"model":name,"k":args.k,
                            "event_ms":start.elapsed_time(end)/args.k,
                            "wall_ms":(time.perf_counter()-wall)*1000/args.k})
        write_json(run / "samples.json", samples)
        summary = {}
        for mode in ("ordinary", "graph"):
            summary[mode] = {}
            for name in models:
                rows = [r for r in samples if r["mode"]==mode and r["model"]==name]
                stats = {"samples":len(rows),"forwards":sum(r["k"] for r in rows)}
                for metric in ("event_ms", "wall_ms"):
                    values = sorted(r[metric] for r in rows)
                    stats[metric] = {"median":statistics.median(values), "p95_nearest_rank":values[math.ceil(.95*len(values))-1],
                        "round_medians":{str(i):statistics.median(r[metric] for r in rows if r["round"]==i) for i in range(args.rounds)}}
                summary[mode][name] = stats
            b = summary[mode]["baseline"]["event_ms"]["median"]
            c = summary[mode]["candidate"]["event_ms"]["median"]
            summary[mode]["speedup"] = b/c
            summary[mode]["time_reduction_fraction"] = 1-c/b
            if incumbent is not None:
                old = summary[mode]["incumbent"]["event_ms"]["median"]
                summary[mode]["speedup_vs_incumbent"] = old/c
                summary[mode]["time_reduction_vs_incumbent_fraction"] = 1-c/old
        result.update(status="succeeded",summary=summary)
        write_json(run / "result.json", result)
        print(json.dumps({"status": "succeeded", "summary":summary}),flush=True)


if __name__ == "__main__":
    main()
