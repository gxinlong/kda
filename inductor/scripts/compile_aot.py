from __future__ import annotations

import argparse
import datetime
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import time
import traceback
import uuid

from gpu_guard import admit, supervise

ROOT = Path(__file__).resolve().parent
PROFILES = ROOT.parent / "configs" / "compile_profiles.json"


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=str) + "\n")


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command_output(args):
    return subprocess.check_output(args, text=True).strip()


def source_identity(root):
    root = Path(root).resolve()
    git_root = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "--show-toplevel"],
        capture_output=True, text=True,
    ) if shutil.which("git") else None
    if git_root is None or git_root.returncode or Path(git_root.stdout.strip()).resolve() != root:
        package = root / "torch"
        if not (package / "__init__.py").is_file():
            raise ValueError(f"Not a Torch package root: {root}")
        files = sorted(p for p in package.rglob("*") if p.suffix in (".py", ".jinja"))
        hashes = {str(p.relative_to(root)): file_hash(p) for p in files}
        return {"root": str(root), "kind": "installed_package", "commit": None,
                "status": "installed_package", "diff_sha256": None,
                "package_source_sha256": hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()}
    untracked = subprocess.check_output(
        ["git", "-C", str(root), "ls-files", "-z", "--others", "--exclude-standard", "--", "*.py"]
    ).decode().split("\0")
    templates = subprocess.check_output(
        ["git", "-C", str(root), "ls-files", "-z", "--others", "--exclude-standard", "--", "*.jinja"]
    ).decode().split("\0")
    return {
        "root": str(root),
        "commit": command_output(["git", "-C", str(root), "rev-parse", "HEAD"]),
        "status": command_output(
            ["git", "-C", str(root), "status", "--short", "--untracked-files=no"]
        ),
        "diff_sha256": hashlib.sha256(
            subprocess.check_output(["git", "-C", str(root), "diff", "HEAD"])
        ).hexdigest(),
        "untracked_python_sha256": {
            name: file_hash(root / name) for name in untracked if name
        },
        "untracked_template_sha256": {
            name: file_hash(root / name) for name in templates if name
        },
    }


def resolve_profile(path, name=None, overrides=()):
    """Resolve a versioned preset without importing Torch or selecting hardware."""
    profiles = json.loads(Path(path).read_text())
    name = name or profiles["default_profile"]
    profile = profiles["profiles"][name]
    configs = dict(profile["inductor_configs"])
    for assignment in overrides:
        key, value = assignment.split("=", 1)
        configs[key] = json.loads(value)
    return profiles, name, profile, configs


def resolve_tolerances(rtol, atol, contract=None):
    if contract is not None:
        for key, value in (("rtol", rtol), ("atol", atol)):
            if value is not None and value != contract[key]:
                raise ValueError(f"{key} differs from the validation contract")
        rtol, atol = contract["rtol"], contract["atol"]
    if rtol is None or atol is None or not all(0 <= x < float("inf") for x in (rtol, atol)):
        raise ValueError("Provide finite nonnegative --rtol and --atol for this model, or a validation contract")
    return rtol, atol


def validate_reference_contract(baseline, contract):
    """Bind a reference to immutable baseline inputs; never infer accuracy approval."""
    if contract["package_sha256"] != baseline["package"]["sha256"]:
        raise ValueError("Validation contract identifies another package")
    if contract["inputs_sha256"] != baseline["saved_inputs"]["sha256"]:
        raise ValueError("Validation contract identifies different inputs")
    resolve_tolerances(None, None, contract)
    if contract["kind"] == "baseline_equivalence":
        if not contract.get("diagnostic_timing_authorization"):
            raise ValueError("Baseline-equivalence timing requires explicit diagnostic authorization")
    elif contract["kind"] == "validated_reference":
        if not contract.get("passed"):
            raise ValueError("Independent validation contract did not pass")
        if (baseline["status"] != "succeeded" or not baseline["validation"]["passed"]
                or contract["reference"]["sha256"] != baseline["reference"]["sha256"]
                or contract["torch_backends"] != baseline["request"]["torch_backends"]
                or any(contract[k] != baseline["request"][k] for k in ("rtol", "atol"))):
            raise ValueError("Validated reference does not match successful baseline validation")
    else:
        raise ValueError("Unsupported validation contract kind")


def tensor_metadata(value):
    import torch
    from torch.utils import _pytree

    leaves, spec = _pytree.tree_flatten(value)
    metadata = []
    for tensor in leaves:
        if isinstance(tensor, torch.Tensor):
            metadata.append(
                {
                    "shape": list(tensor.shape),
                    "stride": list(tensor.stride()),
                    "storage_offset": tensor.storage_offset(),
                    "dtype": str(tensor.dtype),
                    "device": str(tensor.device),
                }
            )
        else:
            metadata.append({"type": type(tensor).__name__, "value": repr(tensor)})
    return {"structure": str(spec), "leaves": metadata}


def load_model(request):
    import torch

    model = Path(request["input"])
    loader = request["loader"]
    if loader == "exported":
        from torch.export.passes import move_to_device_pass

        ep = move_to_device_pass(torch.export.load(model), "cuda:0")
        return ep, {"loader": loader}

    fourier = Path(request["fourier_root"])
    sys.path[:0] = [str(fourier), str(fourier / "fourier")]
    runtime = importlib.import_module("fourier.aot_runtime")
    pinned_triton = runtime.AOT_TRITON_VERSION
    installed_triton = runtime.installed_triton_version()
    if installed_triton != pinned_triton:
        if not request["allow_triton_version_override"]:
            raise RuntimeError(
                f"Adapter pins Triton {pinned_triton}; installed {installed_triton}. "
                "Use an explicit --allow-triton-version-override only for a validated adapter."
            )
        runtime.AOT_TRITON_VERSION = installed_triton
    compiler = importlib.import_module("fourier.aot_compile")
    ops = importlib.import_module("fourier.aot_ops")
    externalize = importlib.import_module("constant2placeholder").const2placeholder
    model_config = json.loads((model.parent / "aotconfig.json").read_text())
    registration = ops.register_required_ops(
        model_config, artifact_path=model, strict=True
    )
    ep = torch.export.load(model)
    original_inputs = tensor_metadata(ep.example_inputs)
    original_constraints = {str(k): str(v) for k, v in ep.range_constraints.items()}
    exported, names, weights, shapes, name_map = externalize(
        ep, model_config["dynamic"], model_config["max_batch_size"]
    )
    cuda_ep, _ = compiler.export_program_to_cuda(
        exported, model_config["dynamic"], shapes
    )
    schema_path = model.parent / "model.schema.json"
    schema_check = {"status": "not_available"}
    if schema_path.is_file():
        schema = json.loads(schema_path.read_text())
        expected = [row["name"] for row in schema["inputs"]]
        actual = [node.name for node in cuda_ep.graph.nodes if node.op == "placeholder"]
        if actual != expected or len(weights) != schema["weights_len"]:
            raise RuntimeError(
                f"Exported input ABI differs from {schema_path}: "
                f"weights={len(weights)}, expected={schema['weights_len']}; "
                f"input_names={actual}, expected={expected}"
            )
        schema_check = {"status": "passed", "sha256": file_hash(schema_path)}
    lowerings = ops.register_compile_lowerings(cuda_ep)
    return cuda_ep, {
        "loader": loader,
        "source": source_identity(fourier),
        "helper_hashes": {
            name: file_hash(fourier / "fourier" / name)
            for name in ("aot_compile.py", "constant2placeholder.py", "aot_runtime.py")
        },
        "model_config": model_config,
        "original_inputs": original_inputs,
        "original_constraints": original_constraints,
        "weight_names": names,
        "weight_count": len(weights),
        "schema_check": schema_check,
        "name_map": name_map,
        "registration": registration,
        "lowerings": lowerings,
        "triton_import_compatibility": {
            "pinned": pinned_triton,
            "used": runtime.AOT_TRITON_VERSION,
            "scope": "process-local import compatibility; no source edits",
        },
        "example_input_policy": "original ExportedProgram examples, unchanged",
    }


def worker(request_path):
    request = json.loads(Path(request_path).read_text())
    run = Path(request["run_dir"])
    manifest = {"status": "running", "request": request, "pid": os.getpid()}
    started = time.monotonic()
    try:
        if request.get("source_root"):
            # Editable-install .pth hooks can precede PYTHONPATH.
            sys.path.insert(0, request["source_root"])
        import torch
        import torch._inductor.config as config
        import triton

        for name in ("aoti_compile_and_package", "aoti_load_package"):
            if not hasattr(torch._inductor, name):
                raise RuntimeError(f"Torch {torch.__version__} lacks required API {name}")
        source_root = request.get("source_root")
        if source_root and Path(torch.__file__).resolve().parents[1] != Path(source_root):
            raise RuntimeError(f"Expected Torch from {source_root}, got {torch.__file__}")
        if torch.cuda.device_count() != 1:
            raise RuntimeError("Expected exactly one visible CUDA device")
        torch.cuda.set_device(0)
        manifest["source"] = source_identity(Path(torch.__file__).resolve().parents[1])
        source_diff = (b"" if manifest["source"].get("kind") == "installed_package" else
                       subprocess.check_output(["git", "-C", str(Path(torch.__file__).resolve().parents[1]), "diff", "HEAD"]))
        (run / "source.diff").write_bytes(source_diff)
        manifest["runtime"] = {
            "python": sys.executable,
            "torch": torch.__version__,
            "torch_binary_git_version": torch.version.git_version,
            "torch_file": torch.__file__,
            "torch_c_extension": os.path.realpath(torch._C.__file__),
            "torch_c_extension_sha256": file_hash(torch._C.__file__),
            "inductor_file": torch._inductor.__file__,
            "cuda_build": torch.version.cuda,
            "triton": triton.__version__,
            "gpu": str(torch.cuda.get_device_properties(0)),
        }
        if request.get("baseline_autotune_configs"):
            from torch._inductor.codecache import torch_key
            from torch._inductor.runtime.autotune_cache import AutotuneCache

            info = request["baseline_autotune_configs"]
            if file_hash(info["path"]) != info["sha256"]:
                raise RuntimeError("Baseline autotune configuration manifest changed")
            saved = json.loads(Path(info["path"]).read_text())
            if saved["baseline_package_sha256"] != request["validation_contract"]["package_sha256"]:
                raise RuntimeError("Autotune configurations belong to another baseline")
            if saved["native_sha256"] != manifest["runtime"]["torch_c_extension_sha256"]:
                raise RuntimeError("Autotune native runtime changed")
            copied = []
            for record in saved["records"]:
                if file_hash(record["source_path"]) != record["source_sha256"] or file_hash(record["config_path"]) != record["config_sha256"]:
                    raise RuntimeError("Baseline autotune source/configuration changed")
                relative = Path(record["source_relative"])
                if relative.is_absolute() or ".." in relative.parts:
                    raise ValueError("Invalid relative autotune source path")
                filename = run / "cache" / relative
                key = AutotuneCache._prepare_key(str(filename))
                destination = Path(AutotuneCache._make_local_cache_key(str(filename.parent), key))
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(record["config_path"], destination)
                copied.append({"source_sha256":record["source_sha256"],"source_relative":str(relative),
                               "destination":str(destination),"config_sha256":record["config_sha256"]})
            manifest["autotune_config_replay"] = {"baseline_torch_key":saved["torch_key"],
                "current_torch_key":torch_key().hex(),"records":copied,
                "policy":"Only identical generated-source filenames can hit these records; Inductor still verifies configs_hash. No source, code, binary or graph cache is imported."}
        for key, value in request["torch_backends"].items():
            parent, field = key.rsplit(".", 1)
            target = torch.backends
            for part in parent.split("."):
                target = getattr(target, part)
            setattr(target, field, value)
        manifest["tf32"] = {
            "matmul": torch.backends.cuda.matmul.allow_tf32,
            "cudnn": torch.backends.cudnn.allow_tf32,
        }
        tracked_keys = list(
            dict.fromkeys(
                [
                    *request["inductor_configs"],
                    *request["inherit_inductor_configs"],
                    "combo_kernel_max_num_nodes",
                    "memory_planning",
                    "memory_pool",
                    "freezing",
                    "cta_fusion.enabled",
                    "region_combo.cta",
                    "region_combo.template",
                    "benchmark_combo_kernel",
                ]
            )
        )
        missing = [key for key in request["inductor_configs"] if not hasattr(config, key)]
        if missing:
            raise RuntimeError(f"Requested profile options unsupported by Torch {torch.__version__}: {missing}. "
                               "Choose an explicit compatible profile; no options were silently dropped.")
        with config.patch(request["inductor_configs"]):
            manifest["effective_configs"] = {
                key: getattr(config, key, "unsupported") for key in tracked_keys
            }
        write_json(run / "manifest.json", manifest)
        print("Loading and preparing ExportedProgram", flush=True)
        with torch.no_grad():
            ep, adapter = load_model(request)
            args, kwargs = ep.example_inputs
            with config.patch(request["inductor_configs"]):
                manifest["effective_configs"] = {
                    key: getattr(config, key, "unsupported") for key in tracked_keys
                }
            manifest["adapter"] = adapter
            manifest["inputs"] = tensor_metadata((args, kwargs))
            manifest["constraints"] = {
                str(key): str(value) for key, value in ep.range_constraints.items()
            }
            graph_text = str(ep.graph)
            (run / "graph.txt").write_text(graph_text)
            manifest["graph_sha256"] = hashlib.sha256(graph_text.encode()).hexdigest()
            from torch.utils import _pytree

            if request.get("comparison_baseline"):
                baseline_info = request["comparison_baseline"]
                if file_hash(baseline_info["manifest_path"]) != baseline_info["manifest_sha256"]:
                    raise RuntimeError("Baseline manifest changed")
                baseline = json.loads(Path(baseline_info["manifest_path"]).read_text())
                if baseline["graph_sha256"] != manifest["graph_sha256"] or baseline["inputs"] != manifest["inputs"]:
                    raise RuntimeError("Prepared graph or input ABI differs from baseline")
                manifest["saved_inputs"] = baseline["saved_inputs"]
                manifest["reference"] = request["validation_contract"]["reference"]
                for artifact in (manifest["saved_inputs"], manifest["reference"]):
                    if file_hash(artifact["path"]) != artifact["sha256"]:
                        raise RuntimeError("Baseline input/reference artifact changed")
                args, kwargs = torch.load(manifest["saved_inputs"]["path"], map_location="cuda:0", weights_only=False)
                ep._example_inputs = (args, kwargs)
                reference = torch.load(manifest["reference"]["path"], map_location="cuda:0", weights_only=False)
                manifest["validation_basis"] = request["validation_contract"]
            else:
                example_path = run / "artifacts" / "example_inputs.pt"
                torch.save((args, kwargs), example_path)
                manifest["saved_inputs"] = {
                    "path": str(example_path), "sha256": file_hash(example_path),
                    "policy": "Exact prepared CUDA examples; torch serialization preserves storage relationships",
                }
                print("Evaluating prepared ExportedProgram eager reference", flush=True)
                eager = ep.module()
                reference = _pytree.tree_map(
                    lambda value: value.detach().clone() if isinstance(value, torch.Tensor) else value,
                    eager(*args, **kwargs),
                )
                torch.cuda.synchronize()
                reference_path = run / "artifacts" / "reference_outputs.pt"
                torch.save(reference, reference_path)
                manifest["reference"] = {
                    "path": str(reference_path), "sha256": file_hash(reference_path),
                    "scope": "Eager prepared ExportedProgram; not independent validation of export transformations",
                }
                del eager
            write_json(run / "manifest.json", manifest)
            package = run / "artifacts" / "model.pt2"
            compile_started = time.monotonic()
            print(f"Compiling to {package}", flush=True)
            torch._inductor.aoti_compile_and_package(
                ep,
                package_path=str(package),
                inductor_configs=dict(request["inductor_configs"]),
            )
            manifest["compile_seconds"] = time.monotonic() - compile_started
            manifest["package"] = {
                "path": str(package),
                "sha256": file_hash(package),
                "bytes": package.stat().st_size,
            }
            manifest["status"] = "compiled"
            write_json(run / "manifest.json", manifest)
            print("Loading AOT package and running original example inputs", flush=True)
            compiled = torch._inductor.aoti_load_package(
                str(package), run_single_threaded=True
            )
            outputs = compiled(*args, **kwargs)
            torch.cuda.synchronize()
            leaves, output_spec = _pytree.tree_flatten(outputs)
            expected, reference_spec = _pytree.tree_flatten(reference)
            comparisons = []
            if output_spec != reference_spec:
                comparisons.append({"passed": False, "error": "Output TreeSpec differs"})
            else:
                for index, (actual, target) in enumerate(zip(leaves, expected)):
                    item = {"index": index}
                    try:
                        torch.testing.assert_close(actual, target, rtol=request["rtol"], atol=request["atol"], equal_nan=False)
                        item["passed"] = True
                    except AssertionError as exc:
                        item.update(passed=False, error=str(exc))
                    if isinstance(actual, torch.Tensor):
                        item["finite"] = bool(torch.isfinite(actual).all())
                        item["shape"] = list(actual.shape)
                        item["dtype"] = str(actual.dtype)
                        if actual.numel():
                            item["min"] = actual.min().item()
                            item["max"] = actual.max().item()
                            item["zero_or_one_fraction"] = ((actual == 0) | (actual == 1)).float().mean().item()
                        if isinstance(target, torch.Tensor) and actual.shape == target.shape and actual.numel():
                            error = (actual.double() - target.double()).abs()
                            item["max_abs_error"] = error.max().item()
                            item["max_relative_error"] = (error / target.double().abs().clamp_min(request["atol"])).max().item()
                    comparisons.append(item)
            validation = {
                "package_load": "passed",
                "forward": "passed",
                "input_policy": "original ExportedProgram example inputs",
                "outputs": tensor_metadata(outputs),
                "finite": [
                    bool(torch.isfinite(value).all())
                    if isinstance(value, torch.Tensor)
                    else None
                    for value in leaves
                ],
                "reference": manifest["reference"],
                "rtol": request["rtol"], "atol": request["atol"],
                "comparisons": comparisons,
                "passed": bool(comparisons) and all(row["passed"] and row.get("finite", True) for row in comparisons),
            }
            write_json(run / "validation.json", validation)
            manifest["validation"] = validation
            if not validation["passed"]:
                raise RuntimeError("Compiled/reference numerical comparison failed; see validation.json")
            manifest["source_after"] = source_identity(
                Path(torch.__file__).resolve().parents[1]
            )
            if manifest["source_after"] != manifest["source"]:
                raise RuntimeError("PyTorch source changed during compilation")
            manifest["status"] = "succeeded"
    except BaseException:
        manifest["status"] = "failed"
        manifest["error"] = traceback.format_exc()
        raise
    finally:
        manifest["elapsed_seconds"] = time.monotonic() - started
        write_json(run / "manifest.json", manifest)


def launch(args):
    profiles_path = args.profiles.resolve(strict=True)
    profiles, profile_name, profile, configs = resolve_profile(profiles_path, args.profile, args.overrides)
    contract = json.loads(args.validation_contract.read_text()) if args.validation_contract else None
    args.rtol, args.atol = resolve_tolerances(args.rtol, args.atol, contract)
    for name in (args.model_name, args.label):
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
            raise ValueError(f"Invalid directory name: {name!r}")
    model = args.input.resolve(strict=True)
    loader = args.loader
    if loader == "auto":
        loader = "torchfourier" if (model.parent / "aotconfig.json").is_file() else "exported"
    if loader == "torchfourier" and (args.fourier_root is None or not args.fourier_root.is_dir()):
        raise ValueError("The torchfourier loader requires an explicit existing --fourier-root")
    if args.allow_triton_version_override and loader != "torchfourier":
        raise ValueError("--allow-triton-version-override applies only to the torchfourier loader")
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    run = args.output_root.resolve() / f"{stamp}_{args.label}_{uuid.uuid4().hex[:8]}"
    run.mkdir(parents=True)
    for name in ("artifacts", "cache", "triton_cache", "tmp", "scripts"):
        (run / name).mkdir()
    print(json.dumps({"run_dir": str(run), "phase": "gpu_admission"}), flush=True)
    try:
        sample = admit(args.gpu, run / "gpu_admission.json", args.max_idle_memory_mib)
    except Exception:
        write_json(run / "exit.json", {"returncode": 1, "status": "blocked_gpu_admission", "error": traceback.format_exc()})
        raise
    gpu_uuid = sample["device"]["uuid"]
    gpu = sample["device"]
    environment = {
        key: value.replace("<work_dir>", str(run)) if value is not None else None
        for key, value in profile["environment"].items()
    }
    environment.update(
        CUDA_VISIBLE_DEVICES=gpu_uuid,
        TORCHINDUCTOR_CACHE_DIR=str(run / "cache"),
        TRITON_CACHE_DIR=str(run / "triton_cache"),
        TMPDIR=str(run / "tmp"),
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONUNBUFFERED="1",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
    )
    if args.source_root:
        source_root = args.source_root.resolve(strict=True)
        if not (source_root / "torch" / "__init__.py").is_file():
            raise ValueError(f"Not a PyTorch source tree: {source_root}")
        python_paths = [str(source_root)]
        environment["PYTHONPATH"] = os.pathsep.join(python_paths)
    child_env = dict(os.environ)
    for key, value in environment.items():
        if value is None:
            child_env.pop(key, None)
        else:
            child_env[key] = value
    request = {
        "input": str(model),
        "input_sha256": file_hash(model),
        "model_name": args.model_name,
        "profile": profile_name,
        "label": args.label,
        "loader": loader,
        "fourier_root": str(args.fourier_root.resolve()) if args.fourier_root else None,
        "source_root": str(args.source_root.resolve()) if args.source_root else None,
        "run_dir": str(run),
        "allow_triton_version_override": args.allow_triton_version_override,
        "rtol": args.rtol,
        "atol": args.atol,
        "gpu": gpu,
        "inductor_configs": configs,
        "profile_source": {"path": str(profiles_path), "sha256": file_hash(profiles_path),
                           "baseline_id": profiles.get("baseline_id"), "overrides": args.overrides},
        "inherit_inductor_configs": profiles["inherit_inductor_configs"],
        "torch_backends": profile["torch_backends"],
        "environment": environment,
        "effective_environment": {
            key: value
            for key, value in child_env.items()
            if key.startswith(("TORCH", "TRITON", "CUDA", "CUBLAS", "NVIDIA_TF32"))
            or key in environment
        },
    }
    if args.baseline_run:
        if args.validation_contract is None:
            raise ValueError("--baseline-run requires --validation-contract")
        baseline_path = args.baseline_run.resolve() / "manifest.json"
        baseline = json.loads(baseline_path.read_text())
        contract = json.loads(args.validation_contract.read_text())
        validate_reference_contract(baseline, contract)
        if (baseline["request"]["input_sha256"] != request["input_sha256"]
                or contract["torch_backends"] != request["torch_backends"]):
            raise ValueError("Baseline comparison contract does not match this experiment")
        request["comparison_baseline"] = {"manifest_path": str(baseline_path), "manifest_sha256": file_hash(baseline_path)}
        request["validation_contract"] = contract
        if args.seed_timing_cache:
            source = args.baseline_run.resolve() / "cache" / "cache"
            if not source.is_dir():
                raise ValueError("Baseline persistent timing cache is missing")
            request["seed_timing_cache"] = {"source": str(source), "files": {
                str(path.relative_to(source)): file_hash(path) for path in source.rglob("*") if path.is_file()}}
            shutil.copytree(source, run / "cache" / "cache")
            # Fusion decisions have their own content-addressed timing memos.
            # Copy only numeric timing records; never generated source or binaries.
            cache_root = args.baseline_run.resolve() / "cache"
            fusion_records = {}
            for path in cache_root.rglob("*.kernel_perf"):
                relative = path.relative_to(cache_root)
                destination = run / "cache" / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, destination)
                fusion_records[str(relative)] = file_hash(path)
            request["seed_timing_cache"]["fusion_perf_files"] = fusion_records
            config_records = {}
            for path in cache_root.rglob("*.best_config"):
                relative = path.relative_to(cache_root)
                destination = run / "cache" / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, destination)
                config_records[str(relative)] = file_hash(path)
            request["seed_timing_cache"]["kernel_config_files"] = config_records
    elif args.validation_contract or args.seed_timing_cache:
        raise ValueError("Baseline contract/cache options require --baseline-run")
    if args.baseline_autotune_configs:
        if not args.baseline_run:
            raise ValueError("--baseline-autotune-configs requires --baseline-run")
        request["baseline_autotune_configs"] = {"path":str(args.baseline_autotune_configs.resolve()),
                                               "sha256":file_hash(args.baseline_autotune_configs)}
    for name in ("compile_aot.py", "gpu_guard.py"):
        shutil.copy2(ROOT / name, run / "scripts" / name)
    shutil.copy2(profiles_path, run / "scripts" / "compile_profiles.json")
    request["companion_sha256"] = {
        name: file_hash(model.parent / name)
        for name in ("aotconfig.json", "model.schema.json", "weights.bin")
        if (model.parent / name).is_file()
    }
    request["driver"] = command_output(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader", "-i", gpu_uuid])
    request["script_sha256"] = file_hash(run / "scripts" / "compile_aot.py")
    write_json(run / "request.json", request)
    (run / "command.sh").write_text(
        "#!/bin/sh\nset -eu\n" + shlex.join([sys.executable, str(ROOT / "compile_aot.py"), *sys.argv[1:]]) + "\n"
    )
    print(json.dumps({"run_dir": str(run), "profile": profile_name, "gpu": gpu}), flush=True)
    command = [
        sys.executable, "-B", str(run / "scripts" / "compile_aot.py"),
        "--worker", str(run / "request.json"),
    ]
    # Hashing and copying can take time; recheck admission immediately before launch.
    admit(gpu_uuid, run / "gpu_admission_before_launch.json", args.max_idle_memory_mib)
    result = supervise(command, env=child_env, cwd=run, gpu=gpu_uuid,
                       log_path=run / "compile.log", telemetry_path=run / "gpu_telemetry.jsonl")
    write_json(run / "exit.json", result)
    print(json.dumps({"run_dir": str(run), **result}), flush=True)
    return result["returncode"]


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--worker":
        worker(Path(sys.argv[2]))
        return 0
    parser = argparse.ArgumentParser(description="Compile an ExportedProgram with an explicit AOTInductor profile.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--gpu", required=True, help="Physical GPU index or UUID; must be idle.")
    parser.add_argument("--profiles", type=Path, default=PROFILES, help="Versioned profile JSON; default is the bundled baseline")
    parser.add_argument("--profile", help="Profile name from --profiles")
    parser.add_argument("--label", default="default")
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=JSON")
    parser.add_argument("--loader", choices=("auto", "exported", "torchfourier"), default="auto")
    parser.add_argument("--fourier-root", type=Path, help="Required only for the optional torchfourier adapter")
    parser.add_argument("--source-root", type=Path, help="Use an isolated PyTorch source tree with wheel runtime links.")
    parser.add_argument("--allow-triton-version-override", action="store_true")
    parser.add_argument("--rtol", type=float, help="Model-specific tolerance; required without a validation contract")
    parser.add_argument("--atol", type=float, help="Model-specific tolerance; required without a validation contract")
    parser.add_argument("--baseline-run", type=Path, help="Reuse an immutable baseline's exact prepared inputs")
    parser.add_argument("--validation-contract", type=Path, help="Explicit authorization and compiled baseline reference")
    parser.add_argument("--seed-timing-cache", action="store_true", help="Copy only persistent timing records, never graph/code caches")
    parser.add_argument("--baseline-autotune-configs", type=Path,
                        help="Source-bound baseline config manifest; migrate only config records to the current torch_key")
    parser.add_argument("--max-idle-memory-mib", type=int, required=True,
                        help="Idle driver-memory ceiling; process and utilization checks also apply.")
    return launch(parser.parse_args())


if __name__ == "__main__":
    sys.exit(main())
