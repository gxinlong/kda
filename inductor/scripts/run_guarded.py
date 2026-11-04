"""Run an isolated kernel experiment with the model workflow's GPU admission."""

import argparse
import datetime
import json
import os
from pathlib import Path
import shlex
import shutil
import sys
import uuid

from compile_aot import file_hash, write_json
from gpu_guard import admit, supervise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--max-idle-memory-mib", type=int, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--source-root", type=Path, help="Optional explicitly selected Torch source/package root")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a worker command is required after --")
    if not args.label or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-" for c in args.label):
        parser.error("--label must be a simple directory component")
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    run = args.output_root.resolve() / f"{stamp}_{args.label}_{uuid.uuid4().hex[:8]}"
    run.mkdir(parents=True)
    sources = {}
    for argument in command:
        source = Path(argument)
        if argument.endswith(".py") and source.is_file():
            (run / "scripts").mkdir(exist_ok=True)
            for sibling in source.parent.glob("*.py"):
                shutil.copy2(sibling, run / "scripts" / sibling.name)
                sources[str(sibling)] = file_hash(sibling)
            break
    gpu = admit(args.gpu, run / "gpu_admission.json", args.max_idle_memory_mib)["device"]["uuid"]
    env = dict(os.environ)
    env.update(CUDA_VISIBLE_DEVICES=gpu,
               KDA_RUN_DIR=str(run), TRITON_CACHE_DIR=str(run / "triton_cache"),
               TORCHINDUCTOR_CACHE_DIR=str(run / "inductor_cache"))
    if args.source_root:
        env["PYTHONPATH"] = str(args.source_root.resolve(strict=True))
    write_json(run / "request.json", {"command": command, "gpu": gpu, "source_sha256": sources,
               "environment": {k: env.get(k) for k in ("CUDA_VISIBLE_DEVICES", "PYTHONPATH", "KDA_RUN_DIR", "TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR")}})
    (run / "command.sh").write_text(shlex.join(command) + "\n")
    print(json.dumps({"run_dir": str(run)}), flush=True)
    result = supervise(command, env=env, cwd=run, gpu=gpu,
                       log_path=run / "worker.log", telemetry_path=run / "gpu_telemetry.jsonl")
    result["source_sha256_after"] = {
        name: file_hash(name) if Path(name).is_file() else None for name in sources
    }
    if result["source_sha256_after"] != sources:
        result.update(returncode=1, error="Experiment script sources changed during execution")
    write_json(run / "exit.json", result)
    print(json.dumps({"run_dir": str(run), **result}), flush=True)
    return result["returncode"]


if __name__ == "__main__":
    sys.exit(main())
