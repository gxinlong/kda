"""Export source-bound autotune choices in the original compilation environment."""

import argparse
import json
from pathlib import Path
import sys

from compile_aot import file_hash, source_identity, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compile-run", required=True, type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(args.source_root.resolve()))
    import torch
    from torch._inductor.codecache import torch_key
    from torch._inductor.runtime.autotune_cache import AutotuneCache

    baseline = json.loads((args.compile_run / "manifest.json").read_text())
    source = source_identity(args.source_root.resolve())
    for key in ("commit", "status", "diff_sha256", "untracked_python_sha256", "untracked_template_sha256", "package_source_sha256", "kind"):
        if source.get(key, {}) != baseline["source"].get(key, {}):
            raise RuntimeError(f"Source differs from baseline: {key}")
    if file_hash(torch._C.__file__) != baseline["runtime"]["torch_c_extension_sha256"]:
        raise RuntimeError("Native runtime differs")
    cache = args.compile_run.resolve() / "cache"
    records = []
    for path in cache.rglob("*.py"):
        key = AutotuneCache._prepare_key(str(path))
        config = Path(AutotuneCache._make_local_cache_key(str(path.parent), key))
        if config.is_file():
            records.append({"source_path": str(path), "source_relative": str(path.relative_to(cache)),
                "source_sha256": file_hash(path), "config_path": str(config),
                "config_sha256": file_hash(config), "config": json.loads(config.read_text())})
    if not records:
        raise RuntimeError("No source-bound autotune configurations found")
    manifest = {"baseline_package_sha256":baseline["package"]["sha256"],
                "baseline_source":source,"torch_key":torch_key().hex(),
                "native_sha256":file_hash(torch._C.__file__),"records":records}
    write_json(args.output, manifest)
    print(json.dumps({"records":len(records),"output":str(args.output),"torch_key":manifest["torch_key"]}))


if __name__ == "__main__":
    main()
