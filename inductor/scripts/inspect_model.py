"""Inspect input identity and exported graph without importing Torch."""

import argparse
import hashlib
import json
from pathlib import Path
import zipfile


def file_identity(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def inspect(path):
    result = {"model": file_identity(path), "adjacent": {}}
    for name in ("aotconfig.json", "model.schema.json", "weights.bin"):
        companion = path.parent / name
        if companion.is_file():
            result["adjacent"][name] = file_identity(companion)
            if name.endswith(".json"):
                result["adjacent"][name]["content"] = json.loads(companion.read_text())
    with zipfile.ZipFile(path) as archive:
        result["archive_members"] = [{"name": member.filename, "bytes": member.file_size} for member in archive.infolist()]
        candidates = [
            name for name in archive.namelist()
            if name.endswith("serialized_exported_program.json")
            or ("/models/" in "/" + name and name.endswith(".json"))
        ]
        result["exported_graphs"] = []
        for member in candidates:
            raw = archive.read(member)
            data = json.loads(raw)
            if "graph_module" not in data:
                continue
            graph = data["graph_module"]["graph"]
            targets = {}
            for node in graph["nodes"]:
                target = str(node["target"])
                targets[target] = targets.get(target, 0) + 1
            result["exported_graphs"].append({
                "member": member, "sha256": hashlib.sha256(raw).hexdigest(),
                "schema_version": data.get("schema_version"),
                "torch_version": data.get("torch_version"),
                "node_count": len(graph["nodes"]), "targets": targets,
                "range_constraints": data.get("range_constraints"),
                "inputs": graph["inputs"], "outputs": graph["outputs"],
            })
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = inspect(args.input.resolve(strict=True))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"output": str(args.output), "model": result["model"], "graphs": len(result["exported_graphs"])}))
