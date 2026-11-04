"""Extract embedded Triton source and literal launch metadata from an AOT wrapper."""

import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import textwrap


def extract(wrapper, output):
    records = {}
    pattern = r"async_compile\.triton\('([^']+)',\s*'''(.*?)''',"
    for match in re.finditer(pattern, wrapper.read_text(), re.DOTALL):
        name, raw = match.groups()
        source = textwrap.dedent(raw).lstrip()
        tree = ast.parse(source)
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
        if len(functions) != 1 or functions[0].name != name:
            raise ValueError(f"Unexpected function boundary: {name}")
        function = functions[0]
        record = {"name": name, "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
                  "wrapper_line": wrapper.read_text().count("\n", 0, match.start()) + 1,
                  "arguments": [arg.arg for arg in function.args.args], "constants": {},
                  "decorators": [ast.unparse(node) for node in function.decorator_list],
                  "launch_metadata": {},
                  "standalone_scope": "Kernel body only. Replay decorator compiler options, constexprs, actual wrapper launch, qualifiers, and stream behavior explicitly; not a drop-in equivalent runner."}
        for decorator in function.decorator_list:
            if isinstance(decorator, ast.Call):
                for keyword in decorator.keywords:
                    record["launch_metadata"][keyword.arg or "**kwargs"] = ast.unparse(keyword.value)
                    if keyword.arg in ("num_stages", "num_warps"):
                        record[keyword.arg] = ast.literal_eval(keyword.value)
                    if keyword.arg == "triton_meta" and isinstance(keyword.value, ast.Dict):
                        for key, value in zip(keyword.value.keys, keyword.value.values):
                            if ast.literal_eval(key) == "signature":
                                record["signature"] = ast.literal_eval(value)
        for node in function.body:
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                record["constants"][node.target.id] = ast.unparse(node.value)
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                key = node.targets[0].id
                if key in ("M", "N", "K", "BATCH") or key.startswith("stride_"):
                    record[key] = ast.unparse(node.value)
        directory = output / name
        directory.mkdir(parents=True, exist_ok=False)
        (directory / "generated.py").write_text(source)
        function.decorator_list = [decorator for decorator in function.decorator_list
                                   if isinstance(decorator, ast.Attribute) and ast.unparse(decorator) == "triton.jit"]
        if not function.decorator_list:
            raise ValueError(f"Missing triton.jit decorator: {name}")
        (directory / "standalone.py").write_text(ast.unparse(tree) + "\n")
        record["standalone_sha256"] = hashlib.sha256((directory / "standalone.py").read_bytes()).hexdigest()
        records[name] = record
    if not records:
        raise ValueError("No embedded Triton kernels found")
    (output / "manifest.json").write_text(json.dumps({"wrapper": str(wrapper.resolve()),
        "wrapper_sha256": hashlib.sha256(wrapper.read_bytes()).hexdigest(), "kernels": records}, indent=2) + "\n")
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wrapper", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    records = extract(args.wrapper, args.output)
    print(json.dumps({"kernels": len(records), "templates": sum("_tem_" in name for name in records), "output": str(args.output)}))
