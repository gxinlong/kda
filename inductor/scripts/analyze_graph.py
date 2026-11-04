"""Read serialized export graph scopes and value edges without importing Torch."""
import argparse
from collections import Counter
import json
from pathlib import Path
import zipfile

from inspect_model import file_identity


def references(value):
    """Yield typed value references, never treating constants/metadata as edges."""
    if isinstance(value, list):
        for child in value:
            yield from references(child)
    elif isinstance(value, dict):
        for kind, child in value.items():
            if kind == "as_graph":
                continue  # Nested graph values belong to a separate scope.
            if kind in ("as_tensor", "as_custom_obj", "as_token"):
                yield child["name"]
            elif kind in ("as_tensors", "as_nested_tensors"):
                def tensors(items):
                    for item in items:
                        if isinstance(item, list):
                            yield from tensors(item)
                        else:
                            yield item["name"]
                yield from tensors(child)
            elif kind in ("as_sym_int", "as_sym_float", "as_sym_bool"):
                if "as_name" in child:
                    yield child["as_name"]
            elif kind in ("as_sym_ints", "as_sym_floats", "as_sym_bools"):
                for item in child:
                    if "as_name" in item:
                        yield item["as_name"]
            elif kind in ("arg", "as_optional_tensor", "as_optional_tensors", "as_string_to_argument"):
                if kind == "as_string_to_argument":
                    for item in child.values():
                        yield from references(item)
                else:
                    yield from references(child)
            elif kind.startswith("as_") and kind not in {
                "as_none", "as_int", "as_ints", "as_float", "as_floats", "as_string", "as_strings",
                "as_scalar_type", "as_memory_format", "as_layout", "as_device", "as_bool", "as_bools",
                "as_operator", "as_complex", "as_int_lists", "as_float_lists",
            }:
                raise ValueError(f"Unsupported serialized argument variant: {kind}; review the export schema")


def nested_graphs(value, path=""):
    if isinstance(value, dict):
        for key, child in value.items():
            location = path + "/" + key
            if key == "as_graph":
                yield location + "/" + child["name"], child["graph"]
            else:
                yield from nested_graphs(child, location)
    elif isinstance(value, list):
        for i, child in enumerate(value):
            yield from nested_graphs(child, path + f"/{i}")


def graph_scope(graph, scope):
    producers = {}
    edges, unresolved, nodes, children = [], [], [], []
    for index, value in enumerate(graph["inputs"]):
        for name in references(value):
            producers[name] = f"input:{index}"
    for index, node in enumerate(graph["nodes"]):
        identity = f"node:{index}"
        for port, argument in enumerate(node.get("inputs", [])):
            for name in references(argument):
                edge = {"value": name, "producer": producers.get(name), "consumer": identity, "input_port": port}
                (edges if name in producers else unresolved).append(edge)
        outputs = []
        for port, value in enumerate(node.get("outputs", [])):
            for name in references(value):
                if name in producers:
                    raise ValueError(f"Duplicate value {scope}:{name}; schema requires review")
                producers[name] = f"{identity}:output:{port}"
                outputs.append(name)
        nodes.append({"id": identity, "name": node.get("name"), "target": node["target"],
                      "inputs": node.get("inputs", []), "outputs": node.get("outputs", []),
                      "output_values": outputs, "metadata": node.get("metadata", {})})
        for location, child in nested_graphs(node.get("inputs", [])):
            children.extend(graph_scope(child, scope + "/" + identity + location))
    for port, value in enumerate(graph["outputs"]):
        for name in references(value):
            edge = {"value": name, "producer": producers.get(name), "consumer": f"output:{port}"}
            (edges if name in producers else unresolved).append(edge)
    record = {"scope": scope, "node_count": len(nodes), "targets": dict(Counter(n["target"] for n in nodes)),
              "inputs": graph["inputs"], "outputs": graph["outputs"], "nodes": nodes,
              "edges": edges, "unresolved_references": unresolved,
              "value_metadata": {k: v for k, v in graph.items() if k.endswith("_values")}}
    return [record, *children]


def analyze(path):
    exports = []
    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            if not (name.endswith("serialized_exported_program.json") or ("/models/" in "/" + name and name.endswith(".json"))):
                continue
            data = json.loads(archive.read(name))
            if "graph_module" not in data:
                continue
            module = data["graph_module"]
            exports.append({"member": name, "schema_version": data.get("schema_version"),
                            "range_constraints": data.get("range_constraints"),
                            "signature": module.get("signature"), "module_call_graph": module.get("module_call_graph"),
                            "scopes": graph_scope(module["graph"], name)})
    if not exports:
        raise ValueError("No serialized ExportedProgram found; an AOT package is not an export graph")
    return {"input": file_identity(path), "exports": exports,
            "scope": "Serialized value edges, not storage alias analysis or compiler fusion decisions. Review schema and unresolved references."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.input)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
