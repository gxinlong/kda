"""Rank AOT kernels from complete, synchronized K-forward NVTX blocks."""

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import sqlite3
import statistics

from compile_aot import file_hash, write_json


def analyze(run, top_k=10):
    result = json.loads((run / "result.json").read_text())
    exit_status = json.loads((run / "exit.json").read_text())
    if result["status"] != "succeeded" or exit_status["returncode"]:
        raise RuntimeError("Profile worker or supervisor failed")
    db = sqlite3.connect(f"file:{run / 'trace.sqlite'}?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    schema = {row[0]: [dict(column) for column in db.execute(f'PRAGMA table_info("{row[0]}")')]
              for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    write_json(run / "sqlite_schema.json", schema)
    required = {
        "CUPTI_ACTIVITY_KIND_KERNEL": {"start", "end", "globalPid", "deviceId", "demangledName", "graphNodeId", "gridX", "gridY", "gridZ", "blockX", "blockY", "blockZ", "streamId"},
        "NVTX_EVENTS": {"start", "end", "text", "globalTid"},
        "PROCESSES": {"globalPid", "pid"},
        "TARGET_INFO_GPU": {"uuid", "cuDevice"},
        "StringIds": {"id", "value"},
        "DIAGNOSTIC_EVENT": {"severity"},
    }
    for table, columns in required.items():
        if not columns.issubset({column["name"] for column in schema.get(table, [])}):
            raise RuntimeError(f"Unsupported schema: {table}")
    pids = [dict(row) for row in db.execute(
        "SELECT p.globalPid,p.pid,COUNT(*) AS kernels FROM CUPTI_ACTIVITY_KIND_KERNEL k "
        "JOIN PROCESSES p ON p.globalPid=k.globalPid GROUP BY p.globalPid,p.pid")]
    if len(pids) != 1 or pids[0]["pid"] != result["pid"]:
        raise RuntimeError("Kernel activity does not belong exclusively to the measured worker")
    kernels = [dict(row) for row in db.execute(
        "SELECT k.*,s.value AS name FROM CUPTI_ACTIVITY_KIND_KERNEL k "
        "JOIN StringIds s ON s.id=k.demangledName ORDER BY k.start")]
    if {row["deviceId"] for row in kernels} != {0}:
        raise RuntimeError("Unexpected CUDA device")
    gpu_row = db.execute("SELECT * FROM TARGET_INFO_GPU WHERE uuid=?", (
        result["request"]["gpu"]["uuid"].removeprefix("GPU-"),)).fetchone()
    if gpu_row is None:
        raise RuntimeError("Selected GPU UUID missing from trace metadata")
    gpu = dict(gpu_row)
    if gpu["cuDevice"] != int(result["request"]["gpu"]["index"]):
        raise RuntimeError("Physical GPU mapping differs")
    diagnostics = [dict(row) for row in db.execute("SELECT * FROM DIAGNOSTIC_EVENT WHERE severity>=2")]
    report = {"source": str(run), "trace_sha256": file_hash(run / "trace.nsys-rep"),
              "sqlite_sha256": file_hash(run / "trace.sqlite"), "process": pids[0],
              "gpu": gpu, "diagnostics": diagnostics, "modes": {},
              "top_k": top_k,
              "limitations": "CUDA/NVTX trace only; kernel duration sum is not critical-path model latency; hot repeated inputs"}
    for mode in ("ordinary", "graph"):
        blocks = [dict(row) for row in db.execute(
            "SELECT start,end,text,globalTid FROM NVTX_EVENTS WHERE text LIKE ? ORDER BY start",
            (f"kda/{mode}/block/%",))]
        if len(blocks) < 3 or len(blocks) != result["profile_blocks"][mode] or any(b["end"] is None for b in blocks):
            raise RuntimeError("Missing complete synchronized workload blocks")
        grouped = defaultdict(list)
        counts = []
        identities = []
        graph_nodes = 0
        streams = set()
        first_forward_identity = None
        position_samples = defaultdict(list)
        stable_sequence = True
        for block in blocks:
            selected = [row for row in kernels if block["start"] <= row["start"] and row["end"] <= block["end"]]
            counts.append(len(selected))
            identity = Counter()
            for row in selected:
                dimensions = tuple(row[key] for key in ("gridX", "gridY", "gridZ", "blockX", "blockY", "blockZ"))
                identity[(row["name"], dimensions)] += 1
                grouped[(row["name"], dimensions)].append((row["end"] - row["start"]) / 1000)
                graph_nodes += bool(row["graphNodeId"])
                streams.add(row["streamId"])
            identities.append(identity)
            k = result["request"]["k"]
            if k <= 0 or not selected or len(selected) % k:
                raise RuntimeError("Kernel launches do not divide into complete forwards")
            per_forward = len(selected) // k
            for offset in range(0, len(selected), per_forward):
                forward = selected[offset:offset + per_forward]
                sequence = [(row["name"], tuple(row[key] for key in
                    ("gridX", "gridY", "gridZ", "blockX", "blockY", "blockZ"))) for row in forward]
                if first_forward_identity is None:
                    first_forward_identity = sequence
                stable_sequence &= sequence == first_forward_identity
                for ordinal, row in enumerate(forward, 1):
                    position_samples[ordinal].append((row["end"] - row["start"]) / 1000)
        if not counts[0] or len(set(counts)) != 1 or any(i != identities[0] for i in identities):
            raise RuntimeError("Kernel counts or launch identities vary between complete blocks")
        if mode == "graph" and graph_nodes != sum(counts):
            raise RuntimeError("Graph node coverage incomplete")
        forwards = len(blocks) * result["request"]["k"]
        total_us = sum(sum(values) for values in grouped.values())
        ranking = sorted(({
            "name": name, "grid": dimensions[:3], "block": dimensions[3:],
            "count": len(values), "total_us": sum(values),
            "mean_us": statistics.mean(values), "median_us": statistics.median(values),
            "us_per_forward": sum(values) / forwards,
            "kernel_time_fraction": sum(values) / total_us,
            "calls_per_forward": len(values) / forwards,
        } for (name, dimensions), values in grouped.items()), key=lambda row: -row["total_us"])
        cumulative = 0.0
        for rank, row in enumerate(ranking, 1):
            cumulative += row["kernel_time_fraction"]
            row.update(rank=rank, cumulative_kernel_time_fraction=cumulative)
        ordered = stable_sequence and len(streams) == 1
        by_position = []
        if ordered:
            for ordinal, values in position_samples.items():
                name, dimensions = first_forward_identity[ordinal - 1]
                by_position.append({"ordinal_1based":ordinal,"name":name,
                    "grid":dimensions[:3],"block":dimensions[3:],"count":len(values),
                    "mean_us":statistics.mean(values),"median_us":statistics.median(values),
                    "us_per_forward":sum(values)/forwards,"kernel_time_fraction":sum(values)/total_us})
            by_position.sort(key=lambda row:-row["us_per_forward"])
        report["modes"][mode] = {"blocks": blocks, "forwards": forwards,
            "kernel_counts": counts, "kernels_per_forward": counts[0] / result["request"]["k"],
            "graph_node_kernels": graph_nodes, "streams": sorted(streams),
            "kernel_us_per_forward": total_us / forwards, "ranking": ranking,
            "ordered_single_stream":ordered,"ranking_by_position":by_position,
            "position_scope":"One-based chronological device launch position in a forward; not proof of wrapper/operator provenance."}
        with (run / f"top{top_k}_{mode}.csv").open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(ranking[0]))
            writer.writeheader();writer.writerows(ranking[:top_k])
    write_json(run / "hotspots.json", report)
    lines = ["# Profile hotspot ranking", "", report["limitations"], ""]
    for mode, values in report["modes"].items():
        lines += [f"## {mode}", "", f"{len(values['blocks'])} complete blocks, {values['kernels_per_forward']:g} kernel launches per forward.", "",
                  "Ranking identity is kernel name plus launch dimensions.", "",
                  "| Rank | Kernel | Grid / block | Calls/forward | Mean us/call | us/forward | Kernel time share |", "|---:|---|---|---:|---:|---:|---:|"]
        for row in values["ranking"][:top_k]:
            lines.append(f"| {row['rank']} | {row['name']} | {row['grid']} / {row['block']} | {row['calls_per_forward']:g} | {row['mean_us']:.3f} | {row['us_per_forward']:.3f} | {row['kernel_time_fraction']:.2%} |")
        lines += ["",f"Top-{top_k} cumulative share: {sum(r['kernel_time_fraction'] for r in values['ranking'][:top_k]):.2%}."]
        if values["ordered_single_stream"]:
            lines += ["", "Stable individual launch positions (not yet operator provenance):", "",
                      "| Forward launch ordinal | Kernel | Mean us | Kernel time share |", "|---:|---|---:|---:|"]
            for row in values["ranking_by_position"][:top_k]:
                lines.append(f"| {row['ordinal_1based']} | {row['name']} | {row['mean_us']:.3f} | {row['kernel_time_fraction']:.2%} |")
        lines.append("")
    (run / "hotspots.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k must be positive")
    analyze(args.run.resolve(), args.top_k)
