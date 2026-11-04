"""External GPU admission and worker supervision (standard library only)."""

import csv
import io
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time


def snapshot(gpu):
    fields = "index,uuid,name,memory.used,utilization.gpu,temperature.gpu,power.draw,clocks.sm,clocks.mem,pstate"
    command = ["nvidia-smi", "-i", gpu, "--format=csv,noheader,nounits"]
    raw_gpu = subprocess.check_output(command + ["--query-gpu=" + fields], text=True)
    raw_apps = subprocess.check_output(
        command + ["--query-compute-apps=pid,process_name,used_memory"], text=True
    )
    rows = list(csv.reader(io.StringIO(raw_gpu)))
    if len(rows) != 1:
        raise RuntimeError(f"Expected one GPU, got {raw_gpu!r}")
    device = dict(zip(fields.split(","), (v.strip() for v in rows[0])))
    processes = []
    for row in csv.reader(io.StringIO(raw_apps)):
        if row:
            processes.append(dict(zip(("pid", "name", "memory_mib"), (v.strip() for v in row))))
    return {"time_unix": time.time(), "device": device, "processes": processes}


def idle_errors(sample, max_memory_mib):
    errors = []
    if sample["processes"]:
        errors.append("foreign compute processes are present")
    if float(sample["device"]["utilization.gpu"]) != 0:
        errors.append("GPU utilization is nonzero")
    if float(sample["device"]["memory.used"]) > max_memory_mib:
        errors.append("GPU memory exceeds the configured idle ceiling")
    return errors


def admit(gpu, path, max_memory_mib):
    samples = []
    for index in range(2):
        if index:
            time.sleep(2)
        samples.append(snapshot(gpu))
        Path(path).write_text(json.dumps(samples, indent=2) + "\n")
    errors = [reason for sample in samples for reason in idle_errors(sample, max_memory_mib)]
    if errors:
        raise RuntimeError(f"GPU admission failed: {sorted(set(errors))}; see {path}")
    uuids = {sample["device"]["uuid"] for sample in samples}
    if len(uuids) != 1:
        raise RuntimeError("GPU identity changed during admission")
    return samples[-1]


def process_group_pids(group):
    """Include descendants that a profiler moves into another process group."""
    records = {}
    for proc in Path("/proc").glob("[0-9]*"):
        try:
            pid = int(proc.name)
            aliases = {str(pid)}
            parent = None
            for line in (proc / "status").read_text().splitlines():
                if line.startswith("NSpid:"):
                    aliases.update(line.split()[1:])
                elif line.startswith("PPid:"):
                    parent = int(line.split()[1])
            # This host exposes the initial-namespace PID in the scheduler header.
            match = re.search(r"\((\d+), #threads:", (proc / "sched").read_text().splitlines()[0])
            if match:
                aliases.add(match.group(1))
            records[pid] = (parent, os.getpgid(pid), aliases)
        except (ProcessLookupError, PermissionError, FileNotFoundError):
            continue
    owned = {pid for pid, (_, pgid, _) in records.items() if pgid == group}
    owned.add(group)
    while True:
        descendants = {pid for pid, (parent, _, _) in records.items() if parent in owned}
        if descendants.issubset(owned):
            break
        owned.update(descendants)
    pids = set()
    for pid in owned:
        if pid in records:
            pids.update(records[pid][2])
    return pids


def supervise(command, *, env, cwd, gpu, log_path, telemetry_path):
    """Stop only our new process group when admission evidence becomes invalid."""
    error = None
    with Path(log_path).open("w") as log, Path(telemetry_path).open("w") as telemetry:
        worker = subprocess.Popen(
            command, env=env, cwd=cwd, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        Path(cwd, "worker.json").write_text(json.dumps({"pid": worker.pid, "command": command}) + "\n")
        try:
            while True:
                sample = snapshot(gpu)
                allowed = process_group_pids(worker.pid)
                sample["allowed_pids"] = sorted(allowed)
                sample["unresolved_pids"] = [
                    row["pid"] for row in sample["processes"] if row["pid"] not in allowed
                ]
                telemetry.write(json.dumps(sample) + "\n")
                telemetry.flush()
                if sample["unresolved_pids"]:
                    raise RuntimeError(
                        "Foreign or unmapped GPU PIDs: " + repr(sample["unresolved_pids"])
                        + "; this worker is stopped; namespace identity must be resolved"
                    )
                result = worker.poll()
                if result is not None:
                    break
                time.sleep(1)
        except BaseException as exc:
            error = repr(exc)
            if worker.poll() is None:
                os.killpg(worker.pid, signal.SIGTERM)
                try:
                    worker.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(worker.pid, signal.SIGKILL)
                    worker.wait()
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
        result = worker.wait()
        telemetry.write(json.dumps({"final": snapshot(gpu), "worker_returncode": result, "error": error}) + "\n")
    return {"returncode": result if error is None else 1, "worker_returncode": result, "error": error}
