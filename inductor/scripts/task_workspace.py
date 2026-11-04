"""Initialize, checkpoint, and inspect a resumable model optimization workspace."""
import argparse
import datetime
import json
import os
from pathlib import Path
import shutil

from compile_aot import PROFILES, file_hash, resolve_tolerances


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def initialize(root, context):
    for key in ("model_name", "model_path", "execution", "accuracy"):
        if key not in context or not context[key]:
            raise ValueError(f"Missing task context: {key}")
    for key in ("model_name", "model_path"):
        if not isinstance(context[key], str) or not context[key].strip():
            raise ValueError(f"Missing task context: {key}")
    for key in ("location", "python", "gpu", "hardware"):
        if not isinstance(context["execution"].get(key), str) or not context["execution"][key].strip():
            raise ValueError(f"Missing execution context: {key}")
    ceiling = context["execution"].get("max_idle_memory_mib")
    if type(ceiling) is not int or ceiling < 0:
        raise ValueError("max_idle_memory_mib must be a nonnegative integer")
    accuracy = context["accuracy"]
    resolve_tolerances(accuracy.get("rtol"), accuracy.get("atol"))
    if accuracy.get("reference_policy") not in ("prepared_export_eager", "independent", "baseline_equivalence"):
        raise ValueError("Specify the reference policy for this task")
    if accuracy["reference_policy"] == "baseline_equivalence" and not accuracy.get("authorization"):
        raise ValueError("Baseline equivalence requires this task's explicit user instruction")
    root.mkdir(parents=True, exist_ok=False)
    for folder in ("docs", "metadata", "runs", "profile", "reports", "kernels"):
        (root / folder).mkdir()
    profile = root / "metadata" / "compile_profiles.json"
    shutil.copy2(PROFILES, profile)
    (root / "task.json").write_text(json.dumps({"schema_version": 1, "created_at": now(), **context}, indent=2) + "\n")
    state = {"schema_version": 1, "phase": "initialized", "updated_at": now(),
             "next_action": "Inspect input archive and verify execution/runtime/profile compatibility before compilation.",
             "decision": "No GPU work has been started.",
             "artifacts": {"context": {"path": "task.json", "sha256": file_hash(root / "task.json")},
                           "profiles": {"path": str(profile.relative_to(root)), "sha256": file_hash(profile)}}}
    save(root, state)
    (root / "docs" / "draft.md").write_text("# Model optimization contract\n\nRead ../task.json. Record representative inputs, ABI, reference, target modes, and acceptance criteria before execution.\n")
    (root / "docs" / "plan.md").write_text("# Experiment plan\n\nInspect -> baseline -> measure/profile -> ranked targets -> KDA kernel loops -> guarded templates -> full-model validation and comparison -> retain/reject.\n\nRecord one hypothesis, predicted evidence, and rollback condition per experiment.\n")
    (root / "HANDOFF.md").write_text("# Resume this task\n\nRead task.json and state.json, then run the repository's inductor/scripts/task_workspace.py status --task-root with this directory. Check registered artifact hashes. Read docs/draft.md and docs/plan.md.\n\nBefore launching work, verify host, interpreter, selected GPU and hardware against task.json; inspect any active supervisor/worker and its logs. A checkpoint is not proof a process has stopped. Do not repeat a compile merely because its log is quiet.\n\nKeep the original baseline immutable. Register the current accepted candidate separately as incumbent. Read the repository skill skills/inductor-model-optimization/SKILL.md and inductor/docs/workflow.md for the full protocol.\n")
    return state


def save(root, state):
    temporary = root / "state.json.tmp"
    temporary.write_text(json.dumps(state, indent=2) + "\n")
    temporary.replace(root / "state.json")


def checkpoint(root, phase, next_action, decision, artifacts):
    state = json.loads((root / "state.json").read_text())
    for assignment in artifacts:
        name, path = assignment.split("=", 1)
        path = Path(path).resolve(strict=True)
        record = {"path": os.path.relpath(path, root), "sha256": file_hash(path)}
        old = state["artifacts"].get(name)
        if name in ("baseline", "context", "profiles") and old and old != record:
            raise ValueError(f"Immutable checkpoint artifact cannot be replaced: {name}")
        state["artifacts"][name] = record
    state.update(phase=phase, next_action=next_action, decision=decision, updated_at=now())
    save(root, state)
    with (root / "checkpoints.jsonl").open("a") as stream:
        stream.write(json.dumps(state) + "\n")
    return state


def status(root):
    state = json.loads((root / "state.json").read_text())
    checks = {}
    for name, record in state["artifacts"].items():
        path = root / record["path"]
        checks[name] = "missing" if not path.is_file() else (
            "verified" if file_hash(path) == record["sha256"] else "changed")
    return {"state": state, "artifact_checks": checks,
            "verified": all(value == "verified" for value in checks.values()),
            "scope": "Registered files only; verify artifacts referenced inside manifests and live processes separately"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init")
    init.add_argument("--context", type=Path, required=True)
    for child in (init, sub.add_parser("status"), sub.add_parser("checkpoint")):
        child.add_argument("--task-root", type=Path, required=True)
        if child.prog.endswith("checkpoint"):
            child.add_argument("--phase", required=True)
            child.add_argument("--next", dest="next_action", required=True)
            child.add_argument("--decision", required=True)
            child.add_argument("--artifact", action="append", default=[], metavar="NAME=FILE")
    args = parser.parse_args()
    root = args.task_root.resolve()
    if args.command == "init":
        result = initialize(root, json.loads(args.context.read_text()))
    elif args.command == "checkpoint":
        result = checkpoint(root, args.phase, args.next_action, args.decision, args.artifact)
    else:
        result = status(root)
    print(json.dumps(result, indent=2))
    return 0 if result.get("verified", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
