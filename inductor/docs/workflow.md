# Model optimization loop

## Start with a task contract

Read the repository skill, this guide and the supplied task context before GPU
work. Request only missing scenario information: model and companion inputs,
execution location/interpreter, source/runtime or adapter roots, hardware type,
authorized GPU, idle memory ceiling, representative workload, accuracy reference
and tolerances, target modes and promotion criteria. Hardware knowledge must come
from the user and be verified live; do not inherit a card or host from prior runs.
Read available files before asking for information they already contain.

Copy `configs/task_context.example.json` into task-local storage and fill its null
fields from the supplied context. `prepared_export_eager` is the standard reference
created by the compiler. `independent` requires a supplied reference/validator and
an adapter to the input contract; this runner does not invent one.
`baseline_equivalence` requires the user's explicit instruction for this task.
TF32 in the compiler profile does not authorize ignoring a numerical failure.

Use `task_workspace.py init` to create:

```text
agent_space/<model>_<task-date>/
  task.json
  state.json
  checkpoints.jsonl             # created by the first checkpoint
  HANDOFF.md
  docs/{draft,plan}.md
  metadata/compile_profiles.json
  runs/<unique-run-id>/
  profile/
  reports/
  kernels/<kernel-identity>/
    docs/{draft,plan}.md
    runs/
    outputs/
    profile/
    benchmark.csv
    candidates.jsonl
```

`docs/draft.md` defines ABI, representative inputs, precision, validation and
acceptance. `docs/plan.md` turns them into commands and evidence gates. Do not
start an unattended loop until the input/validation contract is executable.

## Execute the outer loop

| Stage | Required evidence before advancing |
|---|---|
| Inspect | Export versus AOT package, archive/companion hashes, runtime compatibility, ordered ABI and input provenance |
| Baseline compile | Requested/effective profile, immutable source/runtime identity, logs, exit, package and saved input/reference hashes |
| Validate and measure | Structure/finite/numerical gates, Graph consistency, unprofiled ordinary and Graph raw samples, GPU admission/telemetry |
| Rank hotspots | Separate valid trace; complete blocks; top-k summed device time, share, counts, launch dimensions and call-site mapping |
| Kernel loop | True boundary data, bottleneck evidence, mechanism hypotheses, correctness and isolated timing of accepted/rejected candidates |
| Template integration | Narrow justified guards, fallback, actual candidate selection, generated code/resource and non-target audit |
| Model acceptance | Fresh package, original reference/inputs, comparable paired or three-way timing, promotion or rejection with reasons |

Reuse completed stages only after checking their artifact hashes and contract.
Keep the original baseline immutable. Store the last accepted package as a
separate incumbent. A new candidate must improve the intended model metric over
the incumbent; report its change against the original baseline separately.

Start top-k selection with the largest aggregate device cost across **all**
backends. An easy elementwise kernel must not displace a substantially larger
GEMM/BMM merely because its source is easier to extract. Kernel name and grid can
group several shapes/call sites; disaggregate before choosing an implementation.
The sum of concurrent kernel durations is not critical-path latency.

Each kernel has its own KDA contract, draft/plan and candidate ledger. State a
bottleneck, then predict what a structural change should improve: bytes moved,
reuse, dependent loads, parallelism, synchronization, epilogue traffic or launch
count. Tile tuning is one search axis, not the entire method. Use the existing
KernelWiki when relevant to the supplied hardware; use NCU for a concrete metric
question. Read the corresponding skill only when using it.

Integrate the implementation into a template/lowering with guards that encode
its actual assumptions: hardware capability, dtype, precision, layout, alignment,
shape/dynamic constraints and fusion/epilogue behavior. Retain the old fallback.
Do not edit only generated cache files and call that a compiler optimization.
Recompile from the original model; prove routing and selection in the artifact.
Use compiler tracing only when code/boundary inspection cannot answer provenance
or selection questions.

A correct isolated speedup can lose in the model through register pressure,
packing, cache changes, altered fusion, redundant launches or unrelated autotune
choices. Report each rejected candidate. Finishing with no promotion is a valid
outcome when supported by the measured alternatives; do not lower the validation
standard to force a win.

## Preserve and resume evidence

Checkpoint after each meaningful stage with artifact paths and hashes, current
phase, last decision and exact next action. Register `baseline` once and use
`incumbent` for later accepted candidates. Register manifests, reference contracts,
ranking, code audits and final reports separately. `status` checks registered
files only; verify the nested package/input/reference hashes with the run tools.

Record active execution host, supervisor/worker PID, command, start time and log
path in a task-local execution note, then register it. Before restarting a quiet
compile, inspect those processes and the exit record. A checkpoint does not mean
its worker stopped. At a context boundary, update HANDOFF.md with outstanding
hypotheses, known failures, next commands and pending environment constraints.

Execution may be local or remote. Edit in the agreed source checkout and preserve
unrelated dirty changes. If hosts differ, sync only named source/task files without
deleting destination files, verify SHA256, and import the intended source before
Torch. Stage large artifacts, verify complete files, then move them into place.
An interrupted tar stream may leave truncated files; verify the inventory and
transfer only missing/mismatched content. Hostnames, tunnels and recovery commands
belong in task context, never in shared scripts. Keep heavy packages/profiles at
the execution location when possible; retain lightweight evidence locally.

## Final task report

Include the original hotspot ranks and shares, exact shapes/call sites, baseline
and incumbent identities, bottleneck evidence, mechanism changes, validation
scope, per-kernel timing, whole-model ordinary/Graph results, raw sample links,
non-target changes, template conditions and remaining limitations. State whether
numbers describe a pure kernel, a fused region or the complete model. Give the
next agent a verified checkpoint and a concrete next step, not a chat transcript.
