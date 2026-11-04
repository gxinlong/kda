# Tool recipes

Run from the KDA repository root on the appropriate host. All paths below are
shell variables supplied by the current task; none is a machine/model default.
Use absolute paths because guarded workers change their working directory.
`PYTHON` is the supplied execution interpreter, `KDA` the synchronized repository,
`TASK` the task workspace, `EXPORT` the input export, `MODEL` its task label,
`GPU_UUID` the verified authorized device, `IDLE_MIB` the observed idle ceiling,
and `RTOL`/`ATOL` the agreed model tolerances. Set `PYTORCH` only when intentionally
using a compatible source overlay. The JSON context records these choices; tools
still take explicit flags so the executed command is reviewable.

## Initialize and inspect (no Torch or GPU required)

Fill `inductor/configs/task_context.example.json` in a task-local `CONTEXT` file.
Set execution location to the user-supplied local/remote description, hardware to
the user-supplied type, and accuracy reference policy to an agreed value. Null
required fields are rejected. The initializer does not connect to a remote host,
select a card or launch work.

```sh
"$PYTHON" "$KDA/inductor/scripts/task_workspace.py" init \
  --context "$CONTEXT" --task-root "$TASK"
"$PYTHON" "$KDA/inductor/scripts/inspect_model.py" \
  --input "$EXPORT" --output "$TASK/metadata/model.json"
"$PYTHON" "$KDA/inductor/scripts/analyze_graph.py" \
  --input "$EXPORT" --output "$TASK/metadata/export_graph.json"
```

If the input is already an AOT package, graph analysis correctly fails without a
serialized export. The bundled benchmark runner consumes a compilation manifest,
not an arbitrary package path. To measure an externally produced package, first
write a task-local adapter/manifest with package/source/runtime identity, ordered
prepared inputs, reference and validation evidence. Do not recompile merely to
make a package fit the runner, and do not fabricate a successful validation field.

## Compile the baseline

Verify compatibility in [compilation](compilation.md), including branch-specific
environment options, before this command. The standard loader uses export example
inputs; confirm that they represent the requested workload. A different input
preparation requires an explicit adapter and a new recorded contract.

```sh
"$PYTHON" "$KDA/inductor/scripts/compile_aot.py" \
  --input "$EXPORT" --model-name "$MODEL" --loader exported \
  --gpu "$GPU_UUID" --max-idle-memory-mib "$IDLE_MIB" \
  --output-root "$TASK/runs" --label baseline \
  --profiles "$TASK/metadata/compile_profiles.json" --profile default \
  --rtol "$RTOL" --atol "$ATOL"
```

Add `--source-root "$PYTORCH"` for an explicitly selected overlay. For the supported
optional adapter, replace `--loader exported` with `--loader torchfourier` and
supply `--fourier-root "$ADAPTER"`; review compatibility before optionally enabling
the process-local Triton pin override. No adapter path is inferred.

Read the emitted `run_dir`, `exit.json`, logs and `manifest.json`; bind `BASELINE`
to that exact run. A created package does not mean validation passed. After a
successful numerical check, create the reusable reference contract:

```sh
"$PYTHON" "$KDA/inductor/scripts/validation_contract.py" \
  --baseline-run "$BASELINE" --output "$TASK/metadata/reference_contract.json"
```

For a known mismatch, retain the failure. Only if the user explicitly accepts
baseline equivalence for this task, capture a cloned output of the immutable
compiled baseline on **its exact saved inputs**, same custom-op registration,
`run_single_threaded=True`, backend flags and runtime. Synchronize and save the
full output pytree before workspace reuse. Record capture command, package/input
hashes and numerical policy. Then use `validation_contract.py --reference` with
that saved output file, `--authorization` with the user's instruction, and explicit
`--rtol`/`--atol`. This records a declared reference; it does not certify that an
arbitrary output file was produced by the package. Review that provenance first.
An external independent validator requires its own documented evidence/adapter;
the standard contract builder only exports the compiler's successful reference or
explicit baseline equivalence.

## Measure, then profile separately

After successful baseline validation, no extra contract flag is needed for its
own benchmark. For explicitly authorized baseline-equivalence diagnosis, pass
`--validation-contract "$CONTRACT"` to both commands below.

```sh
"$PYTHON" "$KDA/inductor/scripts/benchmark_model.py" \
  --compile-run "$BASELINE" --output-root "$TASK/runs" \
  --gpu "$GPU_UUID" --max-idle-memory-mib "$IDLE_MIB" --mode benchmark
"$PYTHON" "$KDA/inductor/scripts/benchmark_model.py" \
  --compile-run "$BASELINE" --output-root "$TASK/profile" \
  --gpu "$GPU_UUID" --max-idle-memory-mib "$IDLE_MIB" \
  --mode profile --nsys "$NSYS"
```

Bind `PROFILE_RUN` to the profile command's emitted run. Review worker/tool exit,
warnings and trace duration. Export SQLite using that Nsight version:

```sh
"$NSYS" export --type sqlite --output "$PROFILE_RUN/trace.sqlite" \
  "$PROFILE_RUN/trace.nsys-rep"
"$PYTHON" "$KDA/inductor/scripts/analyze_profile.py" "$PROFILE_RUN" --top-k 10
```

Read `hotspots.json`, `hotspots.md`, mode-specific top-k CSV and schema evidence.
Share means the fraction of summed device kernel time in complete blocks, not
end-to-end latency. Inspect shape/call-site grouping before selecting targets.
Unsupported schemas, incomplete ranges or missing Graph nodes require diagnosis;
do not silently fall back to all events in the trace.

## Kernel subloops and optional capture

Create the KDA subtask layout from [workflow](workflow.md). Extract the exact
wrapper member from the package using the archive's inspected member name, then:

```sh
"$PYTHON" "$KDA/inductor/scripts/extract_kernels.py" \
  --wrapper "$WRAPPER" --output "$KERNEL_TASK/outputs/extracted"
```

`generated.py` is the exact implementation. `standalone.py` removes the Inductor
heuristic decorator and is only a starting point: explicitly restore ABI,
constexprs, compiler/backend options and wrapper launch settings in the harness.
Use captured operands and reference outputs before searching.

Build optional Linux capture helpers using the supplied CUDA headers/compiler:

```sh
"$CXX" -std=c++17 -O2 -fPIC -shared -I"$CUDA_INCLUDE" \
  "$KDA/inductor/csrc/capture_kernel.cpp" -ldl -o "$CAPTURE_LIBRARY"
```

For the BMM boundary use `capture_aten_bmm.cpp` instead, keeping its adjacent
`capture_bmm_config.h`. Supply a reviewed capture JSON as described in
[provenance](provenance.md), then run `benchmark_model.py` with `--mode capture`
and `--capture-spec "$CAPTURE_SPEC"` plus the same compilation/GPU/output flags.
The run must capture the target, validate the instrumented full model, and save
capture hashes. It is not a performance run.

Launch standalone candidates under the common GPU guard:

```sh
"$PYTHON" "$KDA/inductor/scripts/run_guarded.py" \
  --gpu "$GPU_UUID" --max-idle-memory-mib "$IDLE_MIB" \
  --output-root "$KERNEL_TASK/runs" --label "$CANDIDATE_LABEL" \
  -- "$PYTHON" "$KERNEL_HARNESS" "$CANDIDATE_FILE"
```

Provide absolute harness paths and explicit source root when needed. The guard
sets `KDA_RUN_DIR`, saves source hashes and telemetry, and checks for foreign work.
The harness must implement the subtask's declared correctness and timing protocol.

## Integrate and compare the model

Compile the source/template change with the same model/profile and
`--baseline-run "$BASELINE" --validation-contract "$CONTRACT"`. Tolerances come
from that contract. Use a fresh output label/cache, preserve original source/runtime
identity, and inspect the new package to prove actual selection. `CANDIDATE_RUN`
means the emitted new compilation run, not the isolated kernel benchmark.

```sh
"$PYTHON" "$KDA/inductor/scripts/audit_codegen.py" \
  --baseline-run "$BASELINE" --candidate-run "$CANDIDATE_RUN" \
  --output "$TASK/reports/$AUDIT_LABEL"
"$PYTHON" "$KDA/inductor/scripts/run_guarded.py" \
  --gpu "$GPU_UUID" --max-idle-memory-mib "$IDLE_MIB" \
  --output-root "$TASK/runs" --label "$COMPARISON_LABEL" \
  --source-root "$BASELINE_RUNTIME_ROOT" \
  -- "$PYTHON" "$KDA/inductor/scripts/compare_models.py" \
  --baseline-run "$BASELINE" --candidate-run "$CANDIDATE_RUN" \
  --validation-contract "$CONTRACT"
```

Add `--incumbent-run "$INCUMBENT"` for later iterations. `BASELINE_RUNTIME_ROOT` is
an immutable source/package root matching the original baseline; do not use the
modified compiler checkout merely because it was used to build the candidate.
Audit supports the inspected C++/embedded-Triton format, not every package or
external-library call. Unsupported extraction fails; review manually rather than
claiming no unrelated changes. Both packages use the same native runtime; if a
change requires a different runtime, define a separate comparison protocol.

For controlled source changes, optional `export_autotune_configs.py` takes
`--compile-run`, `--source-root`, `--output`. Pass its manifest to compilation with
`--baseline-autotune-configs`. If kernels were renamed, `rebind_autotune_configs.py`
takes `--baseline-manifest`, `--candidate-run`, `--output`; compile again using the
new replay manifest, then audit actual code and launches. See [compilation](compilation.md)
for the identity and cache restrictions. Never copy whole compiled caches.

## Checkpoint and resume

```sh
"$PYTHON" "$KDA/inductor/scripts/task_workspace.py" checkpoint \
  --task-root "$TASK" --phase "$PHASE" --next "$NEXT_ACTION" --decision "$DECISION" \
  --artifact "baseline=$BASELINE/manifest.json" \
  --artifact "contract=$CONTRACT" --artifact "ranking=$PROFILE_RUN/hotspots.json"
"$PYTHON" "$KDA/inductor/scripts/task_workspace.py" status --task-root "$TASK"
```

Register additional evidence as it becomes available; do not reference nonexistent
artifacts. Use `incumbent` for accepted candidate manifests. Keep active-process
notes and HANDOFF.md current. `status` is offline and checks registered file
hashes; it does not assert GPU availability or terminate/restart workers.
