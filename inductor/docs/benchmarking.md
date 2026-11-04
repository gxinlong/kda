# Model measurement and hotspot selection

Measure an immutable compiled package. Do not recompile it to solve loading errors.
Freeze package, weight, input and schema hashes; runtime/adapter identity; GPU UUID;
TF32 and CPU settings. Use the original boundary signature and input contents.
Record shape, dtype, stride, offsets, TreeSpec, aliases and dynamic constraints.
Synthetic or example inputs must be labeled; changing batch requires corresponding
length/index/mask changes and must preserve request/note semantics.

Run one worker at a time after fresh GPU admission. Keep external 1 Hz telemetry.
Foreign processes invalidate an exclusive measurement even at 0% utilization.
Verify loaded output structure and finite values, compare with an independent
reference when available, then clone ordinary outputs before any subsequent call.
A known numerical mismatch blocks formal timing. Missing reference is an explicit
limitation; consistency alone does not prove correctness.

When a mismatch is found, retain the original failure and diagnose input identity,
repeatability and arithmetic policy before changing a tolerance. A runtime TF32
change affects external GEMMs but does not rewrite embedded Triton arithmetic.
Do not silently label baseline-output equivalence as independent model accuracy.
`--validation-contract` records either a separate validated reference or explicit
user authorization for diagnostic timing against the original baseline. It binds
the exact package and input hashes and records runtime precision settings.

For a stateless fixed-input workload, use inference mode, GPU-resident weights and
inputs, 30 ordinary warmups, 5 side-stream warmups with proper dependencies, and
capture K=10 complete forwards. Synchronize, clone replay outputs and compare
against the cloned ordinary reference. Prewarm at least 5 replays. Retain all
model/input/weight/output lifetimes. Report capture failure without relabeling
ordinary calls as Graph execution.

Default timing: three rounds of 20 samples per mode, alternating mode order.
Before each sample drain prior work. Time K consecutive ordinary calls, or one
replay containing K calls, using pre-created CUDA events and wall time; wait once
at the end. Divide both block times by K. Store raw samples, order, per-round and
overall medians, nearest-rank P95 and measured forward count. P95 is a percentile
of block averages, not single-request latency. State the warm/hot repeated-input
cache condition. No profiler is attached to acceptance timing.

Collect a separate short nsys trace with API-delimited capture after validation,
warmup and Graph setup. `--trace=cuda,nvtx --cuda-graph-trace=node` suffices for a
kernel ranking; call it a limited trace, not full metrics profiling. Put ordinary
and Graph phases and complete K-forward blocks in distinct NVTX ranges. Verify
tool/worker exits, actual range duration, process/device, node events and warnings.
Inspect SQLite schema before writing extraction queries.

For each mode use complete blocks to report top kernels by **sum of device
durations**, with count, mean, median, grid/block and per-forward aggregate cost.
Select top 10 initially and prioritize the largest attributable model costs across
all backends. An existing Triton implementation is not a reason to skip a larger
external-library GEMM; such calls may become targets for new template candidates.
Report each group's call count and individual durations, because matching kernel
names and launch dimensions can still combine different operator shapes. Verify
counts across at least three complete blocks. One external
GEMM call may launch several kernels. Kernel durations summed over multiple streams
are not model latency or a critical-path measurement.

Map each target to the exact package wrapper and generated implementation. Save
its call parameters and true boundary tensors; random same-shape inputs do not
replace captured data or alias relationships. Create a separate KDA task per
kernel and keep rejected candidate evidence. NCU is optional for a specific
resource/occupancy/memory question; its replay duration is not acceptance timing.
When used, record exact target matching, cache policy and `--clock-control none`.

After template integration, prove the new candidate was selected, validate the
whole model and use alternating or paired baseline/candidate timing on the same
card. Preserve slow samples and drift evidence. Small differences near noise do
not justify promotion; isolated kernel gains do not establish model gains.

`benchmark_model.py --source-root` accepts a relocated immutable compilation tree
after verifying its source contents and native runtime. Source identity checks run
in the outer supervisor before and after measurement, not through subprocesses in
an injected profiler worker. The worker selects that root before importing Torch
and checks the imported path and native binary hash.

`compare_models.py`, launched through `run_guarded.py`, holds both immutable AOT
packages and the same saved inputs in one process. It alternates package order
within each pair and ordinary/Graph order across rounds. Both packages run with
the baseline's source contents and native wheel; a relocated clean worktree is
allowed only after matching the source identity except for its directory name.
The script explicitly prioritizes that worktree before importing Torch because
an installed editable `.pth` hook can otherwise override `PYTHONPATH`.

Profiler launchers may create child sessions. Supervision resolves the task's
descendant tree and each visible host PID mapping, including descendants outside
the launcher's process group. Unknown device processes still invalidate the run.
Admission immediately after a completed test may see residual utilization; retain
that blocked attempt and retry only after fresh idle evidence.

See [tool recipes](tool-recipes.md) for commands. Hardware, GPU, idle memory
ceiling and tolerances must come from the current task.

For structural kernel experiments and generated-code resource validation, see
[kernel-mechanisms.md](kernel-mechanisms.md). Use `--incumbent-run` with
`compare_models.py` to retain both the original and last accepted reference in a
three-way experiment; a candidate must improve on the last accepted package.
