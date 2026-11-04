---
name: inductor-model-optimization
description: Optimize complete models compiled by PyTorch AOTInductor using KDA kernel subloops. Use for baseline compilation, ordinary/CUDA Graph measurement, top-k GPU hotspot analysis, kernel-to-FX/template provenance, mechanism-based kernel optimization, template integration, or resuming a model optimization task. Requires task-supplied model, runtime, hardware and GPU context; contains no host or card defaults. Not for a standalone kernel task or production serving benchmark alone.
---

# Inductor model optimization

Use the model-level loop in this repository. Resolve links relative to this skill;
run tools from the KDA repository root. This skill depends on the repository's
`inductor/`, `docs/` and optional kernel/profiler skills, not private historical runs.
Do not copy only this SKILL.md into another installation and expect tools to exist.

## Start or resume

1. Read [the model workflow](../../inductor/docs/workflow.md). For an existing task,
   read task.json, state.json, HANDOFF.md and draft/plan; run `task_workspace.py status`
   and inspect active workers before deciding which stages can be skipped.
2. Resolve missing model/input contract, execution/runtime, hardware and authorized
   GPU, accuracy policy and promotion criteria from user context and available
   files. Do not inherit a prior task's host, device, precision waiver or paths.
3. For a new task, fill the [context template](../../inductor/configs/task_context.example.json),
   initialize ignored `agent_space/<model>_<task-date>/`, and snapshot the bundled
   [baseline](../../inductor/configs/compile_profiles.json). Use the
   [tool recipes](../../inductor/docs/tool-recipes.md) for executable commands.

## Execute with evidence gates

- **Compile:** read [compilation](../../inductor/docs/compilation.md). Inspect the
  input format, verify profile compatibility, compile baseline, retain exact
  inputs/reference and source/native runtime identities. Never silently use `safe`.
- **Measure and select:** read [benchmarking](../../inductor/docs/benchmarking.md).
  Validate, time ordinary and Graph without profilers, then capture a separate
  trace. Rank by summed device time across all backends; target the largest costs.
- **Map:** read [provenance](../../inductor/docs/provenance.md) for exact wrapper,
  ABI and captured boundary data. Read [compiler analysis](../../inductor/docs/compiler-analysis.md)
  when FX, fusion, template origin or selection needs deeper analysis.
- **Optimize:** follow [KDA's kernel loop](../../docs/agent-flow.md) in each kernel
  workspace and [mechanism guidance](../../inductor/docs/kernel-mechanisms.md).
  Diagnose the bottleneck before implementing structural candidates. Read the
  KernelWiki or NCU skill when its supported domain answers the actual question.
- **Integrate and accept:** add guarded template choices with fallback, recompile,
  audit selection and unrelated changes, validate the whole model, and run paired
  or three-way comparisons against original baseline and incumbent. Promote only
  a supported model-level improvement. Preserve rejected experiments.

Checkpoint each completed stage and update handoff before ending a session.
Report original hotspot ranks/shares, kernel and model gains separately, validation
scope, artifacts and the next action. Missing evidence is a limitation, not a pass.
No deployment, publication, dependency rebuild or foreign-process termination is
part of this workflow unless separately requested.
