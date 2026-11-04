# Model optimization starter

Use `skills/inductor-model-optimization/SKILL.md` and the repository's model-level
workflow. Read the supplied task context or existing task workspace first. Resolve
only missing scenario inputs with the user; do not assume hardware, remote host,
GPU selection, model paths or numerical tolerances from a historical run.

For a new task, initialize `agent_space/<model>_<task-date>/` from the completed
context template. Compile the versioned baseline, validate and measure ordinary
AOT and CUDA Graph separately, then profile and rank all kernels by aggregate GPU
time. Select high-cost kernels and follow KDA's contract/diagnose/implement/
validate/measure loop, including structural implementations beyond tile tuning.
Translate validated candidates into guarded compiler templates with fallback.
Recompile, audit and validate the complete model; compare against both the
original baseline and incumbent before promotion.

For an existing task, verify the checkpoint and artifact hashes, check active
workers, and continue from the recorded next action. Preserve rejected candidates,
raw measurements and precision limitations. Leave a verified checkpoint and a
self-contained handoff for a fresh agent context.
