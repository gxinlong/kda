# KDA agent entrypoint

Read [CLAUDE.md](CLAUDE.md) for repository rules. Repository-facing files are in
English; keep private task inputs and results in ignored workspaces.

For complete-model PyTorch Inductor/AOT optimization, use
[skills/inductor-model-optimization/SKILL.md](skills/inductor-model-optimization/SKILL.md).
It routes to the baseline configuration, compilation/benchmark/analysis tools and
the model loop around KDA. Read task.json, state.json and HANDOFF.md when resuming;
verify artifacts and active workers instead of relying on earlier chat context.

For a standalone kernel task, start with [docs/agent-flow.md](docs/agent-flow.md)
and [prompts/basic-flow.md](prompts/basic-flow.md).
