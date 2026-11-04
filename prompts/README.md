# Prompt Templates

This directory stores generic prompts for the Kernel Design Agents workflow.

The templates are intentionally task-agnostic. Fill in the task objective, constraints, validation command, and promotion criteria before starting an agent session in a separate implementation workspace.

## Available Templates

| Path | Purpose |
|---|---|
| `basic-flow.md` | Minimal prompt for research, planning, implementation, validation, and iteration. |

## How To Use

1. Create or enter the task implementation workspace.
2. Copy the relevant template content into the agent session.
3. Replace placeholders with task-specific details.
4. Ask the agent to read the workspace and write `docs/draft.md`.
5. Convert that draft into an executable plan.
6. Run the implementation loop with validation after each meaningful change.

Task-specific prompts should live with the task they describe. Do not add benchmark-specific datasets, acceptance tables, or private evaluator details to this generic repository.

For complete-model AOTInductor work, use [inductor-model.md](inductor-model.md) and
the repository-local [model optimization skill](../skills/inductor-model-optimization/SKILL.md).
