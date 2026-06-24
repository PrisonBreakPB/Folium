---
name: experiment-runner
description: Use for writing, running, debugging, and analyzing Python experiments, data analysis scripts, and reproducible research code.
---

# Experiment Runner Skill

Use this workflow when the task involves experiment code, data analysis, scripts, or result iteration.

## Workflow

1. Inspect the project layout and existing scripts before creating new files.
2. Prefer small, reproducible scripts with clear inputs and outputs.
3. Keep generated artifacts in an obvious location and report their paths.
4. Run the script when feasible and inspect the output, logs, or generated files.
5. If execution fails, diagnose the first concrete error, patch minimally, and retry.
6. Summarize final results with commands run, files changed, and remaining limitations.

## Safety

- Avoid long-running or interactive commands unless the user explicitly wants them.
- Add timeouts or non-interactive flags for commands that may hang.
- Do not delete or overwrite user data without explicit approval.
