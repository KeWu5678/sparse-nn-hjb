---
status: proposed
---

# Use uv for public Python commands

The repository already uses `uv` for dependency resolution, CI, and documented
one-off tools, but the root Makefile, MLflow guides, and open-loop generator
docstrings still assume an environment at `.venv/bin/python`. That path works on
the primary development machine but makes the public command contract depend on
one local environment layout.

The proposed decision is to make `uv run python` the default for every tracked
user-facing Python command. The Makefile would use an overridable
`PY ?= uv run python`, and command examples would use the same invocation.
Direct script paths remain the entry points; this proposal does not add console
scripts or a second operations dispatcher merely to remove their small
`sys.path` bootstraps.

This would give local use and CI one environment boundary while retaining a
`PY` override for an already-active interpreter or unusual deployment. The cost
is requiring `uv` for the documented workflow and allowing it to synchronize the
project environment before a command runs.

Keeping `.venv/bin/python` was considered because it avoids synchronization at
invocation time, but it assumes that the environment exists at exactly that
path. Adding packaged console entry points was also considered; the current
three-command public surface is too small to justify that extra interface.
