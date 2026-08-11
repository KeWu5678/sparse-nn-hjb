---
status: accepted
---

# CI is a merge gate for code, and invisible to the paper

CI has run on GitHub Actions since 2026-07-02 (PR #30): ruff plus the pytest
suite, on pushes to `main` and on every pull request. It was advisory. `main`
carried no branch protection, so a pull request whose suite was red could still
be merged, and the check served only as a signal the author chose to read.

This repository has two kinds of traffic through `main`, and they want opposite
treatment. Code changes arrive as pull requests (#14–#32) and should not be
allowed to land broken. Thesis and proof commits are pushed straight to `main` —
twelve consecutive such commits touched no `.py` file at all — and gain nothing
from a runner that lints and runs 83 tests against unchanged sources.

## Decision

- **Branch protection on `main` requires the `test` check.** A pull request
  whose suite is red cannot be merged.
- **Administrators are exempt** (`enforce_admins: false`). Direct pushes of
  paper and proof commits to `main` continue to work exactly as before. This is
  not incidental: doc-only pushes no longer produce a `test` check at all, so
  enforcing protection on the owner would reject them outright.
- **No up-to-date requirement** (`strict: false`). `main` advances mostly with
  thesis commits that cannot break code; forcing every open pull request to
  rebase after each one is friction without a corresponding risk.
- **No review requirement.** A single author cannot approve their own pull
  request, so requiring one would wedge every pull request opened.
- **`paths-ignore` on the `push` trigger only** — `papar/**`, `**.md`,
  `.gitignore`. It is deliberately absent from `pull_request`: a required check
  that never runs also never reports, leaving the pull request permanently on
  "Expected — waiting for status".
- **Python is pinned to 3.14**, matching the local development interpreter.
  Unpinned, `uv` selected the runner's system interpreter (3.12.3 on
  ubuntu-24.04), so the tested version tracked Ubuntu's packaging and would
  change without a commit.
- **Lint runs through `pre-commit run --all-files`** rather than a separate
  `ruff check` invocation, making the local hooks and CI one configuration. It
  also covers the whitespace, end-of-file, large-file and merge-conflict hooks,
  which a `--no-verify` commit would otherwise bypass entirely.
- **A `github-actions[bot]` comment reports results on each pull request**,
  edited in place rather than reposted per push. It carries pass/fail counts,
  duration and, when red, the failing test names.
- **Concurrency cancels superseded runs** on the same ref.

## Consequences

The status check, not the comment, is what blocks a merge; the comment is
information only and can be removed without weakening the gate. Renaming the
`test` job silently disables branch protection, since the protection rule names
that job as its required context — the rule must be updated first.

Coverage is deliberately not measured or reported. The number would be dominated
by `src/plots.py` (471 statements, untested by design), and a threshold on a
single-researcher research repository generates tests written to move a metric
rather than to catch an error. The solver core it would report on already sits
between 79% and 100%.

`research` was dropped from the push trigger: it is fully merged into `main` and
has been stale since 2026-07-02. Were it revived, pull requests from it would
still run CI through the `pull_request` trigger.
