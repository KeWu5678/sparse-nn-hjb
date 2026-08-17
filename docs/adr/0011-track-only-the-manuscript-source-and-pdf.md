# 11. Track only the manuscript source and compiled PDF

Date: 2026-08-17

## Status

Accepted

## Context

The manuscript includes a bibliography and figures generated from local
experiment records. Tracking those paper-facing artifacts added thousands of
generated lines and binary files to algorithm pull requests even though the
authoritative run records and analysis workspace remain local.

The TeX source is still needed for reviewable manuscript history, while the
compiled PDF is the portable publication artifact. Requiring a clean checkout
to rebuild the PDF would require versioning every local paper input and would
reintroduce the artifact churn this decision removes.

This decision is narrower than ADR 0003: stable outputs of ordinary curated
studies may still be promoted under `experiments/`. It concerns the current
paper workspace and its manuscript-specific generated studies.

## Decision

Under `paper/`, track only `paper_0805.tex` and `paper_0805.pdf`. Keep the
bibliography, included figures, build products, working notes, paper-specific
analysis scripts, and paper-specific experiment reports local and ignored.

The TeX source is authoritative for manuscript edits. The tracked PDF is the
readable artifact for a clean checkout. Rebuilding the PDF requires the local
paper workspace and is intentionally not a clean-checkout guarantee.

## Consequences

- Algorithm pull requests do not carry generated paper figures and reports.
- Manuscript changes remain reviewable as text and as a compiled PDF.
- A clean checkout can read the paper but cannot rebuild it without restoring
  the local bibliography and figures.
- Ordinary curated experiment outputs remain governed by ADR 0003 rather than
  this paper-specific policy.

## Alternative considered

**Track every TeX dependency.** Rejected because it couples code changes to a
large, frequently regenerated binary artifact set whose source records remain
local.
