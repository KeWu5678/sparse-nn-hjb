# Mathematical terminology for the objective rewrite

Status: controlled vocabulary for `paper_0805.tex`.

This is a controlled vocabulary, not a glossary of every mathematical word in
the paper. Symbols and their definitions belong in `notation.md`.

## Admission rule

A term is recorded here only if all three conditions hold:

1. it is established mathematical terminology;
2. it occurs in a definition or in the body of a theorem, lemma, corollary, or
   remark; and
3. it is expected to occur in at least three distinct places in the revised
   paper.

A new symbol must be defined before it is used. Proof maneuvers, theorem
conclusions, algorithm steps, and one-off descriptive phrases are not promoted
to terms.

## Terms

**Continuous part**:
The nonatomic component in the unique decomposition of a finite signed Radon
measure into atomic and nonatomic components.
_Avoid_: Diffuse part

**Covering number**:
The least number of metric balls of a prescribed radius required to cover a
set.
_Avoid_: Metric entropy; metric entropy is derived from the behavior of
covering numbers rather than being one covering number

**Normalized measure / normalized atom**:
The weighted measure coordinate `μₚ = wₚμ` and the corresponding dictionary
atom `Kₚ = K/wₚ` used on the unbounded parameter domain.
_Avoid_: Renormalized measure; scaled neuron

**Reduced width**:
The number of distinct nonzero atoms in a measure representation, after zero
coefficients are omitted and repeated locations are merged.
_Avoid_: Nominal width, when a representation contains redundant neurons

**Atomwise optimality**:
The first-order stationarity condition imposed separately at every existing
atom of a finite-objective measure.
_Avoid_: Pointwise optimality, which may suggest a condition at inactive
locations

## Admission record

| Term | Formal locations | Occurrences in `paper_0805.tex` |
| --- | --- | ---: |
| Continuous part | Measure decomposition, representation/optimality/support results, and homogeneous penalty definition | 10 |
| Covering number | Finite-support theorem, introduction, and conclusion | 3 |
| Normalized measure / normalized atom | Normalization theorem, existence proof, and finite-support theorem | 10 combined phrase occurrences |
| Reduced width | Homogeneous local-minimizer definition and global/local support results | 6 |
| Atomwise optimality | Homogeneous optimality subsection, theorem, and existence proof | 4 |
