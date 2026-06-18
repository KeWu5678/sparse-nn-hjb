# capacity_selection_split

**status:** refuted

**statement (the claim that was made).** The cone advantage decomposes **additively**,
$$N(\text{signed},V,\varepsilon)-N(\text{semiconcave},V,\varepsilon)\ =\ \underbrace{T_{\text{cap}}}_{\text{from the head}}+\underbrace{T_{\text{sel}}}_{\text{from }c_i\ge0},$$
with the two terms separately measurable by subtracting the fitted head $\tfrac{\tilde C}{2}\|x\|^2$
from the data and re-fitting the signed model.

**refutation.** The subtraction is **confounded**: the fitted $\tilde C$ is co-adapted with the
atoms, so $V-\tfrac{\tilde C}{2}\|x\|^2$ is neither bulk-free nor cone-friendly. Faithful repo
PDAP gave **contradictory** answers depending on the protocol: a rigged synthetic target said
"$T_{\text{sel}}\approx0$"; the real VDP test said the opposite (subtracting the bulk made
signed *worse*, 44 vs 12 atoms). An additive split with protocol-dependent sign is not a
well-defined decomposition. Conclusion: the advantage does **not** split additively this way.

**what survived.** `head_reduction` (head exact, free) is the correct, non-additive residue.

**attempts.** 1. matched-head subtraction (synthetic) → "capacity is everything". 2. matched-
head subtraction (real VDP) → opposite. 3. abandoned the decomposition framing entirely.
Detail/log: `../refs/theorem-A-d-and-history.md`, `../../_archive/program-log.md`.
