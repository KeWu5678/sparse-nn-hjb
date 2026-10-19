---
status: accepted
---

# One value-sample type, owned by `src/data.py`

Open-loop generation and PDAP training exchange value samples through an
`.npz` file with arrays `x`, `v`, and `dv`. That one contract was implemented
twice: generation wrote it through the frozen dataclass
`src/OpenLoop/value_samples.py:ValueSamples` (`v` shaped `(n,)`), while training
read it into a dict alias of the same name in `src/data.py` (`v` shaped
`(N, 1)`) with its own copy of the shape checks.

We decided that `src/data.py` owns the single `ValueSamples` type: the frozen
dataclass moves there and replaces the dict alias, and
`src/OpenLoop/value_samples.py` is deleted. Writing, loading, and
concatenating value samples live beside each other in `src/data.py`; solvers,
generators, and training all construct and consume the same validated,
immutable object with typed fields `x`, `v`, `dv`.

The dataclass stores `v` as `(n,)`, one scalar value per sample. The `(N, 1)`
shape is a requirement of the network and tensor batches, so it is applied in
one place, `to_tensors`, which splitting, saved-run validation, and region
evaluation all use to convert arrays to tensors. Concatenation requires at least
one chunk; an empty set is built explicitly with `ValueSamples.empty(dimension)`
instead of defaulting to two columns. The dataclass
marks its arrays read-only, because `frozen=True` prevents field reassignment
but not in-place array writes. The on-disk format is unchanged, and legacy
`.npy` files are still adapted by `load_value_samples`.

## Considered Options

Keeping the training-side dict as the only type and deleting the dataclass
touches only the generation code and leaves training and saved-run reloading
untouched. It was rejected because any caller can build or mutate a dict that
skips validation, every consumer must know the shape contract from a docstring,
and string keys are not typed fields. The cleaner single abstraction was judged
worth the wider change to training code and tests.

## Consequences

Generation code under `src/OpenLoop/` imports from `src/data.py`, which imports
PyTorch; the generator scripts already import `src.data` for `DATA_DIR`. The
change is behavior-preserving and is committed separately from the shared
open-loop collocation refactor.
