# Parameter naming for SparseProblem

Research checked on 2026-10-07 using PyTorch 2.10 documentation and source.
The user accepted the `inner_weight` / `inner_bias` recommendation on 2026-10-07;
the rename is implemented, verified, and independently reviewed. The
[handoff](../REFACTOR_HANDOFF.md) records the checks and scope.

Use `W` → `inner_weight` and `b` → `inner_bias` for public attributes
and corresponding public arguments. These names preserve this project's
inner/outer vocabulary and follow PyTorch's singular tensor names.

PyTorch's `nn.Linear` exposes `weight` for a matrix and `bias` for a vector;
singular names do not imply scalar values. Its functional equivalent takes
`input`, `weight`, and `bias`.
([Linear documentation](https://docs.pytorch.org/docs/2.10/generated/torch.nn.Linear.html),
[functional.linear documentation](https://docs.pytorch.org/docs/2.10/generated/torch.nn.functional.linear.html))

When several roles coexist, PyTorch also uses qualified names:
`MultiheadAttention` directly stores `in_proj_weight` and `in_proj_bias`,
and can store `q_proj_weight`, `k_proj_weight`, and `v_proj_weight`.
([PyTorch v2.10.0 source](https://github.com/pytorch/pytorch/blob/v2.10.0/torch/nn/modules/activation.py))
This supplies precedent for a role prefix; it does not prescribe `inner`
or mandate one naming pair for custom modules.

Our comparison, applying the repository's
[consistent-vocabulary and sufficient-context rules](../refactor.md):

| Pair | Assessment here |
|---|---|
| `inner_weight`, `inner_bias` | Recommended: identifies the existing domain role using PyTorch's singular convention. |
| `hidden_weight`, `hidden_bias` | Understandable, but changes the established inner/outer vocabulary to layer terminology. |
| `weight`, `bias` | Clear inside one affine layer; insufficient context in a model containing both inner and outer weights. |
| `inner_weights`, `inner_biases` | Readable, but pluralization adds no role information; prefer singular consistently with PyTorch. |

Mathematical `W` and `b` can remain local where defined nearby, as `refactor.md`
permits. No new layer or module hierarchy is needed. If the naming scope later
includes `c`, `outer_weight` would form a consistent counterpart; that is a
separate naming decision.

[SparseProblem](../src/problems/sparse_problem.py#L89) directly registers
`inner_weight`/`inner_bias` buffers and the `c` parameter. Registered names become
[state_dict keys](https://docs.pytorch.org/docs/2.10/generated/torch.nn.Module.html#torch.nn.Module.state_dict),
so [History capture and restoration](../src/PDAP/history.py#L93) now writes those
names and translates existing `W`/`b` snapshots and older `hidden.*` snapshots at
the reader. The existing atom-field fallback and independent checkpoint storage
are retained.
