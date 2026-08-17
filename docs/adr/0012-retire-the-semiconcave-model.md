# Retire the semiconcave model implementation

Status: accepted

The manuscript and current experiments use only signed shallow networks. The
separate semiconcave parametrization required its own one-sided insertion,
nonnegative warm start, augmented coefficient vector, and model-specific tests,
while its historical experiments found no consistent advantage. We therefore
remove it from the active implementation and configuration instead of repairing
an unused candidate-search branch. The term *semiconcave* remains valid when it
describes a value function; only the `SemiconcaveModel` implementation is
retired. Historical results remain archived, and commit `fdfd0e8` preserves the
last complete implementation.
