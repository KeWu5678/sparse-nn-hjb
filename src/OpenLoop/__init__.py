"""Open-loop optimal-control data generation.

The shared finite-horizon collocation solver lives at this level; problem
definitions and the pendulum's own solver live below ``vdp``, ``allen_cahn``,
and ``pendulum``. Value samples are ``src.data.ValueSamples``.
"""
