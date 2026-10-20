"""Importing an open-loop experiment must not write anything."""

import importlib
import runpy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("problem", ["vdp", "pendulum"])
def test_openloop_import_does_not_create_output_directories(monkeypatch, problem):
    # Initialize plotting dependencies before testing the experiment's import.
    importlib.import_module("src.plots")

    def unexpected_mkdir(*args, **kwargs):
        pytest.fail("importing an experiment must not create output directories")

    monkeypatch.setattr(Path, "mkdir", unexpected_mkdir)
    namespace = runpy.run_path(str(ROOT / f"experiments/00_openloop/{problem}/generate.py"))
    assert callable(namespace["main"])
