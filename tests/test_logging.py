"""Exercise entry-point logging in subprocesses without changing pytest's loggers."""

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from textwrap import dedent

import numpy as np
import pytest


def test_default_outputs_share_fields_and_filter_debug(tmp_path):
    result = _run_logging(tmp_path, """
        configure_logging(log_file=output_dir / "run.jsonl")
        logger = structlog.get_logger("example.structured").bind(run="demo")
        logger.debug("hidden structured detail")
        logger.info("structured event", count=3)
        standard = logging.getLogger("example.standard")
        standard.debug("hidden standard detail")
        standard.info("standard %s", "event", extra={"count": 3})
    """)

    records = [json.loads(line) for line in (tmp_path / "run.jsonl").read_text().splitlines()]
    assert [record["event"] for record in records] == ["structured event", "standard event"]
    assert [record["logger"] for record in records] == ["example.structured", "example.standard"]
    assert records[0]["run"] == "demo"
    for record in records:
        assert record["level"] == "info"
        assert record["count"] == 3
        assert datetime.fromisoformat(record["timestamp"]).utcoffset().total_seconds() == 0
        assert "_record" not in record and "_from_structlog" not in record
        assert result.stderr.count(record["event"]) == 1
    assert result.stderr.count("count=3") == 2
    assert "run=demo" in result.stderr
    assert "hidden" not in result.stderr
    assert result.stdout == ""


def test_reconfiguration_updates_outputs_and_levels_for_existing_loggers(tmp_path):
    result = _run_logging(tmp_path, """
        configure_logging(log_file=output_dir / "first.jsonl")
        logger = structlog.get_logger("example.structured").bind(run="demo")
        standard = logging.getLogger("example.standard")
        logger.info("first event")

        configure_logging(log_file=output_dir / "second.jsonl", level="DEBUG")
        logger.debug("structured debug")
        standard.debug("standard debug")

        configure_logging(log_file=output_dir / "second.jsonl", level="WARNING")
        logger.info("hidden structured info")
        standard.info("hidden standard info")
        logger.warning("structured warning")
        standard.warning("standard warning")
    """)

    first = [json.loads(line) for line in (tmp_path / "first.jsonl").read_text().splitlines()]
    second = [json.loads(line) for line in (tmp_path / "second.jsonl").read_text().splitlines()]
    assert [record["event"] for record in first] == ["first event"]
    assert [record["event"] for record in second] == [
        "structured debug", "standard debug", "structured warning", "standard warning",
    ]
    assert [record["level"] for record in second] == ["debug", "debug", "warning", "warning"]
    assert second[0]["run"] == second[2]["run"] == "demo"
    for record in first + second:
        assert result.stderr.count(record["event"]) == 1
    assert "hidden" not in result.stderr
    assert result.stdout == ""


def test_exceptions_keep_tracebacks_and_fields_in_both_outputs(tmp_path):
    result = _run_logging(tmp_path, """
        configure_logging(log_file=output_dir / "errors.jsonl")
        logger = structlog.get_logger("example.structured")
        try:
            raise ValueError("structured failure")
        except ValueError:
            logger.exception("structured exception", item=1)
        try:
            raise RuntimeError("standard failure")
        except RuntimeError:
            logging.getLogger("example.standard").exception("standard exception", extra={"item": 2})
        logger.info("recovered")
    """)

    records = [json.loads(line) for line in (tmp_path / "errors.jsonl").read_text().splitlines()]
    assert [record["event"] for record in records] == [
        "structured exception", "standard exception", "recovered",
    ]
    for item, (record, error) in enumerate(zip(records, (
        "ValueError: structured failure", "RuntimeError: standard failure",
    )), start=1):
        assert record["level"] == "error"
        assert record["item"] == item
        assert record["exception"].count("Traceback (most recent call last)") == 1
        assert error in record["exception"]
        assert result.stderr.count(error) == 1
        assert result.stderr.count(record["event"]) == 1
        assert f"item={item}" in result.stderr
        assert "exc_info" not in record
    assert result.stderr.count("Traceback (most recent call last)") == 2
    assert "exception" not in records[-1]
    assert result.stdout == ""


def test_console_can_be_disabled_and_restored_without_changing_file_output(tmp_path):
    result = _run_logging(tmp_path, """
        configure_logging(log_file=output_dir / "run.jsonl", console=False)
        logger = structlog.get_logger("example.structured")
        standard = logging.getLogger("example.standard")
        logger.debug("hidden debug")
        standard.debug("hidden standard debug")
        logger.info("structured file only")
        standard.info("standard file only")

        configure_logging(log_file=output_dir / "run.jsonl")
        logger.info("console restored")
        standard.info("standard console restored")
    """)

    records = [json.loads(line) for line in (tmp_path / "run.jsonl").read_text().splitlines()]
    assert [record["event"] for record in records] == [
        "structured file only", "standard file only", "console restored", "standard console restored",
    ]
    assert "file only" not in result.stderr and "hidden" not in result.stderr
    assert len(result.stderr.splitlines()) == 2
    assert result.stderr.count("standard console restored") == 1
    assert result.stdout == ""


def test_context_reaches_both_apis_and_restores_after_an_exception(tmp_path):
    result = _run_logging(tmp_path, """
        configure_logging(log_file=output_dir / "context.jsonl")
        logger = structlog.get_logger("example.structured")
        standard = logging.getLogger("example.standard")
        with structlog.contextvars.bound_contextvars(run_id="outer", seed=1):
            logger.info("outer context")
            try:
                with structlog.contextvars.bound_contextvars(run_id="inner", experiment="demo"):
                    logger.info("structured context", seed=2)
                    standard.info("standard context", extra={"seed": 3})
                    raise ValueError("leave inner context")
            except ValueError:
                standard.info("restored context")
        logger.info("unbound context")
    """)

    records = [json.loads(line) for line in (tmp_path / "context.jsonl").read_text().splitlines()]
    assert [record.get("run_id") for record in records] == ["outer", "inner", "inner", "outer", None]
    assert [record.get("seed") for record in records] == [1, 2, 3, 1, None]
    assert [record.get("experiment") for record in records] == [None, "demo", "demo", None, None]
    assert result.stderr.count("run_id=inner") == 2
    assert result.stderr.count("run_id=outer") == 2
    assert result.stdout == ""


@pytest.mark.parametrize("console,level,override_file", [
    (True, "INFO", False), (False, "INFO", True), (True, "WARNING", True),
])
def test_hydra_runner_owns_jsonl_output_and_honors_logging_overrides(
    tmp_path, monkeypatch, console, level, override_file,
):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    dataset = tmp_path / "samples.npz"
    x = np.arange(4.0).reshape(-1, 1)
    np.savez(dataset, x=x, v=x**2, dv=2*x)
    run_dir = tmp_path / "run"
    log_file = tmp_path / "custom" / "events.jsonl" if override_file else run_dir / "run.jsonl"
    command = [
        sys.executable, "scripts/train.py", f"data.path={dataset}", "data.train_fraction=0.5",
        "training.loop_order=insertion_first", "training.num_iterations=0",
        f"hydra.run.dir={run_dir}", f"env.verbose={str(console).lower()}", f"env.log_level={level}",
    ]
    if override_file:
        command.append(f"env.log_file={log_file}")
    result = subprocess.run(
        command, cwd=Path(__file__).resolve().parents[1],
        check=True, capture_output=True, text=True, timeout=30,
    )

    records = [json.loads(line) for line in log_file.read_text().splitlines()]
    record_path = next(run_dir.glob("*.json"))
    record = json.loads(record_path.read_text())
    if level == "WARNING":
        assert records == []
        assert result.stderr == ""
    else:
        events = [record["event"] for record in records]
        assert events == [
            "run_started", "data_loaded", "training_progress", "training_finished",
            "evaluation_summary", "run_completed",
        ]
        assert len(events) == len(set(events))
        assert all({"event", "level", "logger", "timestamp"} <= record.keys() for record in records)
        assert all(entry["run_id"] == record["run_id"] for entry in records)
        assert all(entry["experiment"] == record["name"] for entry in records)
        assert all(entry["seed"] == record["config"]["env"]["seed"] for entry in records)
        assert records[1]["path"] == str(dataset) and records[1]["input_dim"] == 1
        assert records[-1]["record_path"] == str(record_path)
        if console:
            for event in events:
                assert result.stderr.count(event) == 1
        else:
            assert result.stderr == ""
    assert result.stdout == ""
    assert not list(tmp_path.rglob("*.log"))
    if override_file:
        assert not (run_dir / "run.jsonl").exists()
    assert (run_dir / ".hydra" / "config.yaml").exists()
    assert record["status"] == "completed"
    assert record["config"]["env"]["verbose"] is console
    assert record["config"]["env"]["log_level"] == level
    assert record["config"]["env"]["log_file"] == (str(log_file) if override_file else None)


@pytest.mark.parametrize("require_mlflow,error", [
    (False, "FileNotFoundError"), (True, "RuntimeError"),
])
def test_hydra_failure_logs_one_outcome_and_preserves_the_original_error(
    tmp_path, monkeypatch, require_mlflow, error,
):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    run_dir = tmp_path / "failed"
    result = subprocess.run(
        [sys.executable, "scripts/train.py", f"data.path={tmp_path / 'missing.npz'}",
         f"hydra.run.dir={run_dir}", "env.verbose=false",
         f"env.require_mlflow={str(require_mlflow).lower()}"],
        cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=30,
    )

    assert result.returncode != 0
    records = [json.loads(line) for line in (run_dir / "run.jsonl").read_text().splitlines()]
    assert [record["event"] for record in records] == ["run_started", "run_failed"]
    assert records[0]["run_id"] == records[1]["run_id"]
    assert records[1]["level"] == "error"
    assert records[1]["exception"].count("Traceback (most recent call last)") == 1
    assert error in records[1]["exception"]
    assert error in result.stderr
    assert not list(run_dir.glob("*.json"))


def _run_logging(output_dir, source):
    prelude = """
        import logging
        import sys
        from pathlib import Path
        import structlog
        from src.logging import configure_logging
        output_dir = Path(sys.argv[1])
    """
    return subprocess.run(
        [sys.executable, "-c", dedent(prelude) + dedent(source), str(output_dir)],
        cwd=Path(__file__).resolve().parents[1], check=True, capture_output=True, text=True,
    )
