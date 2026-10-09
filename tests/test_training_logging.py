"""Progress logging changes output cadence, never optimization or evaluation."""

import json
import subprocess
import sys
from pathlib import Path
from textwrap import dedent

import pytest


@pytest.mark.parametrize("order,iterations,accepted,cadence,verbose,expected_progress", [
    ("insertion_first", 4, True, None, True, [0, 2]),
    ("correction_first", 4, True, 2, True, [0, 2]),
    ("insertion_first", 4, True, 1, True, [0, 1, 2, 3]),
    ("insertion_first", 4, True, 0, True, []),
    ("insertion_first", 4, True, -1, True, []),
    ("insertion_first", 4, True, 2, False, []),
    ("insertion_first", 0, True, 2, True, [0]),
    ("insertion_first", 4, False, 2, True, [0]),
])
def test_progress_and_terminal_summary_preserve_all_checkpoints(
    tmp_path, order, iterations, accepted, cadence, verbose, expected_progress,
):
    settings = dict(order=order, iterations=iterations, accepted=accepted, verbose=verbose)
    if cadence is not None:
        settings["display_every"] = cadence
    result = _train_with_logging(tmp_path, settings)
    report = json.loads(result.stdout)
    records = [json.loads(line) for line in (tmp_path / "training.jsonl").read_text().splitlines()]
    checkpoints = max(iterations if accepted else 0, 1)
    assert report["evaluations"] == report["checkpoints"] == report["snapshots"] == checkpoints

    progress = [record for record in records if record["event"] == "training_progress"]
    assert [record["checkpoint"] for record in progress] == expected_progress
    assert [record["train_loss"] for record in progress] == [
        report["losses"][i] for i in expected_progress
    ]
    terminal = [record for record in records if record["event"] == "training_finished"]
    assert len(terminal) == int(verbose)
    if verbose:
        assert records[-1] == terminal[0]
        assert terminal[0]["checkpoints"] == checkpoints
        assert terminal[0]["stop_reason"] == ("iteration_budget" if accepted else "no_candidate")
        assert terminal[0]["best_iteration"] == report["best_iteration"]
        assert terminal[0]["final_neurons"] == report["final_neurons"]
    # SSN details are DEBUG; default INFO contains only outer progress and summary.
    assert len(records) == len(progress) + len(terminal)
    assert result.stderr.count("training_progress") == len(progress)
    assert result.stderr.count("training_finished") == len(terminal)


def _train_with_logging(output_dir, settings):
    source = """
        import json
        import sys
        from pathlib import Path
        import torch
        import src.training as training
        from src.config import resolve_model
        from src.config.schema import TrainingConfig
        from src.data import make_dataloader
        from src.logging import configure_logging
        from src.PDAP import PDAP
        from src.problems import Proximal, SparseProblem

        output_dir, settings = Path(sys.argv[1]), json.loads(sys.argv[2])
        configure_logging(log_file=output_dir / "training.jsonl")
        problem = SparseProblem(resolve_model({"alpha": 0.0}, 1))
        x = torch.tensor([[0.25], [0.75]], dtype=torch.float64)
        loader = make_dataloader((x, x.clone(), torch.ones_like(x)))
        proximal = Proximal(0.0, 0.0, problem.penalty_exponent)
        pdap = PDAP(TrainingConfig(loop_order=settings.pop("order"), fit_outer_iterations=0), proximal)
        size = int(settings.pop("accepted"))
        pdap._insert = lambda *args, **kwargs: (
            x.new_ones((size, 1)), x.new_zeros(size), x.new_ones(size),
        )
        evaluations = 0
        measure = training.metrics
        def count_evaluation(*args, **kwargs):
            global evaluations
            evaluations += 1
            return measure(*args, **kwargs)
        training.metrics = count_evaluation
        history = training.run_training(
            pdap, problem, loader, num_iterations=settings.pop("iterations"), **settings,
        )
        print(json.dumps({
            "evaluations": evaluations,
            "checkpoints": len(history.train_loss),
            "snapshots": len(history.model_states),
            "losses": history.train_loss,
            "best_iteration": history.best_iteration,
            "final_neurons": history.final_neurons,
        }))
    """
    return subprocess.run(
        [sys.executable, "-c", dedent(source), str(output_dir), json.dumps(settings)],
        cwd=Path(__file__).resolve().parents[1],
        check=True, capture_output=True, text=True, timeout=30,
    )
