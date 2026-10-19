import mlflow
import pytest


@pytest.fixture
def tracking_store(tmp_path, monkeypatch) -> mlflow.MlflowClient:
    """A throwaway MLflow store named by ``MLFLOW_TRACKING_URI``.

    The default experiment name, ``run``, is created up front with its artifacts
    under ``tmp_path``. MLflow would otherwise resolve a new experiment's
    artifact root against the working directory, i.e. inside the repository.
    """
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    client = mlflow.MlflowClient(uri)
    client.create_experiment("run", artifact_location=(tmp_path / "mlartifacts").as_uri())
    return client
