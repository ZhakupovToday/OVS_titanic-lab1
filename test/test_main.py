# -*- coding: utf-8 -*-
import numpy as np
import pytest
from fastapi.testclient import TestClient

PASSENGER = {"pclass": 1, "sex": "female", "age": 29, "sibsp": 0,
             "parch": 0, "fare": 211.34, "embarked": "S"}
GOOD = {"Authorization": "Bearer 00000"}


class FakeModel:
    """Заглушка вместо обученного пайплайна."""

    def __init__(self, probability: float) -> None:
        self.probability = probability

    def predict_proba(self, frame):
        return np.array([[1 - self.probability, self.probability]])


@pytest.fixture
def client(monkeypatch) -> TestClient:
    monkeypatch.setenv("MODEL_PATH", "faked/model.pkl")
    import main
    main.app.dependency_overrides[main.get_model] = lambda: FakeModel(0.93)
    yield TestClient(main.app)
    main.app.dependency_overrides.clear()


def test_healthcheck(client) -> None:
    response = client.get("/healthcheck")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_inference(client) -> None:
    response = client.post("/predictions", headers=GOOD, json=PASSENGER)
    assert response.status_code == 200
    assert response.json() == {"survived": True, "probability": 0.93}


def test_inference_with_missing_values(client) -> None:
    passenger = {"pclass": 3, "sex": "male", "sibsp": 0, "parch": 0}
    response = client.post("/predictions", headers=GOOD, json=passenger)
    assert response.status_code == 200


def test_token_not_correct(client) -> None:
    response = client.post("/predictions", json=PASSENGER,
                           headers={"Authorization": "Bearer kedjkj"})
    assert response.status_code == 401
    assert response.json() == {"detail": "Invalid authentication credentials"}


def test_token_absent(client) -> None:
    response = client.post("/predictions", json=PASSENGER)
    assert response.status_code == 401
    assert response.json() == {"detail": "Not authenticated"}


@pytest.mark.parametrize("field, value", [
    ("pclass", 4),
    ("sex", "unknown"),
    ("age", -5),
    ("embarked", "X"),
    ("sibsp", "many"),
])
def test_invalid_input(client, field, value) -> None:
    response = client.post("/predictions", headers=GOOD,
                           json={**PASSENGER, field: value})
    assert response.status_code == 422


def test_required_field_absent(client) -> None:
    passenger = {k: v for k, v in PASSENGER.items() if k != "sex"}
    response = client.post("/predictions", headers=GOOD, json=passenger)
    assert response.status_code == 422
