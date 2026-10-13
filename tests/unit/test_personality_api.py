from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient

from src.api.app import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_endpoint_returns_all_traits():
    response = client.post("/predict", json={"text": "I enjoy being around people and making plans with friends."})
    assert response.status_code == 200
    payload = response.json()
    assert set(payload["traits"].keys()) == {
        "agreeableness",
        "openness",
        "conscientiousness",
        "extraversion",
        "neuroticism",
    }
    assert all(0.0 <= value <= 100.0 for value in payload["traits"].values())


def test_predict_batch_endpoint():
    response = client.post(
        "/predict/batch",
        json={
            "texts": [
                "I like structure and careful planning before I act.",
                "I am energetic and enjoy engaging with other people.",
            ]
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert len(payload) == 2
    for item in payload:
        assert set(item["traits"].keys()) == {
            "agreeableness",
            "openness",
            "conscientiousness",
            "extraversion",
            "neuroticism",
        }


def test_summarize_endpoint_returns_summary():
    response = client.post(
        "/summarize",
        json={
            "text": (
                "I enjoy planning trips with friends. "
                "I keep a detailed schedule for each day. "
                "I also like discovering new places and trying new activities."
            )
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["summary"]
    assert "strongest predicted dimension" in payload["summary"]
    assert "not a clinical or definitive personality assessment" in payload["summary"]
