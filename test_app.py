"""
Unit tests for the Personality Predictor API.
Run with: pytest test_app.py -v
"""
import pytest
from fastapi.testclient import TestClient

# rename app_v2.py → app.py before running
from app import app

client = TestClient(app)

INTROVERT_SAMPLE = {
    "Time_spent_Alone": 9.0, "Stage_fear": "Yes",
    "Social_event_attendance": 1.0, "Going_outside": 1.0,
    "Drained_after_socializing": "Yes", "Friends_circle_size": 2.0,
    "Post_frequency": 1.0
}
EXTROVERT_SAMPLE = {
    "Time_spent_Alone": 1.0, "Stage_fear": "No",
    "Social_event_attendance": 9.0, "Going_outside": 7.0,
    "Drained_after_socializing": "No", "Friends_circle_size": 14.0,
    "Post_frequency": 9.0
}

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "healthy"

def test_predict_introvert():
    r = client.post("/predict", json=INTROVERT_SAMPLE)
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"] == "Introvert"
    assert body["confidence"] > 0.7
    assert "probabilities" in body

def test_predict_extrovert():
    r = client.post("/predict", json=EXTROVERT_SAMPLE)
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"] == "Extrovert"
    assert body["confidence"] > 0.7

def test_probabilities_sum_to_one():
    r = client.post("/predict", json=INTROVERT_SAMPLE)
    p = r.json()["probabilities"]
    assert abs(p["Introvert"] + p["Extrovert"] - 1.0) < 1e-4

def test_invalid_range():
    bad = {**INTROVERT_SAMPLE, "Time_spent_Alone": 99.0}
    r = client.post("/predict", json=bad)
    assert r.status_code == 422   # Pydantic validation error

def test_invalid_binary_field():
    bad = {**INTROVERT_SAMPLE, "Stage_fear": "Maybe"}
    r = client.post("/predict", json=bad)
    assert r.status_code == 422

def test_missing_field():
    bad = {k:v for k,v in INTROVERT_SAMPLE.items() if k != "Post_frequency"}
    r = client.post("/predict", json=bad)
    assert r.status_code == 422

def test_root():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json()["status"] == "online"
