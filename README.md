# Personality Predictor API

Random Forest classifier predicting Introvert / Extrovert from behavioural features.

**Test Accuracy:** 92.4% | **ROC-AUC:** 0.956 | **Brier Score:** 0.067

## Quick Start

### Run locally
```bash
pip install -r requirements.txt
uvicorn app:app --reload
# Docs → http://localhost:8000/docs
```

### Run with Docker
```bash
docker build -t personality-api .
docker run -p 8000:8000 personality-api
```

### Deploy to Render (free)
1. Push this repo to GitHub
2. render.com → New → Web Service → connect repo
3. Runtime = Docker → Deploy
4. Get your public URL → done

## API

### POST /predict
```json
{
  "Time_spent_Alone": 7.0,
  "Stage_fear": "Yes",
  "Social_event_attendance": 2.0,
  "Going_outside": 2.0,
  "Drained_after_socializing": "Yes",
  "Friends_circle_size": 3.0,
  "Post_frequency": 2.0
}
```

Response:
```json
{
  "prediction": "Introvert",
  "confidence": 0.934,
  "probabilities": {"Introvert": 0.934, "Extrovert": 0.066},
  "model_version": "1.0.0"
}
```

## Run Tests
```bash
# rename app_v2.py to app.py first
pytest test_app.py -v
```

## Files
| File | Purpose |
|---|---|
| `app.py` | FastAPI application |
| `personality_model.pkl` | Trained pipeline (all preprocessing inside) |
| `feature_names.pkl` | Feature order |
| `Dockerfile` | Container definition |
| `render.yaml` | Render deployment config |
| `test_app.py` | Pytest unit tests |
| `requirements.txt` | Dependencies |
