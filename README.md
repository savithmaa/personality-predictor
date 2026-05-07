# Personality Predictor API

 **Live API Endpoint:** [https://personality-predictor-production-e454.up.railway.app](https://personality-predictor-production-e454.up.railway.app)
 **API Documentation (Swagger UI):** [https://personality-predictor-production-e454.up.railway.app/docs](https://personality-predictor-production-e454.up.railway.app/docs)

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

### Deploy to Railway (Current Deployment)
1. Push this repo to GitHub
2. Go to **Railway.app** → New Project → Deploy from GitHub repo
3. Railway will automatically detect the Dockerfile and build the project
4. Go to Networking → Generate Domain

## API

### POST /predict
```bash
curl -X POST "https://personality-predictor-production-e454.up.railway.app/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "Time_spent_Alone": 7.0,
    "Stage_fear": "Yes",
    "Social_event_attendance": 2.0,
    "Going_outside": 2.0,
    "Drained_after_socializing": "Yes",
    "Friends_circle_size": 3.0,
    "Post_frequency": 2.0
  }'
```

Response:
```json
{
  "prediction": "Introvert",
  "confidence": 0.8462,
  "probabilities": {
    "Introvert": 0.8462,
    "Extrovert": 0.1538
  },
  "model_version": "1.0.0"
}
```

## Run Tests
```bash
pytest test_app.py -v
```

## Files
| File | Purpose |
|---|---|
| `app.py` | FastAPI application |
| `personality_model.pkl` | Trained pipeline (all preprocessing inside) |
| `feature_names.pkl` | Feature order |
| `Dockerfile` | Container definition |
| `test_app.py` | Pytest unit tests |
| `requirements.txt` | Dependencies |
