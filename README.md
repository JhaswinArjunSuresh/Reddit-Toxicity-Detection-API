# 🔎 Reddit Toxicity Detection API

Detects toxicity in text using a fine-tuned transformer.

## Endpoints
- `POST /classify` with `{ "text": "your reddit comment here" }`
- `GET /health` for health check.

## Run locally
```bash
pip install -r requirements.txt
uvicorn app.main:app --reload

