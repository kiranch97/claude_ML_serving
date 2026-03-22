# claude_ML_serving: Model Serving on Kubernetes

> P3 MLOps — FastAPI Inference Server  
> Part of the kiranch97 Solutions Architecture

---

## Overview

Deploys a **FastAPI-based ML model serving platform** on Kubernetes that:

- Loads trained models from MLflow registry (claude_ML_project3)
- Exposes REST API endpoints for inference
- Supports hot-reload of models without pod restart
- Health checks, metrics, and structured JSON responses

---

## Architecture

```
  Client / Streamlit / Jupyter
              |
     http://localhost:30086
     http://serving.local:30080
              |
  +-----------v-----------+
  |  FastAPI Serving      |  Deployment (1 replica)
  |  port: 8000           |  Image: python:3.11-slim
  |                       |  Loads model from MLflow
  +-----------+-----------+
              |
  +-----------v-----------+
  |  MLflow Registry      |  (claude_ML_project3)
  |  http://mlflow:5000   |  namespace: mlflow
  +-----------------------+
```

---

## Directory Structure

```
claude_ML_serving/
├── README.md
├── app/
│   ├── main.py              # FastAPI inference server
│   ├── requirements.txt     # Python dependencies
│   └── Dockerfile           # Container image
└── k8s/
    ├── namespace.yaml
    ├── deployment.yaml      # FastAPI serving deployment
    ├── service-nodeport.yaml # NodePort: 30086
    └── ingress.yaml         # serving.local
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Liveness + model status |
| GET | `/model-info` | Loaded model metadata |
| POST | `/predict` | Run inference |
| POST | `/reload` | Hot-reload model from MLflow |
| GET | `/docs` | Swagger UI |

### Example Request

```bash
curl -X POST http://localhost:30086/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Congratulations! You have won a free prize!"}'
```

### Example Response

```json
{
  "prediction": "spam",
  "confidence": 0.9821,
  "probabilities": {"ham": 0.0179, "spam": 0.9821},
  "model_version": "SpamClassifier/2",
  "inference_ms": 12.4
}
```

---

## Deploy

```bash
git clone https://github.com/kiranch97/claude_ML_serving.git
cd claude_ML_serving
kubectl apply -f k8s/
```

Add to /etc/hosts: `127.0.0.1  serving.local`

---

## Access

| Service | URL |
|---------|-----|
| API (NodePort) | http://localhost:30086 |
| Swagger UI | http://localhost:30086/docs |
| Via Ingress | http://serving.local:30080 |

---

## Author
**kiranch97** — Built collaboratively with Claude AI | March 2026
