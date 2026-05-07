#  Linear Regression MLOps Pipeline : Complete Guide

> **A production-grade Machine Learning project that teaches the ENTIRE MLOps lifecycle — from wiring your model to deploying it with Docker, CI/CD, and serving predictions via API.**

## 📖 Table of Contents

1.  [What is MLOps and Why Should You Care?](#-what-is-mlops-and-why-should-you-care)
2.  [Architecture Overview](#-architecture-overview)
3.  [Project Structure Explained](#-project-structure-explained)
4.  [Prerequisites](#-prerequisites)
5.  [Step-by-Step Setup](#-step-by-step-setup)
6.  [Understanding Each Component](#-understanding-each-component)
7.  [Running Locally](#-running-locally)
8.  [Running with Docker](#-running-with-docker)
9.  [CI/CD with GitHub Actions](#-cicd-with-github-actions)
10. [Deploying for Free](#-deploying-for-free)
11. [Making Predictions](#-making-predictions)
12. [Common Errors and Fixes](#-common-errors-and-fixes)
13. [What to Add Next (Interview Gold)](#-what-to-add-next-interview-gold)

---

##  What is MLOps and Why Should You Care?

**MLOps = Machine Learning + DevOps**

In Academia, you train models in Jupyter notebooks. In the real world:
- Models need to be **retrained** automatically when new data arrives
- Models need to be **served** as APIs so apps can use them
- Models need to be **versioned** (like code) so you can roll back
- Models need to be **tested** before deployment
- Models need to be **monitored** in production

This project covers ALL of these concepts with a simple Linear Regression model so you can focus on the **engineering** rather than complex ML math.

### The MLOps Lifecycle (What This Project Implements)

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   1. CODE   │────▶│  2. BUILD   │────▶│  3. TRAIN   │────▶│  4. DEPLOY  │
│  Write ML   │     │  Docker +   │     │  Run model  │     │  Serve API  │
│  pipeline   │     │  CI/CD      │     │  + track    │     │  to users   │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
       │                                                            │
       └────────────────────── FEEDBACK LOOP ───────────────────────┘
                        (retrain weekly / on new data)
```

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                     GitHub Actions CI/CD Pipeline                   │
├─────────────────────────────────────────────────────────────────────┤
│  git push → Lint Code → Run Tests → Train Model → Build Docker      │
│           → Push to Registry → Create Release → Deploy              │
└─────────────────────────────────────────────────────────────────────┘

┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  📊 Data     │───▶│  🏋️ Training  │───▶│  📦 Model    │───▶│  🌐 Serving   │ 
│  Pipeline    │    │  Pipeline    │    │  Registry    │    │  (FastAPI)   │
│              │    │              │    │              │    │              │
│ Load data    │    │ Scale feats  │    │ model.joblib │    │ /predict     │
│ Split train/ │    │ Fit model    │    │ scaler.joblib│    │ /health      │
│ test         │    │ Evaluate     │    │ metadata.json│    │ /model/info  │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
        │                   │                   │                   │
        └───────────────────┴───────────────────┴───────────────────┘
                              MLflow Experiment Tracking
                         (logs params, metrics, artifacts)
```

---

## 📁 Project Structure Explained

```
linear-regression-mlops/
│
├── src/                          #  ALL source code lives here
│   ├── __init__.py
│   ├── data/                     # Step 1: Data handling
│   │   ├── loader.py            # Load datasets (CSV or sklearn)
│   │   └── preprocessor.py     # Train/test split
│   ├── features/                 # Step 2: Feature engineering
│   │   ├── engineering.py       # Scaling (StandardScaler, MinMax)
│   │   └── persistence.py      # Save/load artifacts
│   ├── models/                   # Step 3: Model training
│   │   └── train.py            # Full training pipeline + MLflow
│   └── serving/                  # Step 4: Model serving
│       └── app.py              # FastAPI server with /predict endpoint
│
├── tests/                        # Automated tests
│   ├── test_data.py             # Test data loading
│   ├── test_features.py        # Test feature scaling
│   ├── test_model.py           # Test model training
│   └── test_api.py             # Test API endpoints
│
├── configs/
│   └── config.yaml              #  All hyperparameters in one place
│
├── docker/                       #  Docker configurations
│   ├── Dockerfile.train         # Container for training
│   ├── Dockerfile.serve         # Container for serving
│   └── Dockerfile.render        # All-in-one (train + serve)
│
├── k8s/                          # Kubernetes manifests
│   ├── namespace.yaml
│   ├── deployment.yaml
│   ├── service.yaml
│   ├── ingress.yaml
│   └── hpa.yaml                 # Auto-scaling
│
├── .github/workflows/            # CI/CD Pipelines
│   ├── ci.yaml                  # Lint + Test on every push
│   ├── train-and-deploy.yaml   # Train + Build + Release
│   └── deploy-k8s.yaml         # Deploy to Kubernetes
│
├── models/                       #  Saved model artifacts (gitignored)
├── data/                         #  Raw data (gitignored)
├── mlruns/                       #  MLflow tracking (gitignored)
│
├── docker-compose.yaml           # Run everything with one command
├── Makefile                      # Shortcuts for common commands
├── requirements.txt              # Python dependencies (pinned versions)
├── pyproject.toml               # Project config (pytest, ruff, coverage)
└── README.md                    # You are here!
```

---

## 📋 Prerequisites

### Software You Need

| Tool | Why | Install |
|------|-----|---------|
| **Python 3.10+** | Run ML code | `brew install python@3.11` or [python.org](https://python.org) |
| **Git** | Version control | `brew install git` or [git-scm.com](https://git-scm.com) |
| **Docker** | Containerization | [docker.com/get-started](https://docs.docker.com/get-started/) |
| **GitHub Account** | CI/CD + hosting | [github.com](https://github.com) |

### Concepts You Should Know (Basics Are Enough)

- Python (functions, classes, imports)
- What a REST API is (GET, POST requests)
- What Git/GitHub is (push, pull, branches)
- Basic ML (what is linear regression, train/test split)

---

## Step-by-Step Setup

### Step 1: Clone the Repository

```bash
git clone https://github.com/sanjeevranjaniitb/lr-mlops-e2e.git
cd lr-mlops-e2e
```

### Step 2: Create a Virtual Environment

**Why?** Virtual environments isolate your project's dependencies from your system Python. Without this, package conflicts will ruin your day.

```bash
# Create virtual environment
python3 -m venv .venv

# Activate it
source .venv/bin/activate        # macOS/Linux
# .venv\Scripts\activate         # Windows

# Verify you're in the venv
which python
# Should show: .../lr-mlops-e2e/.venv/bin/python
```

### Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**What gets installed:**
- `scikit-learn` — ML algorithms (LinearRegression, StandardScaler)
- `pandas`, `numpy` — Data manipulation
- `mlflow` — Experiment tracking (logs your metrics)
- `fastapi`, `uvicorn` — Web server for serving predictions
- `pytest` — Testing framework
- `ruff` — Code linter (catches bugs before runtime)
- `joblib` — Save/load Python objects (models) to disk

### Step 4: Train the Model

```bash
python -m src.models.train
```

**What happens behind the scenes:**
1. Loads California Housing dataset (20,640 samples, 8 features)
2. Splits into 80% train / 20% test
3. Applies StandardScaler (zero mean, unit variance)
4. Fits LinearRegression
5. Evaluates: RMSE, MAE, R², MSE
6. Logs everything to MLflow
7. Saves `model.joblib`, `scaler.joblib`, `metadata.json` to `models/`

**Expected output:**
```
INFO - MLflow server unavailable, using local file tracking
INFO - Loading California Housing dataset...
INFO - Loaded dataset with shape: (20640, 9)
INFO - Train set size: 16512
INFO - Test set size: 4128
INFO - Fitted standard scaler on 8 features
INFO - Training linear_regression...
INFO - Metrics: {'rmse': 0.7456, 'mae': 0.5332, 'r2': 0.5758, 'mse': 0.5559}
INFO - Model saved to .../models/model.joblib
INFO - Training complete!
```

### Step 5: Start the API Server

```bash
uvicorn src.serving.app:app --host 0.0.0.0 --port 8000 --reload
```

Open http://localhost:8000/docs in your browser — you'll see the interactive Swagger UI!

### Step 6: Make Your First Prediction

Open a new terminal:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [[8.3252, 41.0, 6.984, 1.024, 322.0, 2.556, 37.88, -122.23]]}'
```

**Response:**
```json
{
  "predictions": [4.152],
  "model_type": "linear_regression",
  "feature_names": ["MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population", "AveOccup", "Latitude", "Longitude"]
}
```

The prediction `4.152` means the model estimates the median house value is **$415,200** (values are in $100k units).

### Step 7: Run Tests

```bash
pytest tests/ -v
```

All 19 tests should pass. These test:
- Data loading works correctly
- Feature scaling produces expected results
- Model training returns valid metrics
- API endpoints return correct responses

---

## 🔍 Understanding Each Component

### 1. Data Pipeline (`src/data/`)

**`loader.py`** — Loads data from sklearn or CSV files.

```python
# This is what happens inside:
from sklearn.datasets import fetch_california_housing
df = fetch_california_housing(as_frame=True).frame
# Returns a DataFrame with 20,640 rows and 9 columns
```

**`preprocessor.py`** — Splits data into train/test sets.

```python
# 80% for training, 20% for testing
# random_state=42 ensures reproducibility (same split every time)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

**Why split?** You train on 80% and evaluate on the 20% the model has never seen. This tells you how well the model generalizes to new data.

---

### 2. Feature Engineering (`src/features/`)

**`engineering.py`** — Scales features to have zero mean and unit variance.

```python
# Before scaling: MedInc ranges [0.5, 15], Population ranges [3, 35682]
# After scaling: Both have mean=0, std=1
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)
```

**Why scale?** Linear regression is sensitive to feature magnitudes. Without scaling, features with large values (like Population) dominate the model.

**Important:** You `fit` the scaler on training data only, then `transform` both train and test. Never fit on test data (that's data leakage!).

---

### 3. Model Training (`src/models/train.py`)

The training pipeline does everything in sequence:

```python
def train():
    # 1. Load config (hyperparameters from YAML)
    # 2. Setup MLflow tracking
    # 3. Load data
    # 4. Split into train/test
    # 5. Scale features
    # 6. Train model
    # 7. Evaluate (RMSE, MAE, R², MSE)
    # 8. Log metrics to MLflow
    # 9. Save model artifacts to disk
```

**MLflow** tracks every training run — parameters, metrics, and artifacts. Think of it as "Git for ML experiments."

---

### 4. Model Serving (`src/serving/app.py`)

FastAPI creates a REST API with these endpoints:

| Endpoint | Method | What It Does |
|----------|--------|--------------|
| `/health` | GET | Is the server alive? Is the model loaded? |
| `/predict` | POST | Send features, get predictions back |
| `/model/info` | GET | What model is loaded? What are its metrics? |
| `/docs` | GET | Interactive API documentation (Swagger UI) |

**How `/predict` works internally:**
```
Request comes in → Validate input → Scale features → model.predict() → Return JSON
```

---

### 5. Configuration (`configs/config.yaml`)

All hyperparameters live in ONE file. No magic numbers scattered in code.

```yaml
model:
  type: "linear_regression"      # Change to "ridge" for L2 regularization
  hyperparameters:
    fit_intercept: true

features:
  scaling: "standard"            # Options: standard, minmax, none

data:
  test_size: 0.2                 # 20% test split
  random_state: 42               # Reproducibility
```

**Why YAML config?** In production, you change hyperparameters without touching code. This is how real ML teams work.

---

### 6. Testing (`tests/`)

Tests ensure your code doesn't break when you make changes.

```python
# Example: test that scaling produces zero mean
def test_standard_scaling(sample_data):
    fe = FeatureEngineer(scaling="standard")
    transformed = fe.fit_transform(sample_data)
    assert np.allclose(transformed.mean(axis=0), 0, atol=1e-10)
```

**Run tests before every commit.** CI/CD does this automatically.

---

##  Running with Docker

### What is Docker?

Docker packages your app + all dependencies into a **container** — a lightweight, portable box that runs the same everywhere (your laptop, cloud server, Kubernetes).

**Without Docker:** "It works on my machine" 
**With Docker:** "It works everywhere" 

### Option 1: Run the Pre-Built Image (Easiest)

This image is already built and hosted on GitHub Container Registry:

```bash
docker run -p 8000:8000 ghcr.io/sanjeevranjaniitb/lr-mlops-e2e:latest
```

That's it. Open http://localhost:8000/docs.

**What just happened:**
1. Docker pulled the image from GitHub's container registry
2. Started a container with the trained model inside
3. Exposed port 8000 so you can access the API

### Option 2: Build and Run Locally

```bash
# Build the training image
docker build -f docker/Dockerfile.train -t lr-train .

# Run training (saves model to ./models/)
docker run -v $(pwd)/models:/app/models lr-train

# Build the serving image
docker build -f docker/Dockerfile.serve -t lr-serve .

# Run the API server
docker run -p 8000:8000 lr-serve
```

### Option 3: Docker Compose (Full Stack)

```bash
docker compose up --build
```

This starts:
- **Trainer** — trains the model, then exits
- **API** — starts after training completes, serves on :8000
- **MLflow** — experiment tracking UI on :5000

### Understanding the Dockerfiles

**`Dockerfile.train`** — Trains the model:
```dockerfile
FROM python:3.11-slim          # Start with Python 3.11
WORKDIR /app                   # Set working directory
COPY requirements.txt .        # Copy deps file
RUN pip install -r requirements.txt  # Install deps
COPY src/ src/                 # Copy source code
COPY configs/ configs/         # Copy config
CMD ["python", "-m", "src.models.train"]  # Run training
```

**`Dockerfile.serve`** — Serves predictions:
```dockerfile
FROM python:3.11-slim
# ... (same setup) ...
COPY models/ models/           # Copy trained model
EXPOSE 8000                    # Document the port
CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

**`Dockerfile.render`** — Multi-stage build (trains AND serves):
```dockerfile
# Stage 1: Train
FROM python:3.11-slim AS trainer
# ... install deps, copy code, train model ...

# Stage 2: Serve (smaller final image)
FROM python:3.11-slim
COPY --from=trainer /app/models/ models/  # Copy model from stage 1
CMD ["uvicorn", ...]
```

---

##  CI/CD with GitHub Actions

### What is CI/CD?

- **CI (Continuous Integration):** Every time you push code, automated tests run. If tests fail, you know immediately.
- **CD (Continuous Deployment):** If tests pass, the code is automatically deployed to production.

### Our Pipeline (3 Workflows)

#### Workflow 1: `ci.yaml` — Lint & Test (runs on every push)

```
Push to GitHub → Install deps → Lint code (ruff) → Run tests (pytest) → Report coverage
```

**What is linting?** A linter (ruff) checks your code for:
- Unused imports
- Undefined variables
- Style violations
- Common bugs

#### Workflow 2: `train-and-deploy.yaml` — Train & Deploy (runs on push to main)

```
Push to main → Train model → Smoke test → Build Docker image → Push to GHCR → Create Release
```

**Step by step:**
1. **Train:** Runs `python -m src.models.train` on GitHub's servers
2. **Smoke test:** Loads the model and makes one prediction to verify it works
3. **Build:** Creates a Docker image with the trained model baked in
4. **Push:** Uploads the image to `ghcr.io/sanjeevranjaniitb/lr-mlops-e2e:latest`
5. **Release:** Creates a GitHub Release with model files attached

#### Workflow 3: `deploy-k8s.yaml` — Deploy to Kubernetes (optional)

```
Train workflow completes → Pull kubeconfig → Apply K8s manifests → Verify rollout
```

### How to See Your Pipelines

1. Go to https://github.com/sanjeevranjaniitb/lr-mlops-e2e/actions
2. You'll see each workflow run with ✅ (pass) or ❌ (fail)
3. Click on any run to see detailed logs

### GitHub Actions Key Concepts

```yaml
name: CI - Lint & Test          # Name shown in GitHub UI

on:                              # WHEN does this run?
  push:
    branches: [main]            # On push to main branch

jobs:                            # WHAT does it do?
  lint-and-test:
    runs-on: ubuntu-latest      # Use a free Linux VM
    steps:
      - uses: actions/checkout@v4        # Get your code
      - uses: actions/setup-python@v5    # Install Python
      - run: pip install -r requirements.txt  # Install deps
      - run: ruff check src/ tests/      # Lint
      - run: pytest tests/ -v            # Test
```

**Free limits:** Public repos get unlimited GitHub Actions minutes. Private repos get 2,000 min/month free.

---

## ☸️ Deploying to Kubernetes (Step-by-Step)

### What is Kubernetes?

Kubernetes (K8s) is a container orchestration platform. It:
- Runs multiple copies of your app (replicas) for high availability
- Automatically restarts crashed containers
- Scales up/down based on traffic (HPA)
- Does rolling updates (zero-downtime deployments)

Think of it as a "smart Docker" that manages containers across multiple machines.

### Key Kubernetes Concepts

| Concept | What It Is | Analogy |
|---------|-----------|---------|
| **Pod** | Smallest unit — one running container | A single server process |
| **Deployment** | Manages pods — ensures N replicas are running | A fleet manager |
| **Service** | Stable network endpoint for pods | A load balancer |
| **Namespace** | Logical isolation (like folders) | A project folder |
| **HPA** | Auto-scales pods based on CPU/memory | Auto-scaling group |
| **Ingress** | Routes external traffic to services | A reverse proxy (nginx) |

### Our K8s Architecture

```
                    Internet
                       │
                 ┌─────▼─────┐
                 │  Ingress  │  (routes traffic)
                 └─────┬─────┘
                       │
                 ┌─────▼─────┐
                 │  Service  │  (load balances)
                 │  :80      │
                 └──┬─────┬──┘
                    │     │
              ┌─────▼──┐ ┌▼──────┐
              │  Pod 1 │ │ Pod 2 │  (2 replicas)
              │  :8000 │ │ :8000 │
              └────────┘ └───────┘
                    │
              ┌─────▼─────┐
              │    HPA    │  (scales 1-4 pods)
              └───────────┘
```

---

### Option A: Deploy Locally with Minikube

#### Step 1: Install Minikube

```bash
# macOS
brew install minikube

# Linux
curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64
sudo install minikube-linux-amd64 /usr/local/bin/minikube

# Windows (PowerShell as Admin)
choco install minikube
```

#### Step 2: Start the Cluster

```bash
minikube start --driver=docker
```

This creates a single-node Kubernetes cluster inside Docker. Takes ~2 minutes on first run.

#### Step 3: Load the Docker Image into Minikube

Minikube has its own Docker daemon. You need to make your image available inside it:

```bash
# Pull from GHCR and load into minikube
docker pull ghcr.io/sanjeevranjaniitb/lr-mlops-e2e:latest
minikube image load ghcr.io/sanjeevranjaniitb/lr-mlops-e2e:latest
```

#### Step 4: Deploy the Application

```bash
# Create namespace (logical isolation)
kubectl apply -f k8s/namespace.yaml

# Create deployment (runs 2 pods)
kubectl apply -f k8s/deployment.yaml

# Create service (load balancer)
kubectl apply -f k8s/service.yaml

# Create auto-scaler
kubectl apply -f k8s/hpa.yaml
```

#### Step 5: Verify Everything is Running

```bash
# Check pods (should show 2/2 Running)
kubectl get pods -n ml-serving

# Check service
kubectl get svc -n ml-serving

# Check auto-scaler
kubectl get hpa -n ml-serving
```

**Expected output:**
```
NAME                            READY   STATUS    RESTARTS   AGE
lr-model-api-66f9d7b7c6-bj5tw   1/1     Running   0          2m
lr-model-api-66f9d7b7c6-xnz26   1/1     Running   0          2m
```

#### Step 6: Access the API

```bash
# Port-forward the service to localhost
kubectl port-forward svc/lr-model-api 9000:80 -n ml-serving
```

Now test it:
```bash
# Health check
curl http://localhost:9000/health

# Make a prediction
curl -X POST http://localhost:9000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [[8.3252, 41.0, 6.984, 1.024, 322.0, 2.556, 37.88, -122.23]]}'
```

#### Step 7: Useful kubectl Commands

```bash
# View pod logs
kubectl logs -f deployment/lr-model-api -n ml-serving

# Describe a pod (debug issues)
kubectl describe pod <pod-name> -n ml-serving

# Scale manually
kubectl scale deployment/lr-model-api --replicas=3 -n ml-serving

# Delete everything
kubectl delete namespace ml-serving

# Stop minikube (preserves state)
minikube stop

# Delete minikube cluster entirely
minikube delete
```

---

### Option B: Deploy to Cloud Kubernetes (Free Tiers)

#### Civo Cloud ($250 Free Credit)

1. Sign up at https://www.civo.com (GitHub login works)
2. Create cluster: Dashboard → Kubernetes → Create
   - Name: `lr-mlops`
   - Nodes: 1 × Small
   - Wait ~90 seconds
3. Download kubeconfig file
4. Use it:
   ```bash
   export KUBECONFIG=~/Downloads/civo-lr-mlops-kubeconfig
   kubectl apply -f k8s/namespace.yaml
   kubectl apply -f k8s/deployment.yaml
   kubectl apply -f k8s/service.yaml
   kubectl apply -f k8s/hpa.yaml
   ```

#### Automate with GitHub Actions

1. Base64 encode your kubeconfig:
   ```bash
   cat ~/Downloads/civo-lr-mlops-kubeconfig | base64
   ```
2. Add as GitHub Secret: repo → Settings → Secrets → `KUBECONFIG`
3. The `deploy-k8s.yaml` workflow will auto-deploy on every successful training run

---

### Understanding the K8s Manifest Files

#### `namespace.yaml` — Creates an isolated space
```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: ml-serving    # All our resources go here
```

#### `deployment.yaml` — Runs your containers
```yaml
spec:
  replicas: 2                    # Run 2 copies for high availability
  template:
    spec:
      containers:
        - name: api
          image: ghcr.io/sanjeevranjaniitb/lr-mlops-e2e:latest
          resources:
            requests:
              memory: "128Mi"    # Minimum memory guaranteed
              cpu: "100m"        # 0.1 CPU cores minimum
            limits:
              memory: "256Mi"    # Maximum memory allowed
              cpu: "500m"        # 0.5 CPU cores maximum
          livenessProbe:         # K8s restarts pod if this fails
            httpGet:
              path: /health
              port: 8000
          readinessProbe:        # K8s only sends traffic if this passes
            httpGet:
              path: /health
              port: 8000
```

#### `service.yaml` — Load balances across pods
```yaml
spec:
  type: ClusterIP              # Internal only (use Ingress for external)
  selector:
    app: lr-model-api          # Routes to pods with this label
  ports:
    - port: 80                 # Service listens on :80
      targetPort: 8000         # Forwards to container :8000
```

#### `hpa.yaml` — Auto-scales based on CPU
```yaml
spec:
  minReplicas: 1               # Minimum 1 pod
  maxReplicas: 4               # Maximum 4 pods
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          averageUtilization: 70  # Scale up when CPU > 70%
```

---

##  Deploying for Free (Without Kubernetes)

### Render.com (Simplest — 5 clicks)

1. Go to https://render.com → Sign up with GitHub
2. Click **"New" → "Web Service"**
3. Connect repo: `sanjeevranjaniitb/lr-mlops-e2e`
4. Settings:
   - **Dockerfile Path:** `docker/Dockerfile.render`
   - **Plan:** Free
5. Click Deploy

You get a URL like `https://lr-mlops-api.onrender.com`

>  Free tier sleeps after 15 min inactivity. First request after sleep takes ~30s.

### Run Anywhere with Docker

Anyone can run your model with one command:
```bash
docker run -p 8000:8000 ghcr.io/sanjeevranjaniitb/lr-mlops-e2e:latest
```

---

##  Making Predictions

### Using curl

```bash
# Single prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [[8.3252, 41.0, 6.984, 1.024, 322.0, 2.556, 37.88, -122.23]]}'

# Batch prediction (multiple samples)
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [[8.3, 41.0, 6.9, 1.0, 322.0, 2.5, 37.88, -122.23], [3.5, 25.0, 5.5, 1.1, 1500.0, 3.0, 34.05, -118.25]]}'
```

### Using Python

```python
import requests

response = requests.post(
    "http://localhost:8000/predict",
    json={"features": [[8.3252, 41.0, 6.984, 1.024, 322.0, 2.556, 37.88, -122.23]]}
)
print(response.json())
# {'predictions': [4.152], 'model_type': 'linear_regression', ...}
```

### Feature Descriptions

| Feature | Description | Example |
|---------|-------------|---------|
| MedInc | Median income (in $10k) | 8.3252 |
| HouseAge | Median house age (years) | 41.0 |
| AveRooms | Average rooms per household | 6.984 |
| AveBedrms | Average bedrooms per household | 1.024 |
| Population | Block group population | 322.0 |
| AveOccup | Average occupancy | 2.556 |
| Latitude | Block group latitude | 37.88 |
| Longitude | Block group longitude | -122.23 |

**Output:** Predicted median house value in $100k units. So `4.152` = **$415,200**.

---

##  Common Errors and Fixes

| Error | Cause | Fix |
|-------|-------|-----|
| `ModuleNotFoundError: No module named 'mlflow'` | Wrong Python environment | Activate venv: `source .venv/bin/activate` |
| `FileNotFoundError: Model not found` | Haven't trained yet | Run `python -m src.models.train` |
| `coverage: FAIL Required test coverage not reached` | Coverage threshold too high | Lower `fail_under` in `pyproject.toml` |
| `denied: installation not allowed to Create organization package` | GHCR permissions | Add `permissions: packages: write` to workflow |
| `ErrImagePull` in Kubernetes | Image not accessible | Use `minikube image load` or set `imagePullPolicy: IfNotPresent` |
| `Connection refused` on port 8000 | Server not started | Start with `uvicorn src.serving.app:app --port 8000` |
| MLflow hangs on import (Python 3.13) | Version incompatibility | Use Python 3.10 or 3.11 |

---

##  What to Add Next (Interview Gold)

Once you've mastered this project, here's what to add to impress in interviews:

### Level 1: Quick Wins
- [ ] Add data validation with Great Expectations or Pydantic
- [ ] Add Prometheus metrics endpoint (`/metrics`)
- [ ] Add request logging with structured JSON logs
- [ ] Add A/B testing (serve two models, compare)

### Level 2: Production Features
- [ ] Model monitoring (detect data drift with Evidently AI)
- [ ] Feature store (Feast)
- [ ] Canary deployments (gradually shift traffic to new model)
- [ ] Add authentication (API keys or JWT)
- [ ] Rate limiting

### Level 3: Advanced MLOps
- [ ] Multi-model serving (serve different models per endpoint)
- [ ] Online learning (update model with streaming data)
- [ ] Shadow mode (run new model alongside old, compare silently)
- [ ] GPU inference with ONNX Runtime
- [ ] Distributed training with Ray

### Level 4: Platform Engineering
- [ ] Terraform for infrastructure as code
- [ ] ArgoCD for GitOps-based K8s deployments
- [ ] Istio service mesh for traffic management
- [ ] Grafana dashboards for model performance

---

##  Complete Pipeline Summary

```
You write code
    │
    ▼
git push to GitHub
    │
    ▼
GitHub Actions triggers automatically
    │
    ├── CI: ruff lint → pytest (19 tests) → coverage check
    │
    ├── Train: load data → scale → fit model → evaluate → save artifacts
    │
    ├── Build: Docker image with model baked in
    │
    ├── Push: Image → ghcr.io/sanjeevranjaniitb/lr-mlops-e2e:latest
    │
    ├── Release: model.joblib + metadata.json → GitHub Releases
    │
    └── Deploy: kubectl apply → Kubernetes cluster (optional)
              │
              ▼
        API is live at /predict
              │
              ▼
        Users send features, get predictions back
```

**Total cost: $0** (GitHub Actions free for public repos, GHCR free, Minikube free)

---

##  Contributing

1. Fork the repo
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Make changes and run tests: `make test`
4. Push and create a PR

## 📄 License

MIT — use this however you want.
