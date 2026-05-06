# Linear Regression MLOps Pipeline

A production-grade linear regression project with a complete MLOps pipeline using Docker, MLflow, FastAPI, Kubernetes, and GitHub Actions.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     GitHub Actions CI/CD                          │
├─────────────────────────────────────────────────────────────────┤
│  Push → Lint/Test → Train → Build Image → Deploy to K8s         │
└─────────────────────────────────────────────────────────────────┘

┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────────┐
│  Data    │───▶│ Training │───▶│  Model   │───▶│  Kubernetes  │
│ Pipeline │    │ Pipeline │    │ Registry │    │  Deployment  │
└──────────┘    └──────────┘    └──────────┘    └──────────────┘
     │               │               │                  │
     └───────────────┴───────────────┴──────────────────┘
                          MLflow Tracking
```

## Project Structure

```
linear-regression-mlops/
├── src/
│   ├── data/           # Data loading and preprocessing
│   ├── features/       # Feature engineering
│   ├── models/         # Model training and evaluation
│   └── serving/        # FastAPI inference server
├── tests/              # Unit and integration tests
├── configs/            # Configuration files
├── docker/             # Dockerfiles
├── k8s/                # Kubernetes manifests
├── .github/workflows/  # CI/CD pipelines
├── mlruns/             # MLflow tracking (local)
├── models/             # Saved model artifacts
└── data/               # Raw and processed data
```

## Quick Start

### Local Development

```bash
# Create virtual environment
python3.10 -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Train the model
python -m src.models.train

# Start API server
uvicorn src.serving.app:app --host 0.0.0.0 --port 8000

# Make a prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [[8.3252, 41.0, 6.984, 1.024, 322.0, 2.556, 37.88, -122.23]]}'
```

### Docker Compose (Full Stack)

```bash
docker compose up --build
# Training runs first, then API starts on :8000, MLflow on :5000
```

---

## End-to-End Deployment Guide (100% Free)

### Overview of the Free Pipeline

| Component | Free Service | What It Does |
|-----------|-------------|--------------|
| Source Code | GitHub | Version control |
| CI/CD | GitHub Actions | Lint, test, train, build |
| Container Registry | GitHub Container Registry (ghcr.io) | Store Docker images |
| Model Storage | GitHub Releases | Versioned model artifacts |
| Kubernetes Cluster | See options below | Run the API |

### Step 1: Push to GitHub

```bash
git init
git add .
git commit -m "Initial commit: ML pipeline with K8s deployment"
git remote add origin https://github.com/YOUR_USERNAME/linear-regression-mlops.git
git push -u origin main
```

### Step 2: GitHub Actions Runs Automatically

On push to `main`, the pipeline:
1. **CI** — Lints code with ruff, runs 19 unit tests
2. **Train** — Trains the model, uploads artifacts
3. **Build** — Builds Docker image, pushes to ghcr.io
4. **Release** — Creates a GitHub Release with model files
5. **Deploy** — Deploys to your Kubernetes cluster

### Step 3: Get a Free Kubernetes Cluster

#### Option A: Civo Cloud (Recommended)
- **$250 free credit** for new accounts (lasts months for small workloads)
- Real managed Kubernetes, fast provisioning (~90 seconds)
- Steps:
  1. Sign up at https://www.civo.com
  2. Create a 1-node cluster (Small instance)
  3. Download kubeconfig
  4. Base64 encode it: `cat kubeconfig | base64`
  5. Add as GitHub Secret: `KUBECONFIG`

#### Option B: Oracle Cloud Free Tier (Always Free)
- **Always free** — 4 ARM Ampere A1 cores, 24GB RAM
- Steps:
  1. Sign up at https://cloud.oracle.com/free
  2. Create an OKE cluster (Oracle Kubernetes Engine)
  3. Use the "Always Free" eligible shapes
  4. Download kubeconfig and add to GitHub Secrets

#### Option C: Killercoda / Play with Kubernetes (Temporary)
- **Free ephemeral clusters** for testing (4-hour sessions)
- https://killercoda.com/playgrounds/scenario/kubernetes
- Good for testing, not for persistent deployment

#### Option D: Local with Minikube or Kind
```bash
# Install minikube
brew install minikube  # macOS
minikube start

# Deploy
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/hpa.yaml

# Access the API
minikube service lr-model-api -n ml-serving
```

### Step 4: Configure GitHub Secrets

Go to your repo → Settings → Secrets and variables → Actions:

| Secret | Value |
|--------|-------|
| `KUBECONFIG` | Base64-encoded kubeconfig from your cluster |

The `GITHUB_TOKEN` is automatic — it handles ghcr.io authentication.

### Step 5: Trigger Deployment

```bash
# Any push to main triggers the full pipeline
git push origin main

# Or trigger manually from GitHub Actions tab
```

### Step 6: Verify

```bash
# Check pods
kubectl get pods -n ml-serving

# Port-forward to test locally
kubectl port-forward svc/lr-model-api 8000:80 -n ml-serving

# Test prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [[8.3252, 41.0, 6.984, 1.024, 322.0, 2.556, 37.88, -122.23]]}'
```

---

## Alternative: Deploy Without Kubernetes (Simpler)

If you want something simpler than K8s:

### Render.com (Free Tier)
1. Connect your GitHub repo at https://render.com
2. Select "New Web Service" → Docker
3. Set Dockerfile path: `docker/Dockerfile.render`
4. Deploy — it trains at build time and serves automatically

### Railway.app
1. Connect repo at https://railway.app
2. It auto-detects the Dockerfile
3. Set `PORT=8000` environment variable

---

## Development Commands

```bash
make help          # Show all commands
make install       # Install dependencies
make lint          # Run linter
make format        # Auto-format code
make test          # Run tests with coverage
make train         # Train the model
make serve         # Start API server (dev mode with reload)
make docker-up     # Start everything with Docker Compose
make clean         # Remove generated files
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check + model status |
| POST | `/predict` | Generate predictions |
| GET | `/model/info` | Model metadata and metrics |
| GET | `/docs` | Interactive API documentation (Swagger) |

## Configuration

Edit `configs/config.yaml` to customize:
- Model type (linear_regression, ridge)
- Hyperparameters
- Feature scaling method
- Train/test split ratio
- MLflow tracking URI
