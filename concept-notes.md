# 📘 MLOps Concept Notes — Docker, Kubernetes, CI/CD for ML at Scale

> **Goal:** After reading this document, you will be able to deploy any ML model to production with Docker, Kubernetes, CI/CD, and understand how to control throughput, latency, and scalability on AWS.

---

## Table of Contents

1. [Part 1: Docker for ML — From Zero to Container](#part-1-docker-for-ml--from-zero-to-container)
2. [Part 2: CI/CD — Every Command Explained](#part-2-cicd--every-command-explained)
3. [Part 3: Kubernetes — Orchestrating ML at Scale](#part-3-kubernetes--orchestrating-ml-at-scale)
4. [Part 4: Production Deployment on AWS](#part-4-production-deployment-on-aws)
5. [Part 5: Controlling Throughput, Scalability & Latency](#part-5-controlling-throughput-scalability--latency)
6. [Part 6: Observability & Monitoring](#part-6-observability--monitoring)
7. [Part 7: Complete Command Reference](#part-7-complete-command-reference)

---

## Part 1: Docker for ML — From Zero to Container

### What Problem Does Docker Solve?

Without Docker:
```
Developer A: "It works on my machine" (Python 3.11, Ubuntu)
Developer B: "It crashes on mine" (Python 3.9, macOS)
Production:  "It fails in deployment" (Python 3.10, Amazon Linux)
```

With Docker:
```
Everyone runs the SAME container → same OS, same Python, same libraries → works everywhere
```

### Core Docker Concepts

```
┌─────────────────────────────────────────────┐
│              Docker Image                   │
│  (Blueprint — like a class in OOP)          │
│                                             │
│  Contains: OS + Python + Libraries + Code   │
│  Immutable: Once built, never changes       │
│  Shareable: Push to registry, pull anywhere │
└─────────────────────────────────────────────┘
                    │
                    │ docker run
                    ▼
┌─────────────────────────────────────────────┐
│              Docker Container               │
│  (Running instance — like an object in OOP) │
│                                             │
│  Has: its own filesystem, network, process  │
│  Ephemeral: dies when stopped               │
│ Isolated: can't see host or other containers│
└─────────────────────────────────────────────┘
```

### Dockerfile — Line by Line Explanation

```dockerfile
# Dockerfile.serve — serves our trained ML model

# 1. BASE IMAGE: Start from official Python 3.11 (Debian slim = small size)
#    Think of this as "which OS and language to install"
#    slim = minimal Debian without extras (gcc, man pages, etc.)
#    Full image: ~900MB, slim: ~150MB
FROM python:3.11-slim

# 2. WORKDIR: Set the working directory inside the container
#    All subsequent commands run from /app
#    Like doing: mkdir /app && cd /app
WORKDIR /app

# 3. COPY requirements first (Docker layer caching optimization)
#    Docker caches each layer. If requirements.txt hasn't changed,
#    Docker skips re-installing packages on rebuild = faster builds
COPY requirements.txt .

# 4. INSTALL: Install Python packages
#    --no-cache-dir: Don't store pip's download cache (saves ~100MB)
RUN pip install --no-cache-dir -r requirements.txt

# 5. COPY source code and model artifacts
COPY src/ src/
COPY configs/ configs/
COPY models/ models/

# 6. EXPOSE: Document which port the app uses (informational only)
#    Does NOT actually open the port — that's done with docker run -p
EXPOSE 8000

# 7. HEALTHCHECK: Docker periodically runs this to check if container is healthy
#    If it fails 3 times in a row, Docker marks container as "unhealthy"
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import httpx; r = httpx.get('http://localhost:8000/health'); assert r.status_code == 200"

# 8. CMD: The command that runs when the container starts
#    Only ONE CMD per Dockerfile (last one wins)
CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Multi-Stage Builds (Advanced — Smaller Images)

```dockerfile
# Stage 1: TRAIN (large image with all ML libraries)
FROM python:3.11-slim AS trainer
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src/ src/
COPY configs/ configs/
RUN mkdir -p models && python -m src.models.train
# At this point, models/ has our trained model

# Stage 2: SERVE (can be smaller — only needs inference libraries)
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src/ src/
COPY configs/ configs/
# Copy ONLY the trained model from stage 1
COPY --from=trainer /app/models/ models/
CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Why multi-stage?**
- Stage 1 has training data, training code, MLflow — all unnecessary for serving
- Stage 2 only has what's needed to serve predictions
- Final image is smaller = faster deploys, less attack surface

### Docker Commands — Every One Explained

```bash
# BUILD an image from a Dockerfile
# -f: which Dockerfile to use
# -t: tag (name:version) for the image
# . : build context (which files Docker can access)
docker build -f docker/Dockerfile.serve -t lr-serve:latest .

# RUN a container from an image
# -p 8000:8000: map host port 8000 → container port 8000
# -d: detached mode (runs in background)
# --name: give the container a human-readable name
docker run -d -p 8000:8000 --name my-api lr-serve:latest

# LIST running containers
docker ps

# LIST all containers (including stopped)
docker ps -a

# VIEW logs from a container
docker logs my-api
docker logs -f my-api  # -f = follow (like tail -f)

# STOP a running container (graceful shutdown)
docker stop my-api

# REMOVE a stopped container
docker rm my-api

# REMOVE an image
docker rmi lr-serve:latest

# EXECUTE a command inside a running container (debugging)
docker exec -it my-api bash
# Now you're inside the container! ls, cat, python, etc.

# PULL an image from a registry
docker pull ghcr.io/sanjeevranjaniitb/lr-mlops-e2e:latest

# PUSH an image to a registry
docker push ghcr.io/sanjeevranjaniitb/lr-mlops-e2e:latest

# VOLUMES: persist data outside the container
# -v host_path:container_path
docker run -v $(pwd)/models:/app/models lr-train:latest
# Now models/ on your host gets the trained model files
```

### Docker Compose — Multi-Container Apps

```yaml
# docker-compose.yaml
services:
  trainer:                          # Service 1: trains the model
    build:
      context: .
      dockerfile: docker/Dockerfile.train
    volumes:
      - ./models:/app/models        # Share models with host

  api:                              # Service 2: serves predictions
    build:
      context: .
      dockerfile: docker/Dockerfile.serve
    ports:
      - "8000:8000"                 # Expose to host
    depends_on:
      trainer:
        condition: service_completed_successfully  # Wait for training

  mlflow:                           # Service 3: experiment tracking
    image: python:3.11-slim
    command: mlflow server --host 0.0.0.0 --port 5000
    ports:
      - "5000:5000"
```

```bash
# Start everything
docker compose up --build

# Start in background
docker compose up -d --build

# Stop everything
docker compose down

# View logs
docker compose logs -f api
```

---

## Part 2: CI/CD — Every Command Explained

### What is CI/CD?

```
CI = Continuous Integration
     "Every code change is automatically tested"
     
CD = Continuous Deployment
     "Every passing change is automatically deployed"

Together: Push code → Tests run → Model trains → Docker builds → Deploys to production
          ALL AUTOMATICALLY. No human intervention needed.
```

### GitHub Actions — How It Works

GitHub Actions runs your code on **free virtual machines** (called "runners") in the cloud.

```
You push code → GitHub detects the push → Spins up a fresh Ubuntu VM →
Runs your workflow steps → Reports pass/fail → VM is destroyed
```

**Free limits:**
- Public repos: UNLIMITED minutes
- Private repos: 2,000 minutes/month

### Workflow 1: CI (ci.yaml) — Line by Line

```yaml
# ─── TRIGGER: When does this workflow run? ───
name: CI - Lint & Test

on:
  push:
    branches: [main, develop]    # Run on push to main or develop
  pull_request:
    branches: [main]             # Run on PRs targeting main

# ─── JOBS: What does it do? ───
jobs:
  lint-and-test:
    runs-on: ubuntu-latest       # Use a free Ubuntu VM (2 CPU, 7GB RAM)

    steps:
      # Step 1: CHECKOUT — Get your code onto the runner
      # Without this, the VM is empty (no code)
      - uses: actions/checkout@v4

      # Step 2: SETUP PYTHON — Install Python on the runner
      # cache: "pip" = cache downloaded packages between runs (faster)
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          cache: "pip"

      # Step 3: INSTALL — Install your project dependencies
      - name: Install dependencies
        run: pip install -r requirements.txt

      # Step 4: LINT — Check code quality (catches bugs before runtime)
      # ruff check: finds unused imports, undefined vars, style issues
      - name: Lint with ruff
        run: ruff check src/ tests/

      # Step 5: FORMAT CHECK — Ensure consistent code style
      # --check: don't fix, just report (fails if formatting is wrong)
      - name: Format check with ruff
        run: ruff format --check src/ tests/

      # Step 6: TEST — Run all tests with coverage
      # -v: verbose (show each test name)
      # --cov=src: measure code coverage for src/ directory
      # --cov-report=xml: output coverage in XML format
      - name: Run tests
        run: pytest tests/ -v --cov=src --cov-report=xml

      # Step 7: UPLOAD — Save coverage report as artifact
      # Artifacts persist after the workflow ends (downloadable)
      - name: Upload coverage
        uses: actions/upload-artifact@v4
        with:
          name: coverage-report
          path: coverage.xml
```

### Workflow 2: Train & Deploy (train-and-deploy.yaml) — Line by Line

```yaml
name: Train & Deploy

# ─── TRIGGERS ───
on:
  push:
    branches: [main]
    paths:                         # ONLY run if these files changed
      - "src/**"                   # Any file in src/
      - "configs/**"               # Any config file
      - "requirements.txt"         # Dependencies changed
  workflow_dispatch:               # Allow manual trigger from GitHub UI
  schedule:
    - cron: "0 6 * * 1"          # Run every Monday at 6 AM UTC
                                   # (automated weekly retraining)

# ─── PERMISSIONS ───
# GITHUB_TOKEN needs these to push images and create releases
permissions:
  contents: write                  # Create releases, push tags
  packages: write                  # Push Docker images to GHCR

# ─── JOB 1: TRAIN ───
jobs:
  train:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          cache: "pip"

      - name: Install dependencies
        run: pip install -r requirements.txt

      # THE ACTUAL TRAINING
      # Runs your full pipeline: load data → scale → fit → evaluate → save
      - name: Run training pipeline
        run: python -m src.models.train

      # SAVE MODEL FILES as workflow artifacts
      # These persist between jobs (train → build-and-push)
      - name: Upload model artifacts
        uses: actions/upload-artifact@v4
        with:
          name: model-artifacts
          path: models/              # Saves model.joblib, scaler.joblib, metadata.json
          retention-days: 30         # Keep for 30 days

      # SMOKE TEST: Quick sanity check that the model works
      - name: Run smoke test on model
        run: |
          python -c "
          import joblib, json, numpy as np
          model = joblib.load('models/model.joblib')
          scaler = joblib.load('models/scaler.joblib')
          with open('models/metadata.json') as f:
              meta = json.load(f)
          X = np.random.randn(1, len(meta['feature_names']))
          X_scaled = scaler.transform(X)
          pred = model.predict(X_scaled)
          assert pred.shape == (1,), f'Expected shape (1,), got {pred.shape}'
          print(f'Smoke test passed. Prediction: {pred[0]:.4f}')
          "

  # ─── JOB 2: BUILD & PUSH DOCKER IMAGE ───
  build-and-push:
    needs: train                   # Only runs AFTER train job succeeds
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'  # Only on main branch

    steps:
      - uses: actions/checkout@v4

      # DOWNLOAD the model files saved by the train job
      - name: Download model artifacts
        uses: actions/download-artifact@v4
        with:
          name: model-artifacts
          path: models/

      # SETUP BUILDX: Enhanced Docker builder (supports caching, multi-platform)
      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3

      # LOGIN to GitHub Container Registry
      # Uses the automatic GITHUB_TOKEN (no secrets needed for public repos)
      - name: Login to GitHub Container Registry
        uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}       # Your GitHub username
          password: ${{ secrets.GITHUB_TOKEN }}  # Auto-provided by GitHub

      # BUILD the Docker image and PUSH to registry
      - name: Build and push Docker image
        uses: docker/build-push-action@v5
        with:
          context: .                           # Build context = repo root
          file: docker/Dockerfile.serve        # Which Dockerfile
          push: true                           # Push to registry (not just build)
          tags: |
            ghcr.io/${{ github.repository }}:latest
            ghcr.io/${{ github.repository }}:${{ github.sha }}
          # ↑ Two tags: "latest" (always current) + commit SHA (immutable)
          cache-from: type=gha                 # Use GitHub Actions cache
          cache-to: type=gha,mode=max          # Save layers to cache

  # ─── JOB 3: CREATE GITHUB RELEASE ───
  create-release:
    needs: [train, build-and-push]
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'

    steps:
      - uses: actions/checkout@v4

      - name: Download model artifacts
        uses: actions/download-artifact@v4
        with:
          name: model-artifacts
          path: models/

      # CREATE a versioned release with model files attached
      # Users can download model.joblib directly from the release page
      - name: Create Release
        uses: softprops/action-gh-release@v2
        with:
          tag_name: model-${{ github.run_number }}
          name: "Model Release #${{ github.run_number }}"
          body: |
            Automated model release from training pipeline.
            Commit: ${{ github.sha }}
          files: |
            models/model.joblib
            models/scaler.joblib
            models/metadata.json
```

### Key CI/CD Concepts

| Concept | Explanation |
|---------|-------------|
| **Workflow** | A YAML file that defines automation (lives in `.github/workflows/`) |
| **Job** | A set of steps that run on the same VM |
| **Step** | A single command or action |
| **Action** | A reusable step (e.g., `actions/checkout@v4`) |
| **Artifact** | Files saved between jobs or after workflow ends |
| **Secret** | Encrypted variable (API keys, passwords) |
| **Runner** | The VM that executes your workflow |
| **Matrix** | Run same job with different configs (e.g., Python 3.10, 3.11, 3.12) |

### Container Registry (GHCR) Explained

```
GitHub Container Registry (ghcr.io) = Docker Hub but free and integrated with GitHub

Image naming: ghcr.io/<owner>/<repo>:<tag>
Example:      ghcr.io/sanjeevranjaniitb/lr-mlops-e2e:latest

Tags:
  :latest     → always points to most recent build
  :<sha>      → immutable, tied to specific commit (for rollbacks)
  :v1.0.0     → semantic version (for releases)
```

---

## Part 3: Kubernetes — Orchestrating ML at Scale

### Why Kubernetes for ML?

Docker runs ONE container. Kubernetes runs THOUSANDS across multiple machines.

```
Docker alone:
  - 1 container crashes → your API is down
  - Traffic spikes → 1 container can't handle it
  - Deploy new model → downtime while restarting

Kubernetes:
  - 1 container crashes → K8s auto-restarts it in seconds
  - Traffic spikes → K8s auto-scales to 10 containers
  - Deploy new model → rolling update, zero downtime
```

### Kubernetes Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     KUBERNETES CLUSTER                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                CONTROL PLANE (Master)                   │    │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐    │    │
│  │  │API Server│ │Scheduler │ │Controller│ │  etcd    │    │    │
│  │  │          │ │          │ │ Manager  │ │(database)│    │    │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘    │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                 │
│  ┌───────────────────┐  ┌───────────────────┐                   │
│  │    WORKER NODE 1  │  │    WORKER NODE 2  │                   │
│  │  ┌─────┐ ┌─────┐  │  │  ┌─────┐ ┌─────┐  │                   │
│  │  │Pod 1│ │Pod 2│  │  │  │Pod 3│ │Pod 4│  │                   │
│  │  └─────┘ └─────┘  │  │  └─────┘ └─────┘  │                   │
│  │  ┌──────────────┐ │  │  ┌──────────────┐ │                   │
│  │  │   kubelet    │ │  │  │   kubelet    │ │                   │
│  │  └──────────────┘ │  │  └──────────────┘ │                   │
│  └───────────────────┘  └───────────────────┘                   │
└─────────────────────────────────────────────────────────────────┘
```

### Every kubectl Command We Ran — Explained

```bash
# ─── NAMESPACE ───
kubectl apply -f k8s/namespace.yaml
# Creates a "namespace" = logical isolation boundary
# Like creating a folder for your project
# Resources in different namespaces can't see each other by default
# Why: keeps your ML stuff separate from other apps in the cluster

# ─── DEPLOYMENT ───
kubectl apply -f k8s/deployment.yaml
# Creates a "Deployment" = tells K8s "run 2 copies of my container"
# K8s will:
#   1. Pull the Docker image
#   2. Start 2 pods (containers)
#   3. Monitor them continuously
#   4. Restart any that crash
#   5. Replace any that fail health checks

# ─── SERVICE ───
kubectl apply -f k8s/service.yaml
# Creates a "Service" = stable network endpoint
# Problem: Pods get random IPs that change when they restart
# Solution: Service gives a fixed IP/DNS that load-balances across pods
# Think of it as a load balancer sitting in front of your pods

# ─── HPA (Horizontal Pod Autoscaler) ───
kubectl apply -f k8s/hpa.yaml
# Creates auto-scaling rules:
#   "If average CPU > 70%, add more pods (up to 4)"
#   "If average CPU < 70%, remove pods (down to 1)"
# This is how you handle traffic spikes automatically

# ─── MONITORING COMMANDS ───

# See all pods and their status
kubectl get pods -n ml-serving
# STATUS meanings:
#   Pending     = waiting for resources (node is full)
#   Running     = container is alive and healthy
#   CrashLoop   = container keeps crashing (check logs!)
#   ImagePull   = can't download the Docker image
#   Completed   = ran and exited (normal for jobs)

# See detailed info about a pod (for debugging)
kubectl describe pod <pod-name> -n ml-serving
# Shows: events, resource usage, restart count, error messages

# View container logs (like docker logs)
kubectl logs <pod-name> -n ml-serving
kubectl logs -f <pod-name> -n ml-serving  # -f = follow (live tail)

# View logs from ALL pods in a deployment
kubectl logs -f deployment/lr-model-api -n ml-serving

# ─── PORT FORWARDING ───
kubectl port-forward svc/lr-model-api 9000:80 -n ml-serving
# Maps localhost:9000 → Service:80 → Pod:8000
# This is how you access the API from your laptop
# Only for development! In production, use Ingress or LoadBalancer

# ─── SCALING ───
kubectl scale deployment/lr-model-api --replicas=5 -n ml-serving
# Manually set to 5 pods (overrides HPA temporarily)

# ─── ROLLING UPDATE (zero-downtime deploy) ───
kubectl set image deployment/lr-model-api \
  api=ghcr.io/sanjeevranjaniitb/lr-mlops-e2e:new-tag -n ml-serving
# K8s will:
#   1. Start new pods with new image
#   2. Wait for them to pass health checks
#   3. Gradually shift traffic to new pods
#   4. Terminate old pods
#   Result: ZERO downtime during deployment

# ─── ROLLBACK (if new version is broken) ───
kubectl rollout undo deployment/lr-model-api -n ml-serving
# Instantly reverts to the previous working version

# ─── CLEANUP ───
kubectl delete namespace ml-serving
# Deletes EVERYTHING in the namespace (pods, services, deployments)
```

### Health Checks — Why They Matter

```yaml
# In deployment.yaml:
livenessProbe:
  httpGet:
    path: /health
    port: 8000
  initialDelaySeconds: 10    # Wait 10s before first check
  periodSeconds: 30          # Check every 30s

readinessProbe:
  httpGet:
    path: /health
    port: 8000
  initialDelaySeconds: 5
  periodSeconds: 10
```

**Liveness Probe:** "Is the container alive?"
- If it fails → K8s kills and restarts the container
- Catches: deadlocks, infinite loops, OOM

**Readiness Probe:** "Is the container ready to serve traffic?"
- If it fails → K8s stops sending traffic to this pod
- Catches: model still loading, dependency unavailable
- Other pods still serve traffic (no downtime!)

### Resource Limits — Preventing Noisy Neighbors

```yaml
resources:
  requests:
    memory: "128Mi"    # Guaranteed minimum (scheduler uses this)
    cpu: "100m"        # 100 millicores = 0.1 CPU
  limits:
    memory: "256Mi"    # Maximum allowed (killed if exceeded = OOMKilled)
    cpu: "500m"        # Throttled if exceeded (not killed)
```

**Why this matters for ML:**
- ML models can be memory-hungry (large numpy arrays)
- Without limits, one pod can starve others
- `requests` = what the scheduler guarantees
- `limits` = hard ceiling (memory = kill, CPU = throttle)

---

## Part 4: Production Deployment on AWS

### AWS Services for ML Deployment

```
┌─────────────────────────────────────────────────────────────────┐
│                    AWS ML Deployment Stack                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────-┐      │ 
│  │  ECR    │    │  EKS    │    │  ALB    │    │CloudWatch│      │
│  │Registry │───▶│(K8s)    │◀───│(LB)     │    │(Monitor) │      │
│  └─────────┘    └─────────┘    └─────────┘    └─────────-┘      │
│                       │                                         │
│                  ┌────▼────┐                                    │
│                  │ Fargate │  (serverless containers)           │
│                  └─────────┘                                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

| AWS Service | What It Does | Equivalent |
|-------------|-------------|------------|
| **ECR** (Elastic Container Registry) | Stores Docker images | Like GHCR but on AWS |
| **EKS** (Elastic Kubernetes Service) | Managed Kubernetes | Like minikube but production-grade |
| **Fargate** | Serverless containers (no servers to manage) | Like Lambda but for containers |
| **ALB** (Application Load Balancer) | Routes traffic to pods | Like K8s Ingress |
| **CloudWatch** | Logs + metrics + alerts | Like Prometheus + Grafana |
| **SageMaker** | Managed ML platform | All-in-one (but expensive) |

### Option 1: EKS (Kubernetes on AWS) — Full Control

```bash
# 1. Install eksctl (EKS cluster management tool)
brew install eksctl

# 2. Create a cluster (takes ~15 minutes)
eksctl create cluster \
  --name ml-production \
  --region us-east-1 \
  --nodegroup-name ml-workers \
  --node-type t3.medium \        # 2 vCPU, 4GB RAM ($0.0416/hr)
  --nodes 2 \                    # Start with 2 nodes
  --nodes-min 1 \                # Scale down to 1
  --nodes-max 5                  # Scale up to 5

# 3. Push image to ECR (AWS's container registry)
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-east-1.amazonaws.com

aws ecr create-repository --repository-name lr-mlops-api

docker tag ghcr.io/sanjeevranjaniitb/lr-mlops-e2e:latest \
  <account-id>.dkr.ecr.us-east-1.amazonaws.com/lr-mlops-api:latest

docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/lr-mlops-api:latest

# 4. Deploy (same K8s manifests, just change the image URL)
kubectl apply -f k8s/

# 5. Expose via ALB (AWS Load Balancer)
# Install AWS Load Balancer Controller addon first, then:
kubectl apply -f k8s/ingress.yaml
# AWS automatically provisions an ALB with a public URL
```

### Option 2: ECS + Fargate — Simpler (No K8s Knowledge Needed)

```bash
# Fargate = serverless containers
# You define: image, CPU, memory, port
# AWS handles: servers, scaling, networking

# 1. Create task definition (like a Deployment in K8s)
aws ecs register-task-definition \
  --family lr-mlops-api \
  --network-mode awsvpc \
  --requires-compatibilities FARGATE \
  --cpu 256 \                    # 0.25 vCPU
  --memory 512 \                 # 512 MB
  --container-definitions '[{
    "name": "api",
    "image": "<account-id>.dkr.ecr.us-east-1.amazonaws.com/lr-mlops-api:latest",
    "portMappings": [{"containerPort": 8000}],
    "healthCheck": {
      "command": ["CMD-SHELL", "curl -f http://localhost:8000/health || exit 1"]
    }
  }]'

# 2. Create service (runs and maintains desired count)
aws ecs create-service \
  --cluster ml-cluster \
  --service-name lr-api \
  --task-definition lr-mlops-api \
  --desired-count 2 \            # Run 2 instances
  --launch-type FARGATE

# 3. Auto-scaling
aws application-autoscaling register-scalable-target \
  --service-namespace ecs \
  --resource-id service/ml-cluster/lr-api \
  --scalable-dimension ecs:service:DesiredCount \
  --min-capacity 1 \
  --max-capacity 10
```

### Cost Comparison (AWS)

| Setup | Monthly Cost (light traffic) | Monthly Cost (heavy traffic) |
|-------|------------------------------|------------------------------|
| EKS + t3.medium (2 nodes) | ~$75 | ~$200 (auto-scaled) |
| Fargate (2 tasks, 0.25 vCPU) | ~$20 | ~$100 (auto-scaled) |
| SageMaker Endpoint (ml.t2.medium) | ~$50 | ~$200 |
| Lambda + API Gateway | ~$0-5 | ~$30 (pay per request) |

**Recommendation for startups:** Start with Fargate. Move to EKS when you need more control.

---

## Part 5: Controlling Throughput, Scalability & Latency

### The Three Pillars of Production ML Performance

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  THROUGHPUT          LATENCY              SCALABILITY           │
│  (requests/sec)     (response time)      (handle growth)        │
│                                                                 │
│  "How many?"        "How fast?"          "How big?"             │
│                                                                 │
│  Target: 1000 RPS   Target: <100ms p99   Target: 10x traffic    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Controlling THROUGHPUT (Requests Per Second)

#### 1. Horizontal Scaling (More Pods)

```yaml
# hpa.yaml — Auto-scale based on requests per second
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
spec:
  minReplicas: 2
  maxReplicas: 20
  metrics:
    # Scale on CPU
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 60    # Scale up at 60% CPU

    # Scale on custom metric (requests per second)
    - type: Pods
      pods:
        metric:
          name: http_requests_per_second
        target:
          type: AverageValue
          averageValue: "100"       # Scale up when >100 RPS per pod
```

#### 2. Worker Processes (More Threads Per Pod)

```dockerfile
# Instead of 1 worker:
CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]

# Use multiple workers (1 worker per CPU core):
CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]

# Or use Gunicorn (production WSGI server) with Uvicorn workers:
CMD ["gunicorn", "src.serving.app:app", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000"]
```

**Rule of thumb:** workers = 2 × CPU cores + 1

#### 3. Request Batching

```python
# Instead of predicting one sample at a time,
# batch multiple requests together:

@app.post("/predict/batch")
async def predict_batch(request: BatchRequest):
    X = np.array(request.features)  # Could be 100 samples
    X_scaled = scaler.transform(X)
    predictions = model.predict(X_scaled)  # One call, 100 predictions
    return {"predictions": predictions.tolist()}
```

### Controlling LATENCY (Response Time)

#### 1. Model Optimization

```python
# BEFORE: sklearn model (good for small models)
# Prediction time: ~1-5ms

# FASTER: Convert to ONNX Runtime
import onnxruntime as ort
from skl2onnx import convert_sklearn

# Convert sklearn model to ONNX
onnx_model = convert_sklearn(model, initial_types=[...])
# Save
with open("model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())

# Load with ONNX Runtime (2-10x faster inference)
session = ort.InferenceSession("model.onnx")
predictions = session.run(None, {"X": X_scaled})[0]
# Prediction time: ~0.1-1ms
```

#### 2. Caching (For Repeated Predictions)

```python
from functools import lru_cache
import hashlib

# Cache predictions for identical inputs
prediction_cache = {}

@app.post("/predict")
async def predict(request: PredictionRequest):
    # Create cache key from input
    cache_key = hashlib.md5(str(request.features).encode()).hexdigest()
    
    if cache_key in prediction_cache:
        return prediction_cache[cache_key]  # Cache hit: ~0.01ms
    
    # Cache miss: compute prediction
    result = model.predict(...)
    prediction_cache[cache_key] = result
    return result
```

#### 3. Async Processing (For Heavy Models)

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

executor = ThreadPoolExecutor(max_workers=4)

@app.post("/predict")
async def predict(request: PredictionRequest):
    # Run CPU-heavy prediction in thread pool
    # This prevents blocking the event loop
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        executor, 
        lambda: model.predict(X_scaled)
    )
    return {"predictions": result.tolist()}
```

#### 4. Connection Pooling & Keep-Alive

```yaml
# In K8s Ingress annotations:
metadata:
  annotations:
    nginx.ingress.kubernetes.io/proxy-connect-timeout: "5"
    nginx.ingress.kubernetes.io/proxy-read-timeout: "60"
    nginx.ingress.kubernetes.io/upstream-keepalive-connections: "100"
```

### Controlling SCALABILITY

#### Cluster Autoscaler (Add More Machines)

```yaml
# When pods can't be scheduled (not enough resources),
# Cluster Autoscaler adds more EC2 instances

# AWS EKS: Install Cluster Autoscaler
# It watches for "Pending" pods and provisions new nodes

# Node group config:
nodeGroups:
  - name: ml-workers
    instanceType: t3.large      # 2 vCPU, 8GB
    minSize: 1                  # Minimum 1 node
    maxSize: 10                 # Maximum 10 nodes
    desiredCapacity: 2          # Start with 2
```

#### Pod Disruption Budget (Safe Scaling)

```yaml
# Ensure at least 1 pod is always running during updates/scaling
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: lr-model-api-pdb
spec:
  minAvailable: 1              # Always keep at least 1 pod running
  selector:
    matchLabels:
      app: lr-model-api
```

### Latency Optimization Checklist

| Technique | Latency Reduction | Effort |
|-----------|-------------------|--------|
| ONNX Runtime | 2-10x faster | Medium |
| Response caching | 100x for cache hits | Low |
| Gunicorn workers | 2-4x throughput | Low |
| Model quantization | 2-4x faster | Medium |
| GPU inference | 10-100x for deep learning | High |
| CDN for static responses | Eliminates network latency | Low |
| Regional deployment | -50-200ms | Low |
| gRPC instead of REST | -5-20ms | Medium |

---

## Part 6: Observability & Monitoring

### The Three Pillars of Observability

```
┌─────────────────────────────────────────────────────────────────┐
│                    OBSERVABILITY                                │
├──────────────────┬──────────────────┬───────────────────────────┤
│     METRICS      │      LOGS        │         TRACES            │
│  (Numbers)       │  (Events)        │  (Request journey)        │
│                  │                  │                           │
│  CPU: 45%        │  ERROR: model    │  Request → API → Model    │
│  Latency: 23ms   │  failed to load  │  → Scaler → Predict       │
│  RPS: 150        │  at 14:32:01     │  Total: 45ms              │
│                  │                  │                           │
│  Tool: Prometheus│  Tool: ELK/Loki  │  Tool: Jaeger/X-Ray       │
└──────────────────┴──────────────────┴───────────────────────────┘
```

### Metrics with Prometheus + Grafana

#### Step 1: Add Prometheus Metrics to Your API

```python
# src/serving/metrics.py
from prometheus_client import Counter, Histogram, Info, generate_latest
from fastapi import Response

# Define metrics
PREDICTION_COUNT = Counter(
    "predictions_total",
    "Total number of predictions made",
    ["model_type", "status"]
)

PREDICTION_LATENCY = Histogram(
    "prediction_duration_seconds",
    "Time spent making predictions",
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
)

MODEL_INFO = Info(
    "model",
    "Information about the loaded model"
)

# In your predict endpoint:
import time

@app.post("/predict")
async def predict(request: PredictionRequest):
    start_time = time.time()
    try:
        result = model.predict(X_scaled)
        PREDICTION_COUNT.labels(model_type="linear_regression", status="success").inc()
        return result
    except Exception as e:
        PREDICTION_COUNT.labels(model_type="linear_regression", status="error").inc()
        raise
    finally:
        PREDICTION_LATENCY.observe(time.time() - start_time)

# Expose metrics endpoint
@app.get("/metrics")
async def metrics():
    return Response(content=generate_latest(), media_type="text/plain")
```

#### Step 2: Deploy Prometheus in K8s

```yaml
# prometheus-config.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: prometheus-config
  namespace: ml-serving
data:
  prometheus.yml: |
    scrape_configs:
      - job_name: 'lr-model-api'
        scrape_interval: 15s
        kubernetes_sd_configs:
          - role: pod
            namespaces:
              names: ['ml-serving']
        relabel_configs:
          - source_labels: [__meta_kubernetes_pod_label_app]
            regex: lr-model-api
            action: keep
```

#### Step 3: Key Metrics to Monitor for ML

| Metric | What It Tells You | Alert Threshold |
|--------|-------------------|-----------------|
| `prediction_duration_seconds` | How fast is inference? | p99 > 500ms |
| `predictions_total` | How many predictions? | Sudden drop = problem |
| `model_prediction_value` | Distribution of outputs | Drift from training |
| `http_requests_total` | Total API traffic | Spike = possible attack |
| `container_memory_usage_bytes` | Memory consumption | > 80% of limit |
| `container_cpu_usage_seconds` | CPU consumption | > 70% sustained |
| `kube_pod_restart_count` | Pod crashes | Any restart = investigate |

### Structured Logging

```python
# BAD: Unstructured logs (hard to search/filter)
print(f"Prediction made: {prediction}")
logger.info(f"User requested prediction for {features}")

# GOOD: Structured JSON logs (searchable, filterable)
import json
import logging

class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_data = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
        }
        if hasattr(record, "extra"):
            log_data.update(record.extra)
        return json.dumps(log_data)

# Usage:
logger.info("Prediction made", extra={
    "prediction": 4.15,
    "features_count": 8,
    "latency_ms": 23.5,
    "model_version": "v1.2.0",
    "request_id": "abc-123"
})

# Output:
# {"timestamp": "2024-01-15 14:32:01", "level": "INFO", 
#  "message": "Prediction made", "prediction": 4.15, 
#  "latency_ms": 23.5, "request_id": "abc-123"}
```

### Model-Specific Monitoring (ML Observability)

#### Data Drift Detection

```python
# Monitor if incoming data distribution changes from training data
import numpy as np
from scipy import stats

class DriftDetector:
    def __init__(self, reference_data: np.ndarray):
        self.reference_mean = reference_data.mean(axis=0)
        self.reference_std = reference_data.std(axis=0)
    
    def check_drift(self, new_data: np.ndarray) -> dict:
        """Compare new data distribution to training data."""
        new_mean = new_data.mean(axis=0)
        
        # Z-score: how many standard deviations away from training mean?
        z_scores = (new_mean - self.reference_mean) / self.reference_std
        
        # If any feature drifted more than 3 std devs → alert
        drifted_features = np.where(np.abs(z_scores) > 3)[0]
        
        return {
            "is_drifted": len(drifted_features) > 0,
            "drifted_features": drifted_features.tolist(),
            "z_scores": z_scores.tolist()
        }
```

#### Prediction Distribution Monitoring

```python
# Track prediction distribution over time
# If predictions suddenly cluster around one value → model might be broken

PREDICTION_VALUE = Histogram(
    "model_prediction_value",
    "Distribution of prediction values",
    buckets=[0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 6, 7, 8, 10]
)

@app.post("/predict")
async def predict(request):
    prediction = model.predict(X_scaled)[0]
    PREDICTION_VALUE.observe(prediction)  # Track distribution
    return {"prediction": prediction}
```

### Alerting Rules

```yaml
# Prometheus alerting rules
groups:
  - name: ml-model-alerts
    rules:
      # Alert if latency is too high
      - alert: HighPredictionLatency
        expr: histogram_quantile(0.99, prediction_duration_seconds_bucket) > 0.5
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "p99 latency > 500ms for 5 minutes"

      # Alert if error rate is too high
      - alert: HighErrorRate
        expr: rate(predictions_total{status="error"}[5m]) / rate(predictions_total[5m]) > 0.05
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "Error rate > 5% for 2 minutes"

      # Alert if no predictions (model might be down)
      - alert: NoPredictions
        expr: rate(predictions_total[10m]) == 0
        for: 10m
        labels:
          severity: critical
        annotations:
          summary: "No predictions in last 10 minutes"

      # Alert on pod restarts
      - alert: PodCrashLooping
        expr: rate(kube_pod_container_status_restarts_total[15m]) > 0
        labels:
          severity: warning
        annotations:
          summary: "Pod is restarting"
```

### Complete Monitoring Stack (Free & Open Source)

```
┌─────────────────────────────────────────────────────────────────┐
│                    MONITORING STACK                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│ Your API ──metrics──▶ Prometheus ──query──▶ Grafana (dashboards)│
│     │                      │                                    │
│     │──logs──▶ Loki ───────┘──▶ Grafana (log explorer)          │
│     │                                                           │
│     │──traces──▶ Jaeger/Tempo ──▶ Grafana (trace viewer)        │
│                                                                 │
│  Alertmanager ◀── Prometheus alerts ──▶ Slack/PagerDuty/Email   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

Install with Helm (K8s package manager):
```bash
# Install Prometheus + Grafana stack
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm install monitoring prometheus-community/kube-prometheus-stack -n monitoring --create-namespace

# Access Grafana dashboard
kubectl port-forward svc/monitoring-grafana 3000:80 -n monitoring
# Open http://localhost:3000 (admin/prom-operator)
```

### AWS-Specific Monitoring

```bash
# CloudWatch Container Insights (EKS)
# Automatically collects: CPU, memory, network, disk for all pods

# Enable:
aws eks create-addon \
  --cluster-name ml-production \
  --addon-name amazon-cloudwatch-observability

# X-Ray for distributed tracing:
# Add to your FastAPI app:
from aws_xray_sdk.core import xray_recorder
from aws_xray_sdk.ext.fastapi.middleware import XRayMiddleware

app = FastAPI()
XRayMiddleware(app, xray_recorder)
```

---

## Part 7: Complete Command Reference

### Docker Commands Cheat Sheet

```bash
# ─── BUILD ───
docker build -t <name>:<tag> .                    # Build image
docker build -f Dockerfile.serve -t api:v1 .      # Specific Dockerfile
docker build --no-cache -t api:v1 .               # Force rebuild all layers

# ─── RUN ───
docker run -p 8000:8000 <image>                   # Run with port mapping
docker run -d --name api <image>                  # Run detached (background)
docker run -v $(pwd)/data:/app/data <image>       # Mount volume
docker run -e API_KEY=secret <image>              # Set environment variable
docker run --rm <image>                           # Auto-remove after exit
docker run --memory=512m --cpus=1 <image>         # Resource limits

# ─── MANAGE ───
docker ps                                          # List running containers
docker ps -a                                       # List all (including stopped)
docker stop <container>                            # Graceful stop
docker rm <container>                              # Remove container
docker rmi <image>                                 # Remove image
docker system prune -a                             # Remove ALL unused data

# ─── DEBUG ───
docker logs <container>                            # View logs
docker logs -f --tail 100 <container>              # Follow last 100 lines
docker exec -it <container> bash                   # Shell into container
docker inspect <container>                         # Full container details
docker stats                                       # Live resource usage

# ─── REGISTRY ───
docker login ghcr.io                               # Login to GHCR
docker pull ghcr.io/user/repo:tag                  # Pull image
docker push ghcr.io/user/repo:tag                  # Push image
docker tag local:v1 ghcr.io/user/repo:v1           # Tag for registry

# ─── COMPOSE ───
docker compose up --build                          # Build and start all
docker compose up -d                               # Start in background
docker compose down                                # Stop and remove
docker compose logs -f <service>                   # Follow service logs
docker compose exec <service> bash                 # Shell into service
```

### Kubernetes Commands Cheat Sheet

```bash
# ─── CLUSTER ───
kubectl cluster-info                               # Cluster details
kubectl get nodes                                  # List nodes
minikube start                                     # Start local cluster
minikube stop                                      # Stop (preserves state)
minikube delete                                    # Delete cluster

# ─── DEPLOY ───
kubectl apply -f <file.yaml>                       # Create/update resource
kubectl apply -f k8s/                              # Apply all files in directory
kubectl delete -f <file.yaml>                      # Delete resource
kubectl delete namespace <ns>                      # Delete everything in namespace

# ─── VIEW ───
kubectl get pods -n <namespace>                    # List pods
kubectl get svc -n <namespace>                     # List services
kubectl get deployments -n <namespace>             # List deployments
kubectl get hpa -n <namespace>                     # List autoscalers
kubectl get all -n <namespace>                     # List everything
kubectl get events -n <namespace> --sort-by='.lastTimestamp'  # Recent events

# ─── DEBUG ───
kubectl describe pod <pod> -n <ns>                 # Detailed pod info
kubectl logs <pod> -n <ns>                         # Pod logs
kubectl logs -f deployment/<name> -n <ns>          # Follow deployment logs
kubectl exec -it <pod> -n <ns> -- bash             # Shell into pod
kubectl top pods -n <ns>                           # CPU/memory usage

# ─── SCALE ───
kubectl scale deployment/<name> --replicas=5 -n <ns>  # Manual scale
kubectl autoscale deployment/<name> --min=2 --max=10 --cpu-percent=70 -n <ns>

# ─── UPDATE ───
kubectl set image deployment/<name> <container>=<new-image> -n <ns>  # Update image
kubectl rollout status deployment/<name> -n <ns>   # Watch rollout
kubectl rollout undo deployment/<name> -n <ns>     # Rollback
kubectl rollout history deployment/<name> -n <ns>  # View history

# ─── NETWORK ───
kubectl port-forward svc/<name> 8000:80 -n <ns>    # Access service locally
kubectl port-forward pod/<name> 8000:8000 -n <ns>  # Access pod directly
```

### GitHub Actions Workflow Syntax Reference

```yaml
# ─── TRIGGERS ───
on:
  push:
    branches: [main]              # On push to main
    paths: ['src/**']             # Only if src/ changed
  pull_request:
    branches: [main]              # On PR to main
  workflow_dispatch:               # Manual trigger button
  schedule:
    - cron: '0 6 * * 1'          # Every Monday 6 AM UTC

# ─── PERMISSIONS ───
permissions:
  contents: write                  # Push tags, create releases
  packages: write                  # Push Docker images

# ─── JOB DEPENDENCIES ───
jobs:
  test:
    runs-on: ubuntu-latest
  deploy:
    needs: test                    # Only runs after test passes
    if: github.ref == 'refs/heads/main'  # Only on main branch

# ─── SECRETS ───
# Set in: repo → Settings → Secrets → Actions
${{ secrets.MY_SECRET }}           # Access a secret
${{ secrets.GITHUB_TOKEN }}        # Auto-provided by GitHub

# ─── VARIABLES ───
${{ github.sha }}                  # Current commit SHA
${{ github.actor }}                # Username who triggered
${{ github.repository }}           # owner/repo
${{ github.ref }}                  # refs/heads/main
${{ github.run_number }}           # Incrementing run number

# ─── ARTIFACTS ───
- uses: actions/upload-artifact@v4
  with:
    name: my-artifact
    path: models/
    retention-days: 30

- uses: actions/download-artifact@v4
  with:
    name: my-artifact
    path: models/

# ─── CACHING ───
- uses: actions/setup-python@v5
  with:
    python-version: '3.11'
    cache: 'pip'                   # Cache pip packages between runs
```

---

## Summary: The Complete MLOps Pipeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  1. DEVELOP                                                             │
│     Write code → Run tests locally → Commit                             │
│                                                                         │
│  2. PUSH                                                                │
│     git push → GitHub receives code                                     │
│                                                                         │
│  3. CI (Automatic)                                                      │
│     Lint (ruff) → Test (pytest) → Coverage check                        │
│                                                                         │
│  4. TRAIN (Automatic)                                                   │
│     Load data → Feature engineering → Fit model → Evaluate → Save       │
│                                                                         │
│  5. BUILD (Automatic)                                                   │
│     Docker build → Push to GHCR/ECR                                     │
│                                                                         │
│  6. DEPLOY (Automatic)                                                  │
│     kubectl apply → Rolling update → Health checks pass                 │
│                                                                         │
│  7. SERVE                                                               │
│     API receives requests → Scale up/down → Return predictions          │
│                                                                         │
│  8. MONITOR                                                             │
│     Prometheus metrics → Grafana dashboards → Alerts → Slack            │
│                                                                         │
│  9. RETRAIN (Weekly/On drift)                                           │
│     Detect drift → Trigger training → New model → Deploy → Validate     │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Golden Rules for Production ML

1. **Never train in production** — train in CI/CD, deploy artifacts
2. **Always version your models** — tag with commit SHA, never just "latest"
3. **Health checks are mandatory** — K8s needs to know if your model is alive
4. **Set resource limits** — one bad model shouldn't kill the cluster
5. **Monitor predictions, not just infrastructure** — data drift kills silently
6. **Rollback fast** — `kubectl rollout undo` is your best friend
7. **Test before deploy** — smoke tests catch 90% of issues
8. **Log structured JSON** — you'll thank yourself when debugging at 3 AM
9. **Cache aggressively** — same input = same output, don't recompute
10. **Start simple, scale later** — Fargate before EKS, 1 pod before 20

---

*Last updated: May 2026*
*Project: https://github.com/sanjeevranjaniitb/lr-mlops-e2e*
