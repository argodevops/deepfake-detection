# 🧠 Deepfake Detection on Face-Swap Based Videos

An end-to-end AI/ML project to detect face-swapped deepfake videos using XceptionNet. Built with a scalable and production-ready architecture including a modern frontend, robust backend, containerized deployment, and observability stack.

> ⚠️ **Work in Progress — Experimental**
> This project is actively evolving and should be considered experimental. The models, API, and frontend are under active development. Results may be inconsistent across image types, edge cases are not fully handled, and breaking changes may occur without notice. Do not rely on this for production use cases or high-stakes decisions.

---

## 🔍 Overview

Deepfakes pose a significant threat to digital authenticity. This project tackles the challenge by building a high-accuracy classifier to detect face-swapped deepfake videos using a convolutional neural network.

* **Model:** XceptionNet (CNN)
* **Dataset:** FaceForensics++
* **Frontend:** Vite + React
* **Backend:** FastAPI
* **Infrastructure:** Docker, Docker Compose, NGINX
* **Monitoring:** Prometheus

---

## 📁 Folder Structure

```
.
├── project/                     # FastAPI server
├── vision-truth-finder/         # React (Vite) client
├── testdata/                    # Test images: real/ and fake/
├── scripts/                     # Utility scripts (batch predict, benchmarking)
├── project/models/              # XceptionNet model weights + models.json config
├── project/docker-compose.yml   # Docker orchestration
├── project/Dockerfile           # Backend Dockerfile
├── project/nginx/               # NGINX reverse proxy config
├── project/prometheus/          # Prometheus config
├── project/.env                 # Environment variables
├── README.md
├── .gitignore
├── LICENSE
└── run.sh 
```

---

## 🛠️ Features

### 🔬 Deep Learning Model

* Trained a deepfake detection classifier using **XceptionNet**
* Preprocessed **\~42,000 frames** from the **FaceForensics++** dataset
* Achieved **91.76% accuracy** on test set
* Trained on **2×NVIDIA T4 GPUs** with training time under **3 hours**

### ⚙️ Backend (FastAPI)

* Exposes a `/api/predict` endpoint for model inference
* Handles video frame input and passes through preprocessing pipeline
* Uses `python-dotenv` to load environment variables securely
* Structured logging for debugging and traceability

### 🖼️ Frontend (React + Vite)

* Built a lightweight UI for uploading video frames or images
* Shows real-time prediction results and model confidence
* Connected directly to FastAPI via REST

### 🐳 Dockerized Architecture

* Backend and frontend containerized with Docker
* All services orchestrated using Docker Compose
* `.env` support for flexible configuration

### 🌐 NGINX Reverse Proxy

* Serves as the single entrypoint to all services
* Handles CORS, header injection, and internal routing
* Access controlled to internal Docker network

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/argodevops/deepfake-detection
```

### 2. Build and Run All Services

```bash
./run.sh start
```

### 3. Access the Services

* App (via NGINX): [http://localhost:80](http://localhost:80)
* Backend API docs: [http://localhost:8000/docs](http://localhost:8000/docs)

### 4. Stop all Services

```bash
./run.sh stop
```

---

## 📦 Dataset Info

* Used **FaceForensics++** (face-swap subset)
* Preprocessed video files into image frames
* Split into train/test (balanced real/fake)

---

## 📉 Model Performance

| Metric        | Value     |
| ------------- | --------- |
| Accuracy      | 91.76%    |
| Training Time | \~2 hours |
| Hardware      | 2×T4 GPUs |

The model can be downloaded from [Kaggle](https://www.kaggle.com/models/armanchaudhary/xception5o) 

---


## 🔭 Future Work & Improvements

### Model & Detection Quality

* **Video-native inference** — currently the system processes single frames; adding temporal analysis across frame sequences would catch motion-based deepfake artefacts that per-frame models miss
* **Face detection integration** — automatically crop and align faces using MTCNN before passing to the classifier, rather than requiring a well-framed input image
* **Broader manipulation coverage** — the current models target FaceSwap, FaceShifter, Face2Face, and Deepfakes; adding coverage for newer techniques (e.g. diffusion-based face synthesis, voice-driven lip sync) would improve robustness
* **Confidence calibration** — the raw model scores are not well-calibrated probabilities; applying temperature scaling or isotonic regression post-training would make the reported confidence more meaningful

### Infrastructure & Reliability

* **Replace JSONL persistence** — predictions and feedback are currently written to flat files; a lightweight database (SQLite or PostgreSQL) would support querying, pagination, and concurrent writes
* **Authentication** — there is no auth on any endpoint; adding API key or OAuth2 support is needed before any public deployment
* **CORS hardening** — `allow_origins=["*"]` should be restricted to known origins
* **Async model inference** — inference currently blocks the request; offloading to a task queue (e.g. Celery + Redis) would improve throughput and allow returning a job ID for polling
* **HTTPS** — no TLS is configured in NGINX; a certificate (e.g. via Let's Encrypt) is required for production

### Observability

* **Grafana dashboard** — Prometheus scraping is configured but no dashboard exists; building one for request latency, model trigger rates, and FAKE/REAL ratios would make the system easier to monitor
* **Structured prediction logging** — richer logging (input hash, preprocessing time, per-model latency) would help diagnose accuracy issues in production

### Developer Experience

* **Automated tests** — there are no automated tests; adding pytest fixtures that run inference against the `testdata/` images and assert expected verdicts would catch regressions
* **CI pipeline** — no CI is configured; adding GitHub Actions for linting, type-checking, and the test suite would gate merges on quality

---

## 📝 License

This project is for educational and research purposes only.
