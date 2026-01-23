# 🧠 Deepfake Detection on Face-Swap Based Videos

A complete end-to-end AI/ML project to detect face-swapped deepfake videos using XceptionNet. Built with a scalable and production-ready architecture including a modern frontend, robust backend, containerized deployment, and observability stack.

---

## 🔍 Overview

Deepfakes pose a significant threat to digital authenticity. This project tackles the challenge by building a high-accuracy classifier to detect face-swapped deepfake videos using a convolutional neural network.

* **Model:** XceptionNet (CNN)
* **Dataset:** FaceForensics++
* **Frontend:** Vite + React
* **Backend:** FastAPI
* **Infrastructure:** Docker, Docker Compose, NGINX
* **Monitoring:** Prometheus + Grafana

---

## 📁 Folder Structure

```
.
├── project/                     # FastAPI server
├── vision-truth-finder/         # React (Vite) client
├── test/                        # Test image: REAL and video:Fake
├── project/model/               # XceptionNet model
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

* Exposes a `/predict` endpoint for model inference
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

* Frontend: [http://localhost:8080](http://localhost:3000)
* Backend: [http://localhost:8000/docs](http://localhost:8000/docs)

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


## 📝 License

This project is for educational and research purposes only.
