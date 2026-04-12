import base64
import datetime
import io
import ipaddress
import json
import logging
import os
import shutil
import socket
import tempfile
import time
import traceback
import urllib.parse
import uuid
from pathlib import Path
from typing import Any, List, Literal, Optional
from dataclasses import dataclass
import aiohttp

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from PIL import Image
from tensorflow.keras.applications import xception

# ----------------------------
# Logging (full traceback)
# ----------------------------
LOG_DIR = Path(os.getenv("LOG_DIR", "./logs"))
LOG_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = LOG_DIR / "app.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
    ],
)
logger = logging.getLogger("deepfake-api-xception")


# ----------------------------
# Upload / download limits
# ----------------------------
ALLOWED_UPLOAD_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}
MAX_DOWNLOAD_BYTES = 10 * 1024 * 1024  # 10 MB cap on URL-fetched images


def _check_ssrf(url: str) -> None:
    """Raise HTTPException if the URL resolves to a private/internal address."""
    parsed = urllib.parse.urlparse(url)
    hostname = parsed.hostname
    if not hostname:
        raise HTTPException(status_code=400, detail="Invalid image URL")
    try:
        results = socket.getaddrinfo(hostname, None)
    except socket.gaierror:
        raise HTTPException(status_code=400, detail="Could not resolve image URL host")
    for _, _, _, _, sockaddr in results:
        ip = ipaddress.ip_address(sockaddr[0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_multicast or ip.is_reserved:
            raise HTTPException(status_code=400, detail="Image URL resolves to a disallowed address")


# ----------------------------
# Local persistence (JSONL)
# ----------------------------
DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)

PREDICTIONS_FILE = DATA_DIR / "predictions.jsonl"
FEEDBACK_FILE = DATA_DIR / "feedback.jsonl"


def _append_jsonl(path: Path, obj: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _find_prediction_by_id(prediction_id: str) -> Optional[dict]:
    if not PREDICTIONS_FILE.exists():
        return None
    with PREDICTIONS_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if obj.get("prediction_id") == prediction_id:
                return obj
    return None


# ----------------------------
# Model loading
# ----------------------------
TARGET_IMAGE_SIZE = (224, 224)

# Path to models.json (defaults to project root)
MODELS_CONFIG_PATH = os.getenv("MODELS_CONFIG_PATH", "./models.json")

# Folder that contains .keras models
MODELS_DIR = os.getenv("MODELS_DIR", "./models/")

@dataclass(frozen=True)
class ModelSpec:
    name: str
    file: str
    description: str
    domain: str
    output: str
    threshold: float


def _load_models_config(path: str) -> list[ModelSpec]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"models.json not found at {path}")

    with open(path, "r", encoding="utf-8") as f:
        data: dict[str, Any] = json.load(f)

    raw_models = data.get("models")
    if not isinstance(raw_models, list) or not raw_models:
        raise ValueError("models.json must contain a non-empty 'models' array")

    specs: list[ModelSpec] = []
    for i, m in enumerate(raw_models):
        if not isinstance(m, dict):
            raise ValueError(f"models.json models[{i}] must be an object")

        # --- string fields ---
        for key in ("name", "file", "description", "domain", "output"):
            if key not in m or not isinstance(m[key], str) or not m[key].strip():
                raise ValueError(f"models.json models[{i}] missing/invalid '{key}'")

        output = m["output"].strip()
        if output not in ("prob_real", "prob_fake"):
            raise ValueError(
                f"models.json models[{i}] invalid output '{output}' (must be prob_real or prob_fake)"
            )

        # --- numeric threshold ---
        if "threshold" not in m:
            raise ValueError(f"models.json models[{i}] missing 'threshold'")

        threshold = m["threshold"]
        if not isinstance(threshold, (int, float)) or not (0.0 <= float(threshold) <= 1.0):
            raise ValueError(
                f"models.json models[{i}] invalid threshold '{threshold}' (must be number in [0,1])"
            )

        specs.append(
            ModelSpec(
                name=m["name"].strip(),
                file=m["file"].strip(),
                description=m["description"].strip(),
                domain=m["domain"].strip(),
                output=output,
                threshold=float(threshold),
            )
        )

    return specs

def load_models_from_config(models_dir: str, config_path: str) -> tuple[dict[str, tf.keras.Model], dict[str, ModelSpec]]: 
    """ Loads only the models defined in models.json. 
    Returns: 
    - loaded_models: dict[name -> keras_model] 
    - loaded_specs: dict[name -> ModelSpec] (only for successfully loaded models) 
    """ 
    logger.info("Loading model specs from: %s", config_path) 
    specs = _load_models_config(config_path)
    
    if not os.path.isdir(models_dir): 
        raise FileNotFoundError(f"Models folder not found at {models_dir}")
        
    loaded_models: dict[str, tf.keras.Model] = {} 
    loaded_specs: dict[str, ModelSpec] = {}
    
    for spec in specs:
        full_path = os.path.join(models_dir, spec.file)
        
        if not os.path.exists(full_path):
            logger.warning("Model file missing for '%s': %s (skipping)", spec.name, full_path)
            continue
        
        try:
            logger.info("Loading model '%s' from '%s'", spec.name, spec.file)
            model = tf.keras.models.load_model(full_path, compile=False)
            loaded_models[spec.name] = model
            loaded_specs[spec.name] = spec
        except Exception as e: logger.exception("Failed to load model '%s' (%s). Skipping. Error: %s", spec.name, spec.file, e)
        
    if not loaded_models:
        raise FileNotFoundError(
            f"No models were loaded. Check {config_path} and ensure model files exist in {models_dir}.")
    
    logger.info("Loaded %d models.", len(loaded_models)) 
    return loaded_models, loaded_specs

# ----------------------------
# Model inference helpers
# ----------------------------
def preprocess_image_bytes(img_bytes: bytes, target_size=TARGET_IMAGE_SIZE) -> tuple[np.ndarray, Image.Image]:
    """
    Mirror notebook preprocessing:
      - RGB
      - resize to (224,224)
      - np array
      - batch dim
      - xception.preprocess_input
    Returns (preprocessed_batch, original_pil)
    """
    pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    resized = pil_img.resize(target_size)
    arr = np.array(resized)
    batch = np.expand_dims(arr, axis=0)  # (1, H, W, C)
    preprocessed = xception.preprocess_input(batch)
    return preprocessed, pil_img


def run_detection(models: dict[str, tf.keras.Model], preprocessed_batch: np.ndarray) -> list[dict]:
    preds: list[dict] = []
    for name, model in models.items():
        raw = float(model.predict(preprocessed_batch, verbose=0)[0][0])
        spec = MODEL_SPECS.get(name)

        if spec is None:
            raise RuntimeError(f"Missing ModelSpec for loaded model '{name}'")

        output = spec.output
        threshold = float(spec.threshold)

        if output == "prob_real":
            prob_real = raw
            prob_fake = 1.0 - raw
        else:
            prob_fake = raw
            prob_real = 1.0 - raw

        preds.append({
            "name": name,
            "real_score": float(prob_real),
            "fake_score": float(prob_fake),
            "threshold": float(threshold),
            "domain": spec.domain,
            "raw": raw,
            "output": output,
        })
    return preds


def get_final_verdict(predictions: list[dict]) -> dict:
    """
    Notebook rule:
      - REAL only if all models real_score >= REAL_THRESHOLD
      - else FAKE, triggered by most confident fake detector (1-real_score max)
    """
    if not predictions:
        return {
            "verdict": "UNKNOWN",
            "trigger_model": None,
            "fake_confidence": None,
            "details": "No predictions were made."
        }

    is_unanimously_real = True
    top_fake_detector = None
    max_fake_confidence = -1.0

    for pred in predictions:
        rs = pred["real_score"]
        if rs < REAL_THRESHOLD:
            is_unanimously_real = False
            fake_conf = 1.0 - rs
            if fake_conf > max_fake_confidence:
                max_fake_confidence = fake_conf
                top_fake_detector = pred["name"]

    if is_unanimously_real:
        return {
            "verdict": "REAL",
            "trigger_model": None,
            "fake_confidence": 0.0,
            "details": "All models agreed the image is authentic."
        }

    return {
        "verdict": "FAKE",
        "trigger_model": top_fake_detector,
        "fake_confidence": max_fake_confidence,
        "details": f"Detection triggered by '{top_fake_detector}' with {max_fake_confidence:.1%} confidence."
    }

def ensemble_score(predictions: list[dict]) -> float:
    """
    Handy scalar score for your API:
      - strongest fake evidence across models = 1 - min(real_score)
      - range [0..1], higher => more fake
    """
    if not predictions:
        return 0.0
    min_real = min(p["real_score"] for p in predictions)
    return float(1.0 - min_real)

def domain_breakdown(predictions: list[dict], specs: dict[str, ModelSpec]) -> dict[str, dict]:
    """
    Group predictions by domain and compute:
      - domain_threshold: max(threshold_of_models_in_domain)  (strictest)
      - min_prob_real: min(real_score_in_domain)
      - fake_score: 1 - min_prob_real
      - trigger_model: model with min_prob_real
      - crosses_threshold: fake_score >= domain_threshold
    """
    domains: dict[str, dict] = {}

    for p in predictions:
        name = p["name"]
        real_score = float(p["real_score"])

        spec = specs.get(name)
        if spec is None:
            domain = "unknown"
            model_threshold = float(p.get("threshold", 0.5))
        else:
            domain = spec.domain
            model_threshold = float(spec.threshold)

        if domain not in domains:
            domains[domain] = {
                "domain": domain,
                "domain_threshold": model_threshold,   # will be max() as we add models
                "min_prob_real": 1.0,
                "fake_score": 0.0,
                "trigger_model": None,
                "models": [],
                "crosses_threshold": False,
            }
        else:
            # strictest threshold wins (higher fake_score needed to call FAKE)
            domains[domain]["domain_threshold"] = max(domains[domain]["domain_threshold"], model_threshold)

        domains[domain]["models"].append({
            "name": name,
            "real_score": real_score,
            "threshold": model_threshold
        })

        if real_score < domains[domain]["min_prob_real"]:
            domains[domain]["min_prob_real"] = real_score
            domains[domain]["trigger_model"] = name

    for d in domains.values():
        d["fake_score"] = float(1.0 - d["min_prob_real"])
        d["crosses_threshold"] = bool(d["fake_score"] >= float(d["domain_threshold"]))

    return domains


def get_final_verdict_domain_aware(predictions: list[dict]) -> dict:
    """
    Domain-aware verdict with per-domain thresholds:
      - For each domain:
          fake_score = 1 - min(prob_real_in_domain)
          domain_threshold = max(model.threshold for models in domain)
          crosses_threshold = fake_score >= domain_threshold
      - Pick the domain with the strongest fake_score (for reporting).
      - Verdict:
          FAKE if ANY domain crosses its threshold, else REAL.
      - overall_fake_score is still max(domain fake_score) for UI/score display.
      - overall_threshold is the threshold of the strongest domain (useful for UI).
    """
    if not predictions:
        return {
            "verdict": "UNKNOWN",
            "trigger_model": None,
            "fake_confidence": None,
            "details": "No predictions were made.",
            "domains": {},
            "overall_fake_score": 0.0,
            "overall_threshold": None,
        }

    domains = domain_breakdown(predictions, MODEL_SPECS)

    # strongest evidence domain for score/reporting
    best_domain = max(domains.values(), key=lambda d: d["fake_score"])
    overall_fake_score = float(best_domain["fake_score"])
    overall_threshold = float(best_domain["domain_threshold"])

    # decision: any domain crossing its own threshold triggers FAKE
    any_trigger = any(d["crosses_threshold"] for d in domains.values())
    verdict = "FAKE" if any_trigger else "REAL"

    trigger_model = best_domain.get("trigger_model")
    if verdict == "FAKE":
        # optional: for details, pick the *triggering* domain with highest fake_score
        triggering_domains = [d for d in domains.values() if d["crosses_threshold"]]
        trigger_domain = max(triggering_domains, key=lambda d: d["fake_score"])
        trigger_model = trigger_domain.get("trigger_model")
        details = (
            f"Detection triggered by '{trigger_model}' in domain '{trigger_domain['domain']}' "
            f"with {float(trigger_domain['fake_score']):.1%} fake confidence "
            f"(domain threshold {float(trigger_domain['domain_threshold']):.0%})."
        )
        fake_confidence = float(trigger_domain["fake_score"])
    else:
        details = (
            f"No domain crossed its threshold. Strongest domain evidence was '{best_domain['domain']}' "
            f"at {overall_fake_score:.1%} (threshold {overall_threshold:.0%})."
        )
        fake_confidence = overall_fake_score

    return {
        "verdict": verdict,
        "trigger_model": trigger_model,
        "fake_confidence": fake_confidence,
        "details": details,
        "domains": domains,
        "overall_fake_score": overall_fake_score,
        "overall_threshold": overall_threshold,
    }


# ----------------------------
# Load all models from MODELS_DIR based on models.json
# ----------------------------
MODELS: dict[str, tf.keras.Model] = {}
MODEL_SPECS: dict[str, ModelSpec] = {}

try:
    MODELS, MODEL_SPECS = load_models_from_config(MODELS_DIR, MODELS_CONFIG_PATH)
except Exception as e:
    logger.error("Failed to load models: %s", e)
    MODELS, MODEL_SPECS = {}, {}


# ----------------------------
# FastAPI app + instrumentation
# ----------------------------
app = FastAPI()

#instrumentator = Instrumentator()
#instrumentator.instrument(app).expose(app)

limiter = Limiter(key_func=get_remote_address, default_limits=["10/minute"])
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["POST", "GET"],
    allow_headers=["Content-Type"],
)

# Limit request body to 10MB
@app.middleware("http")
async def limit_request_size(request: Request, call_next):
    body = await request.body()
    if len(body) > 1024 * 1024 * 10:
        raise HTTPException(status_code=413, detail="Payload too large")
    return await call_next(request)


#REQUEST_COUNT = Counter("model_requests_total", "Total number of model requests", ["model_name", "status"])
#REQUEST_LATENCY = Histogram("model_request_latency_seconds", "Latency per request", ["model_name"])


# @app.middleware("http")
# async def metrics_middleware(request: Request, call_next):
#     model_name = "xception_ensemble"
#     start_time = time.time()

#     try:
#         response = await call_next(request)
#         status = str(response.status_code)
#     except Exception:
#         status = "500"
#         raise
#     finally:
#         duration = time.time() - start_time
#         REQUEST_LATENCY.labels(model_name).observe(duration)
#         REQUEST_COUNT.labels(model_name, status).inc()

#     return response


# ----------------------------
# Schemas
# ----------------------------
class FeedbackRequest(BaseModel):
    prediction_id: str
    user_actual_result: Literal["REAL", "FAKE"]
    feedback_text: Optional[str] = None
    score: Optional[int] = None  # 1–5

class PredictRequest(BaseModel):
    base64_image: Optional[str] = None
    image_url: Optional[HttpUrl] = None
    video_url: Optional[HttpUrl] = None

class ModelInfoOut(BaseModel):
    name: str
    file: str
    description: str
    domain: str
    output: str
    threshold: float
    loaded: bool

class ModelsOut(BaseModel):
    models_dir: str
    config_path: str
    models_loaded: int
    models: List[ModelInfoOut]


# ----------------------------
# Routes
# ----------------------------
@app.get("/api/health")
def health():
    ok = bool(MODELS)
    return {"status": "ok" if ok else "degraded", "models_loaded": len(MODELS), "models_dir": MODELS_DIR}


@app.get("/api/models", response_model=ModelsOut)
def get_models():
    # Return only loaded models
    models = [
        ModelInfoOut(
        name=spec.name,
        file=spec.file,
        description=spec.description,
        domain=spec.domain,
        output=spec.output,
        threshold=spec.threshold,
        loaded=True,
    )
        for spec in (MODEL_SPECS.get(name) for name in sorted(MODELS.keys()))
        if spec is not None
    ]

    return ModelsOut(
        models_dir=MODELS_DIR,
        config_path=MODELS_CONFIG_PATH,
        models_loaded=len(MODELS),
        models=models,
    )


@app.post("/api/predict")
@limiter.limit("5/second")
async def predict(
    request: Request,
    file: Optional[UploadFile] = File(None),
    req: Optional[PredictRequest] = None,
):
    temp_path: Optional[str] = None

    try:
        if not MODELS:
            raise HTTPException(status_code=500, detail="Models not loaded. Check MODELS_DIR and container files.")

        # ----------------------------
        # Build temp input file
        # ----------------------------
        if file:
            raw_ext = (file.filename or "").rsplit(".", 1)[-1].lower()
            if raw_ext not in ALLOWED_UPLOAD_EXTENSIONS:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported file type '.{raw_ext}'. Allowed: {', '.join(sorted(ALLOWED_UPLOAD_EXTENSIONS))}",
                )
            fd, temp_path = tempfile.mkstemp(suffix=f".{raw_ext}")
            with os.fdopen(fd, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

        elif req:
            if req.image_url:
                url_str = str(req.image_url)
                _check_ssrf(url_str)
                async with aiohttp.ClientSession() as session:
                    async with session.get(url_str, allow_redirects=False) as resp:
                        if resp.status != 200:
                            raise HTTPException(status_code=400, detail="Failed to fetch image URL")
                        img_bytes = await resp.content.read(MAX_DOWNLOAD_BYTES + 1)
                        if len(img_bytes) > MAX_DOWNLOAD_BYTES:
                            raise HTTPException(status_code=400, detail="Remote image exceeds 10 MB size limit")
                        fd, temp_path = tempfile.mkstemp(suffix=".jpg")
                        with os.fdopen(fd, "wb") as f:
                            f.write(img_bytes)

            elif req.base64_image:
                if len(req.base64_image) > (MAX_DOWNLOAD_BYTES * 4 // 3) + 64:
                    raise HTTPException(status_code=400, detail="base64_image exceeds size limit")
                img_bytes = base64.b64decode(req.base64_image)
                fd, temp_path = tempfile.mkstemp(suffix=".jpg")
                with os.fdopen(fd, "wb") as f:
                    f.write(img_bytes)

            elif req.video_url:
                raise HTTPException(status_code=400, detail="video_url not supported by main-xception.py (image-only).")

        if not temp_path:
            raise HTTPException(status_code=400, detail="No valid input provided.")

        if temp_path.lower().endswith(".mp4"):
            raise HTTPException(status_code=400, detail="Video files not supported by main-xception.py (image-only).")

        # ----------------------------
        # Inference (Notebook-aligned)
        # ----------------------------
        with open(temp_path, "rb") as f:
            img_bytes = f.read()

        preprocessed, _orig = preprocess_image_bytes(img_bytes, TARGET_IMAGE_SIZE)
        predictions = run_detection(MODELS, preprocessed)
        verdict = get_final_verdict_domain_aware(predictions)

        # overall score is domain-aware: strongest domain fake evidence wins
        score = float(verdict["overall_fake_score"])
        label = verdict["verdict"]

        prediction_id = str(uuid.uuid4())

        prediction_data = {
            "result": label,
            "score": score,
            "threshold": float(verdict["overall_threshold"]) if verdict.get("overall_threshold") is not None else None,
            "prediction_id": prediction_id,
            "input_type": "image",

            "trigger_model": verdict["trigger_model"],
            "fake_confidence": verdict["fake_confidence"],
            "details": verdict["details"],

            "per_model": predictions,
            "domains": verdict["domains"],
        }

        _append_jsonl(PREDICTIONS_FILE, prediction_data)

        return prediction_data

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Unhandled error in /predict: %s\n%s", e, traceback.format_exc())
        raise HTTPException(status_code=500, detail="Internal server error")
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass


@app.post("/api/feedback")
async def submit_feedback(feedback: FeedbackRequest):
    """
    Fixed to match your stored keys:
      - prediction records store: result, score, threshold, prediction_id
    """
    try:
        prediction_data = _find_prediction_by_id(feedback.prediction_id)
        if not prediction_data:
            raise HTTPException(status_code=404, detail="Prediction not found")

        model_result = str(prediction_data.get("result", "")).upper()
        user_result = str(feedback.user_actual_result).upper()

        is_correct = model_result == user_result

        feedback_data = {
            "id": str(uuid.uuid4()),
            "prediction_id": feedback.prediction_id,
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "model_result": model_result,
            "confidence_score": prediction_data.get("score"),
            "user_actual_result": user_result,
            "is_correct": is_correct,
            "user_score": feedback.score,
            "feedback_text": feedback.feedback_text,
            "input_type": prediction_data.get("input_type"),
        }

        _append_jsonl(FEEDBACK_FILE, feedback_data)

        return {
            "status": "success",
            "message": "Feedback submitted successfully.",
            "feedback_id": feedback_data["id"],
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Unhandled error in /feedback: %s\n%s", e, traceback.format_exc())
        raise HTTPException(status_code=500, detail="Internal server error")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

