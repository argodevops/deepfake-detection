import base64
import datetime
import json
import logging
import os
import shutil
import time
import traceback
import uuid
from pathlib import Path
from typing import Optional

import aiohttp
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from inference import predict_fake_real, predict_from_image

from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Histogram


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
logger = logging.getLogger("deepfake-api")


# ----------------------------
# Local persistence (JSONL)
# ----------------------------
DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)

PREDICTIONS_FILE = DATA_DIR / "predictions.jsonl"
FEEDBACK_FILE = DATA_DIR / "feedback.jsonl"


def _append_jsonl(path: Path, obj: dict) -> None:
    """Append a dict as one JSON line."""
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _find_prediction_by_id(prediction_id: str) -> Optional[dict]:
    """Linear scan in JSONL to find a prediction by id (fine for small files)."""
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
            if obj.get("id") == prediction_id:
                return obj
    return None


# ----------------------------
# FastAPI app + instrumentation
# ----------------------------
app = FastAPI()

instrumentator = Instrumentator()
instrumentator.instrument(app).expose(app)

limiter = Limiter(key_func=get_remote_address, default_limits=["10/minute"])
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # dev friendly; lock down in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Limit request body to 10MB
@app.middleware("http")
async def limit_request_size(request: Request, call_next):
    body = await request.body()
    if len(body) > 1024 * 1024 * 10:
        raise HTTPException(status_code=413, detail="Payload too large")
    return await call_next(request)


REQUEST_COUNT = Counter("model_requests_total", "Total number of model requests", ["model_name", "status"])
REQUEST_LATENCY = Histogram("model_request_latency_seconds", "Latency per request", ["model_name"])


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    model_name = "deepfake_detector"
    start_time = time.time()

    try:
        response = await call_next(request)
        status = str(response.status_code)
    except Exception:
        status = "500"
        raise
    finally:
        duration = time.time() - start_time
        REQUEST_LATENCY.labels(model_name).observe(duration)
        REQUEST_COUNT.labels(model_name, status).inc()

    return response


# ----------------------------
# Schemas
# ----------------------------
class FeedbackRequest(BaseModel):
    prediction_id: str
    user_actual_result: str  # "REAL" or "FAKE"
    feedback_text: Optional[str] = None
    score: Optional[int] = None  # 1–5


class PredictRequest(BaseModel):
    base64_image: Optional[str] = None
    image_url: Optional[HttpUrl] = None
    video_url: Optional[HttpUrl] = None


# ----------------------------
# Routes
# ----------------------------
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
@limiter.limit("5/second")
async def predict(
    request: Request,
    file: Optional[UploadFile] = File(None),
    req: Optional[PredictRequest] = None,
):
    temp_path: Optional[str] = None

    try:
        # ----------------------------
        # Build temp input file
        # ----------------------------
        if file:
            ext = (file.filename or "").split(".")[-1].lower() or "bin"
            temp_path = f"temp_{uuid.uuid4()}.{ext}"
            with open(temp_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

        elif req:
            if req.image_url:
                async with aiohttp.ClientSession() as session:
                    async with session.get(str(req.image_url)) as resp:
                        if resp.status != 200:
                            raise HTTPException(status_code=400, detail="Failed to fetch image URL")
                        img_bytes = await resp.read()
                        temp_path = f"temp_{uuid.uuid4()}.jpg"
                        with open(temp_path, "wb") as f:
                            f.write(img_bytes)

            elif req.video_url:
                async with aiohttp.ClientSession() as session:
                    async with session.get(str(req.video_url)) as resp:
                        if resp.status != 200:
                            raise HTTPException(status_code=400, detail="Failed to fetch video URL")
                        video_bytes = await resp.read()
                        temp_path = f"temp_{uuid.uuid4()}.mp4"
                        with open(temp_path, "wb") as f:
                            f.write(video_bytes)

            elif req.base64_image:
                img_bytes = base64.b64decode(req.base64_image)
                temp_path = f"temp_{uuid.uuid4()}.jpg"
                with open(temp_path, "wb") as f:
                    f.write(img_bytes)

        if not temp_path:
            raise HTTPException(status_code=400, detail="No valid input provided.")

        # ----------------------------
        # Inference
        # ----------------------------
        THRESHOLD = 0.5  # for "FAKE" vs "REAL" decision boundary

        is_video = temp_path.endswith(".mp4")

        if is_video:
            label, score = predict_fake_real(temp_path)
            input_type = "video"
        else:
            label, score = predict_from_image(temp_path)
            input_type = "image"

        try:
            os.remove(temp_path)
        except Exception as e:
            logger.warning("Failed to remove temp file %s: %s", temp_path, e)

        if label is None or score is None:
            return {"error": "No faces detected."}

        prediction_id = str(uuid.uuid4())

        prediction_data = {
            "result": label,
            "score": score,
            "threshold": THRESHOLD,
            "prediction_id": prediction_id,
            "input_type": input_type
        }

        # ----------------------------
        # Store locally (JSONL)
        # ----------------------------
        _append_jsonl(PREDICTIONS_FILE, prediction_data)

        return prediction_data

    except HTTPException:
        raise
    except Exception as e:
        # Full traceback to logs
        logger.error("Unhandled error in /predict: %s\n%s", e, traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Ensure temp file is cleaned up even if something blew up mid-way
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass


@app.post("/feedback")
async def submit_feedback(feedback: FeedbackRequest):
    try:
        prediction_data = _find_prediction_by_id(feedback.prediction_id)
        if not prediction_data:
            raise HTTPException(status_code=404, detail="Prediction not found")

        is_correct = prediction_data["model_result"].upper() == feedback.user_actual_result.upper()

        feedback_data = {
            "id": str(uuid.uuid4()),
            "prediction_id": feedback.prediction_id,
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "model_result": prediction_data["model_result"],
            "confidence_score": prediction_data["confidence_score"],
            "user_actual_result": feedback.user_actual_result,
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
        raise HTTPException(status_code=500, detail=f"Failed to submit feedback: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
