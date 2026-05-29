import base64
import io
import json
import os
import time
from typing import Any

import numpy as np
import tritonclient.grpc as grpcclient
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from PIL import Image
from starlette.concurrency import run_in_threadpool
from tritonclient.utils import InferenceServerException

TRITON_URL = os.getenv("TRITON_URL", "triton:8001")
MODEL_NAME = os.getenv("MODEL_NAME", "image_classifier")
IMG_SIZE = int(os.getenv("IMG_SIZE", "128"))
CLASSES_PATH = os.getenv("CLASSES_PATH", "/models/image_classifier/class_names.json")

app = FastAPI(
    title="Image Classification Gateway",
    description="FastAPI gateway for NVIDIA Triton image classification model",
    version="1.0.0",
)

triton_client: grpcclient.InferenceServerClient | None = None
input_name: str | None = None
output_name: str | None = None
class_names: list[str] = []


def load_classes() -> list[str]:
    if not os.path.exists(CLASSES_PATH):
        raise RuntimeError(f"Classes file not found: {CLASSES_PATH}")
    with open(CLASSES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE))
    array = np.asarray(image, dtype=np.float32) / 255.0
    return array


def decode_base64_image(value: str) -> bytes:
    # Поддерживает обычный base64 и data URL вида data:image/png;base64,...
    if "," in value and value.strip().lower().startswith("data:"):
        value = value.split(",", 1)[1]
    return base64.b64decode(value)


async def read_images_from_request(
    request: Request,
    file: UploadFile | None = None,
    image: str | None = None,
) -> list[bytes]:
    if file is not None:
        return [await file.read()]

    if image:
        return [decode_base64_image(image)]

    try:
        payload: dict[str, Any] = await request.json()
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail="Upload file/files in Swagger UI or send JSON with image/images field",
        ) from exc

    if "image" in payload:
        return [decode_base64_image(payload["image"])]
    if "images" in payload:
        return [decode_base64_image(item) for item in payload["images"]]

    raise HTTPException(status_code=400, detail="JSON must contain image or images field")


def infer(batch: np.ndarray) -> np.ndarray:
    if triton_client is None or input_name is None or output_name is None:
        raise RuntimeError("Triton client is not initialized")

    infer_input = grpcclient.InferInput(input_name, batch.shape, "FP32")
    infer_input.set_data_from_numpy(batch.astype(np.float32))
    infer_output = grpcclient.InferRequestedOutput(output_name)

    response = triton_client.infer(
        model_name=MODEL_NAME,
        inputs=[infer_input],
        outputs=[infer_output],
    )
    result = response.as_numpy(output_name)
    if result is None:
        raise RuntimeError(f"Triton returned empty output: {output_name}")
    return result


@app.on_event("startup")
def startup() -> None:
    global triton_client, input_name, output_name, class_names

    class_names = load_classes()
    triton_client = grpcclient.InferenceServerClient(url=TRITON_URL, verbose=False)

    last_error: Exception | None = None
    for _ in range(30):
        try:
            if triton_client.is_server_live() and triton_client.is_model_ready(MODEL_NAME):
                metadata = triton_client.get_model_metadata(MODEL_NAME)
                input_name = metadata.inputs[0].name
                output_name = metadata.outputs[0].name
                return
        except Exception as exc:
            last_error = exc
        time.sleep(2)

    raise RuntimeError(f"Triton model is not ready: {last_error}")


@app.get("/")
def root() -> dict[str, Any]:
    return {
        "service": "Image Classification Gateway",
        "model": MODEL_NAME,
        "endpoints": ["GET /health", "GET /classes", "POST /predict"],
    }


@app.get("/health")
def health() -> dict[str, Any]:
    try:
        live = triton_client.is_server_live() if triton_client else False
        ready = triton_client.is_model_ready(MODEL_NAME) if triton_client else False
        return {
            "status": "healthy" if live and ready else "unhealthy",
            "triton_live": live,
            "model_ready": ready,
            "model_name": MODEL_NAME,
            "input_name": input_name,
            "output_name": output_name,
            "classes_loaded": len(class_names),
        }
    except InferenceServerException as exc:
        return {"status": "unhealthy", "error": str(exc)}


@app.get("/classes")
def classes() -> dict[str, Any]:
    return {"classes": class_names}


@app.post("/predict")
async def predict(
    request: Request,
    file: UploadFile | None = File(default=None, description="Single image file"),
    image: str | None = Form(default=None, description="Base64 image or data URL"),
) -> dict[str, Any]:
    try:
        raw_images = await read_images_from_request(
            request,
            file=file,
            image=image,
        )
        batch = await run_in_threadpool(
            lambda: np.stack([preprocess_image(item) for item in raw_images]).astype(np.float32)
        )

        start = time.perf_counter()
        scores = await run_in_threadpool(infer, batch)
        inference_ms = (time.perf_counter() - start) * 1000

        predictions = []
        for row in scores:
            class_id = int(np.argmax(row))
            predictions.append({
                "class_id": class_id,
                "class_name": class_names[class_id],
                "confidence": float(row[class_id]),
                "probabilities": {
                    class_name: float(probability)
                    for class_name, probability in zip(class_names, row)
                },
            })

        return {
            "model": MODEL_NAME,
            "batch_size": len(predictions),
            "inference_ms": round(inference_ms, 3),
            "predictions": predictions,
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
