from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import numpy as np
import cv2
from pathlib import Path

from busbar.features import preprocess_image_to_features
from busbar.model_criticality_based import CriticalityBasedModel


app = FastAPI(title="Busbar Heat Detection API")
MODEL_DIR = Path(__file__).parent / "artifacts_criticality"


class Prediction(BaseModel):
    load_category: str
    criticality_score: float


@app.on_event("startup")
def load_models():
    global model
    model = CriticalityBasedModel.load(str(MODEL_DIR))


@app.post("/predict", response_model=Prediction)
async def predict_api(file: UploadFile = File(...)):
    data = await file.read()
    file_bytes = np.frombuffer(data, np.uint8)
    bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if bgr is None:
        return {"load_category": "Low Load", "criticality_score": 0.0}
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    feats, _ = preprocess_image_to_features(rgb, mode="rgb_pseudocolor", min_temp_c=20, max_temp_c=120)
    load_category, criticality = model.predict(feats.reshape(1, -1))
    return {"load_category": str(load_category[0]), "criticality_score": float(criticality[0])}


