from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import os

# Correct path to model.pkl
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "model.pkl")

model = joblib.load(MODEL_PATH)

app = FastAPI()

class InputData(BaseModel):
    feature1: float
    feature2: float
    age: int

@app.post("/predict")
def predict(data: InputData):
    features = [[data.feature1, data.feature2, data.age]]
    prediction = model.predict(features)
    return {"prediction": int(prediction[0])}
