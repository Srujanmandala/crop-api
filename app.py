# app.py
from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np

# Load model
model = pickle.load(open("crop_model.pkl", "rb"))

# FastAPI app
app = FastAPI()

# Input schema
class CropFeatures(BaseModel):
    N: float
    P: float
    K: float
    temperature: float
    humidity: float
    ph: float
    rainfall: float

# Routes
@app.get("/")
def home():
    return {"message": "Crop Recommendation API is running"}

@app.post("/predict")
def predict_crop(data: CropFeatures):
    features = np.array([[data.N, data.P, data.K, 
                          data.temperature, data.humidity, 
                          data.ph, data.rainfall]])
    prediction = model.predict(features)[0]
    return {"recommended_crop": prediction}
