from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models
logistic_model = joblib.load("models/logistic_model.joblib")
tree_model = joblib.load("models/decision_tree_model.joblib")

# Request schema
class PredictionInput(BaseModel):
    invitations: int
    is_weekend: int

@app.get("/")
def home():
    return {"message": "Event Guest Attendance Predictor API running"}

@app.post("/predict/logistic")
def predict_logistic(data: PredictionInput):
    features = np.array([[data.invitations, data.is_weekend]])
    prediction = logistic_model.predict(features)[0]
    return {"prediction": int(prediction)}

@app.post("/predict/tree")
def predict_tree(data: PredictionInput):
    features = np.array([[data.invitations, data.is_weekend]])
    prediction = tree_model.predict(features)[0]
    return {"prediction": int(prediction)}
