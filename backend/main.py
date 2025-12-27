from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI()

logistic_model = joblib.load("models/logistic_model.joblib")
tree_model = joblib.load("models/decision_tree_model.joblib")

@app.get("/")
def home():
    return {"message": "API running"}

@app.post("/predict/logistic")
def predict_logistic(features: list):
    data = np.array(features).reshape(1, -1)
    return {"prediction": int(logistic_model.predict(data)[0])}

@app.post("/predict/tree")
def predict_tree(features: list):
    data = np.array(features).reshape(1, -1)
    return {"prediction": int(tree_model.predict(data)[0])}
