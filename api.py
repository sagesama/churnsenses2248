"""FastAPI service. Run: uvicorn api:app --reload   (docs at /docs)"""
from typing import List, Literal
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from common import FEATURES, clean, load_model

app = FastAPI(title="ChurnSense API")
model = load_model()

class Customer(BaseModel):
    gender: str; SeniorCitizen: Literal[0, 1, "Yes", "No"]; Partner: str; Dependents: str
    tenure: int; PhoneService: str; MultipleLines: str; InternetService: str
    OnlineSecurity: str; OnlineBackup: str; DeviceProtection: str; TechSupport: str
    StreamingTV: str; StreamingMovies: str; Contract: str; PaperlessBilling: str
    PaymentMethod: str; MonthlyCharges: float; TotalCharges: float

def score(rows: List[Customer]):
    X = clean(pd.DataFrame([r.model_dump() for r in rows]))[FEATURES]
    p = model.predict_proba(X)[:, 1]
    return [{"churn_probability": round(float(v), 4), "predicted_churn": bool(v >= .5),
             "risk_level": "High" if v >= .6 else "Medium" if v >= .3 else "Low"} for v in p]

@app.get("/health")
def health(): return {"status": "ok"}

@app.post("/predict")
def predict(c: Customer): return score([c])[0]

@app.post("/predict/batch")
def predict_batch(cs: List[Customer]): return score(cs)
