from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI(title="ThreatLens AI")

sms_model = joblib.load("SMS_Model/model.pkl")
sms_vectorizer = joblib.load("SMS_Model/vectorizer.pkl")

class SmsRequest(BaseModel):
    message: str

@app.get("/")
def home():
    return {"status": "ThreatLens AI backend is running"}

@app.post("/predict/sms")
def predict_sms(req: SmsRequest):
    X = sms_vectorizer.transform([req.message])
    pred = int(sms_model.predict(X)[0])
    label = "SPAM" if pred == 1 else "SAFE"
    return {"prediction": label}