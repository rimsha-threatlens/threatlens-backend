import re
import joblib
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="ThreatLens AI")

# SMS model
sms_model = joblib.load("SMS_Model/model.pkl")
sms_vectorizer = joblib.load("SMS_Model/vectorizer.pkl")

# URL model (files upload_url/URL_Model folder ke andar hain)
url_vectorizer = joblib.load("upload_url/URL_Model/url_vectorizer.pkl")
url_model = joblib.load("upload_url/URL_Model/url_text_model.pkl")


def clean_url(u):
    u = str(u).strip().lower()
    return re.sub(r'^[a-z]+://', '', u)


class SmsRequest(BaseModel):
    message: str


class UrlRequest(BaseModel):
    url: str


@app.get("/")
def home():
    return {"status": "ThreatLens AI backend is running"}


@app.post("/predict/sms")
def predict_sms(req: SmsRequest):
    X = sms_vectorizer.transform([req.message])
    pred = int(sms_model.predict(X)[0])
    label = "SPAM" if pred == 1 else "SAFE"
    return {"prediction": label}


@app.post("/predict/url")
def predict_url(req: UrlRequest):
    X = url_vectorizer.transform([clean_url(req.url)])
    prob = float(url_model.predict_proba(X)[0][1])
    label = "PHISHING" if prob >= 0.5 else "LEGITIMATE"
    return {"prediction": label, "phishing_probability": round(prob, 3)}
