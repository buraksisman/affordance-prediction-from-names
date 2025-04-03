from fastapi import FastAPI
from pydantic import BaseModel
import joblib

# Load once at startup
model = joblib.load("models/model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")
mlb = joblib.load("models/mlb.pkl")

# FastAPI app
app = FastAPI(title="Affordance Prediction API")

# Request schema
class ObjectInput(BaseModel):
    object_name: str

# Prediction route
@app.post("/predict")
def predict_affordances(input: ObjectInput):
    X = vectorizer.transform([input.object_name])
    y_pred = model.predict(X)
    predicted = mlb.inverse_transform(y_pred)
    return { "affordances": list(predicted[0]) if predicted else [] }
