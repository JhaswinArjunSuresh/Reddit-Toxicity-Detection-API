from fastapi import FastAPI
from pydantic import BaseModel
from .toxicity_model import ToxicityClassifier

app = FastAPI()
classifier = ToxicityClassifier()

class TextInput(BaseModel):
    text: str

@app.post("/classify")
def classify_text(input: TextInput):
    result = classifier.predict(input.text)
    return {"text": input.text, "toxicity": result}

@app.get("/health")
def health():
    return {"status": "ok"}

