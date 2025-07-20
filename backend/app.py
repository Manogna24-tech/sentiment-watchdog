from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel
from transformers import pipeline
import torch

API_KEY = "manognas-super-key"
API_KEY_NAME = "x-api-key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

app = FastAPI()

class TextInput(BaseModel):
    text: str

# Load the emotion classification pipeline
emotion_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)

def get_api_key(api_key: str = Security(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid or missing API key")
    return api_key

@app.post("/analyze")
def analyze_text(input: TextInput, api_key: str = Depends(get_api_key)):
    results = emotion_classifier(input.text)[0]  # List of dicts with label/score

    # Sort by score descending
    results.sort(key=lambda x: x['score'], reverse=True)
    top_result = results[0]

    return {
        "emotion": top_result['label'],
        "confidence": round(top_result['score'], 3),
        "all_scores": [
            {
                "label": r['label'],
                "score": r['score']
            } for r in results
        ]
    }