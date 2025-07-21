from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel
from transformers import pipeline

API_KEY = "manognas-super-key"
API_KEY_NAME = "x-api-key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

app = FastAPI()

# Allow frontend access (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, specify domain like ["http://localhost:5500"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TextInput(BaseModel):
    text: str

# Load HuggingFace emotion classifier
emotion_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)

def get_api_key(api_key: str = Security(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid or missing API key")
    return api_key

@app.post("/analyze")
def analyze_text(input: TextInput, api_key: str = Depends(get_api_key)):
    results = emotion_classifier(input.text)[0]
    results.sort(key=lambda x: x["score"], reverse=True)
    top_result = results[0]

    return {
        "emotion": top_result["label"],
        "confidence": round(top_result["score"], 3),
        "all_scores": results
    }