from fastapi import FastAPI, Header, HTTPException, Depends, Security
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel

API_KEY = "manognas-super-key"
API_KEY_NAME = "x-api-key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

app = FastAPI()

class TextInput(BaseModel):
    text: str

def get_api_key(api_key: str = Security(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid or missing API key")
    return api_key

@app.post("/analyze")
def analyze_text(input: TextInput, api_key: str = Depends(get_api_key)):
    # Your sentiment analysis logic here
    return {"message": "Text processed successfully", "input": input.text}