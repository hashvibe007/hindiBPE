from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict
from app.bpe_tokenizer import BPETokenizer
import os

app = FastAPI(title="Hindi BPE Tokenizer API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize tokenizer and load model
tokenizer = BPETokenizer()
model_path = "bpe_model.json"
if os.path.exists(model_path):
    tokenizer.load_model(model_path)
else:
    print(f"Warning: Model file {model_path} not found. Starting with empty model.")


class TokenizeRequest(BaseModel):
    text: str


class TokenStats(BaseModel):
    original_chars: int
    token_count: int
    compression_ratio: float
    unique_tokens: int


class TokenizeResponse(BaseModel):
    original_text: str
    original_tokens: List[str]
    bpe_tokens: List[str]
    stats: Dict
    token_details: List[Dict]
    merge_history: List[Dict]


@app.post("/tokenize")
async def tokenize_text(request: TokenizeRequest):
    try:
        result = tokenizer.tokenize_with_details(request.text)
        # Ensure all required fields are present
        return {
            "original_text": result["original_text"],
            "original_tokens": result["original_tokens"],
            "bpe_tokens": result["bpe_tokens"],
            "stats": {
                "original_chars": result["stats"]["original_chars"],
                "token_count": len(result["bpe_tokens"]),
                "compression_ratio": result["stats"]["compression_ratio"],
                "unique_tokens": result["stats"]["unique_tokens"],
            },
            "token_details": [
                {
                    "token": token,
                    "type": tokenizer._get_token_type(token),
                    "length": len(token),
                }
                for token in result["bpe_tokens"]
            ],
            "merge_history": tokenizer.merge_history[-10:]
            if tokenizer.merge_history
            else [],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/vocabulary-stats")
async def get_vocab_stats():
    return {
        "vocab_size": len(tokenizer.vocab),
        "most_frequent_tokens": tokenizer.token_usage.most_common(20),
        "most_frequent_pairs": dict(
            sorted(
                tokenizer.pair_frequencies.items(), key=lambda x: x[1], reverse=True
            )[:20]
        ),
    }


@app.get("/training-progress")
async def get_training_progress():
    if hasattr(tokenizer, "training_progress"):
        return tokenizer.training_progress
    return {"message": "No training progress available"}
