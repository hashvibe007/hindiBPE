from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict
from app.bpe_tokenizer import BPETokenizer
import os
import asyncio

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
model_path = "bpe_model_latest.json"
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


# Add WebSocket manager
class ConnectionManager:
    def __init__(self):
        self.active_connections = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            await connection.send_json(message)


manager = ConnectionManager()


# Add WebSocket endpoint
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except:
        manager.disconnect(websocket)


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


@app.get("/training-stats")
async def get_training_stats():
    """Get detailed training statistics"""
    if hasattr(tokenizer, "vocab_growth"):
        return {
            "vocab_growth": tokenizer.vocab_growth,
            "base_vocab_stats": tokenizer.base_vocab_stats,
            "current_stats": {
                "total_vocab": len(tokenizer.vocab),
                "learned_vocab": len(tokenizer.learned_vocab),
                "base_vocab": len(tokenizer.BASE_VOCAB),
            },
        }
    return {"message": "No training statistics available"}


@app.post("/resume-training")
async def resume_training(checkpoint_file: str):
    """Resume training from checkpoint"""
    try:
        tokenizer.learn_bpe(text, resume_from=checkpoint_file)
        return {"message": "Training resumed successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/start-training")
async def start_training(request: dict):
    try:
        text = load_sample_data(
            "data/hindi_wiki_corpus.txt",
            max_sentences=request.get("max_sentences", 10000),
        )
        tokenizer = await train_and_save_bpe(
            text, vocab_size=request.get("vocab_size", 10000), manager=manager
        )
        return {"message": "Training completed successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
