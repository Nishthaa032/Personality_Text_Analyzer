from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.personality.inference import PersonalityPredictor
from src.personality.personality_summary import build_personality_summary

app = FastAPI(
    title="Personality Prediction API",
    version="0.1.0",
    description="Personality-only text scoring API built from the verified TF-IDF Big Five model.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://localhost:8080",
        "http://127.0.0.1:8080",
        "http://localhost:8081",
        "http://127.0.0.1:8081",
        "null",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

predictor = PersonalityPredictor()


class PredictTextRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Input text to score for Big Five personality traits.")


class PredictBatchRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, description="List of texts to score.")


class PredictionResponse(BaseModel):
    model: str = "tfidf_ridge"
    traits: dict[str, float]


class SummaryResponse(BaseModel):
    summary: str


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "model": "personality_tfidf_ridge"}


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictTextRequest) -> PredictionResponse:
    text = payload.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="text must not be empty")

    traits = predictor.predict_text(text)
    return PredictionResponse(traits=traits)


@app.post("/predict/batch", response_model=list[PredictionResponse])
def predict_batch(payload: PredictBatchRequest) -> list[PredictionResponse]:
    texts = [text.strip() for text in payload.texts]
    if not texts or any(not text for text in texts):
        raise HTTPException(status_code=400, detail="texts must contain at least one non-empty string")

    results = predictor.predict_batch(texts)
    return [PredictionResponse(traits=traits) for traits in results]


@app.post("/summarize", response_model=SummaryResponse)
def summarize(payload: PredictTextRequest) -> SummaryResponse:
    text = payload.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="text must not be empty")

    traits = predictor.predict_text(text)
    return SummaryResponse(summary=build_personality_summary(traits))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
