"""HTTP API.  Run: uvicorn main:app --reload  (docs at /docs)"""

import io

import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile
from pydantic import BaseModel, Field

import store

app = FastAPI(title="Customer Feedback Classifier")


class Feedback(BaseModel):
    text: str = Field(min_length=1, max_length=5000)


@app.post("/classify")
def classify_one(feedback: Feedback) -> dict:
    return store.classify_and_save([feedback.text])[0]


@app.post("/classify/csv")
async def classify_csv(file: UploadFile) -> dict:
    frame = pd.read_csv(io.BytesIO(await file.read()))
    column = next((c for c in ("feedback", "text", "review") if c in frame.columns), None)
    if column is None:
        raise HTTPException(422, "the CSV needs a 'feedback', 'text' or 'review' column")
    results = store.classify_and_save(frame[column].astype(str).tolist())
    return {"classified": len(results), "results": results}


@app.get("/stats")
def get_stats() -> dict:
    return store.stats()
