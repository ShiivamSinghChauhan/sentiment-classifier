"""HTTP API.  Run: uvicorn main:app --reload  (docs at /docs)"""

from fastapi import FastAPI
from pydantic import BaseModel, Field

import store

app = FastAPI(title="Customer Feedback Classifier")


class Feedback(BaseModel):
    text: str = Field(min_length=1, max_length=5000)


@app.post("/classify")
def classify_one(feedback: Feedback) -> dict:
    return store.classify_and_save([feedback.text])[0]


@app.get("/stats")
def get_stats() -> dict:
    return store.stats()
