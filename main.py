"""HTTP API.  Run: uvicorn main:app --reload  (docs at /docs)"""

import io
import json

import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel, Field

import insights
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


@app.get("/priorities")
def get_priorities() -> list[dict]:
    """Categories ranked by priority, with counts, trend, an example and the top action items."""
    return json.loads(insights.priorities().to_json(orient="records"))


@app.get("/priorities.csv")
def get_priorities_csv() -> Response:
    table = insights.priorities()
    if not table.empty:
        table["top_actions"] = table["top_actions"].str.join("; ")
    return Response(table.to_csv(index=False), media_type="text/csv",
                    headers={"Content-Disposition": "attachment; filename=priorities.csv"})


@app.get("/insights")
def get_insights() -> dict:
    return insights.report()
