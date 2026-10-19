"""SQLite storage for classified feedback, plus batch classification."""

import json
import sqlite3
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from classify import classify

DB = Path(__file__).parent / "data" / "feedback.db"


def connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    conn.execute("""CREATE TABLE IF NOT EXISTS feedback (
        id INTEGER PRIMARY KEY,
        text TEXT NOT NULL,
        sentiment TEXT, category TEXT, urgency TEXT,
        confidence TEXT,
        reviewed_by_llm INTEGER,
        summary TEXT,
        action_items TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP)""")
    return conn


def classify_and_save(texts: list[str]) -> list[dict]:
    texts = [t.strip() for t in texts if t and t.strip()]
    with ThreadPoolExecutor(max_workers=4) as pool:  # Jev is fast; this mostly overlaps network waits
        results = list(pool.map(classify, texts))
    with connect() as conn:
        conn.executemany(
            "INSERT INTO feedback (text, sentiment, category, urgency, confidence, reviewed_by_llm, "
            "summary, action_items) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [(r["text"], r["sentiment"], r["category"], r["urgency"], json.dumps(r["confidence"]),
              r["reviewed_by_llm"], r["summary"], json.dumps(r["action_items"])) for r in results])
    return results


def all_feedback() -> list[dict]:
    with connect() as conn:
        rows = [dict(row) for row in conn.execute("SELECT * FROM feedback ORDER BY id DESC")]
    for row in rows:
        row["action_items"] = json.loads(row["action_items"] or "[]")
    return rows


def stats() -> dict:
    rows = all_feedback()
    return {
        "total": len(rows),
        "sentiment": Counter(r["sentiment"] for r in rows),
        "category": Counter(r["category"] for r in rows),
        "urgency": Counter(r["urgency"] for r in rows),
        "reviewed_by_llm": sum(r["reviewed_by_llm"] for r in rows),
    }
