import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / 'data/feedback.db'

def get_conn():
    DB_PATH.parent.mkdir(exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("""CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    text TEXT,
    sentiment TEXT,
    confidence REAL,
    category TEXT,
    urgency TEXT,
    summary TEXT,
    action_items TEXT,
    error TEXT,
    verified INTEGER DEFAULT 0,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )""")

    # Migrate older DBs that predate newer columns.
    cols = {r[1] for r in conn.execute("PRAGMA table_info(feedback)").fetchall()}
    if "error" not in cols:
        conn.execute("ALTER TABLE feedback ADD COLUMN error TEXT")
    if "verified" not in cols:
        conn.execute("ALTER TABLE feedback ADD COLUMN verified INTEGER DEFAULT 0")
    conn.commit()
    return conn

def save_feedback(result: dict):
    conn = get_conn()
    conn.execute(
        """
        INSERT INTO feedback (text, sentiment, confidence, category, urgency, summary, action_items, error, verified) 
        VALUES (?,?,?,?,?,?,?,?,?)
        """,
        (result["text"], result["sentiment"], result["confidence"], result["category"], result["urgency"], result["summary"], result["action_items"], result.get('error'), int(result.get("verified", False)))
    )
    conn.commit()
    conn.close()


def save_feedback_many(results: list[dict]):
    """insert a batch of results in a single transaction"""
    if not results:
        return
    conn = get_conn()
    conn.executemany(
        "INSERT INTO feedback (text, sentiment, confidence, category, urgency, summary, action_items, error, verified) VALUES (?,?,?,?,?,?,?,?,?)",
        [
            (r["text"], r["sentiment"], r["confidence"], r["category"], r["urgency"], r["summary"], r["action_items"], r.get("error"), int(r.get("verified", False))) 
            for r in results
        ]
    )
    conn.commit()
    conn.close()


def get_all_feedback() -> list[dict]:
    conn = get_conn()
    rows = conn.execute("SELECT * FROM feedback ORDER BY timestamp DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]

def get_stats() -> dict:
    conn = get_conn()
    total = conn.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]
    sentiments = conn.execute("SELECT sentiment, COUNT(*) as cnt FROM feedback GROUP BY sentiment").fetchall()
    urgencies = conn.execute("SELECT urgency, COUNT(*) as cnt FROM feedback GROUP BY urgency").fetchall()
    categories = conn.execute("SELECT category, COUNT(*) as cnt FROM feedback GROUP BY category ORDER BY cnt DESC").fetchall()
    conn.close()
    return {
        "total": total,
        "sentiments": {r[0]: r[1] for r in sentiments},
        "urgencies": {r[0]: r[1] for r in urgencies},
        "categories": {r[0]: r[1] for r in categories}
    }

