"""What to work on first: a priority table computed with pandas.

Each piece of feedback gets a weight, urgency (high 3, medium 2, low 1) times how
negative it is (negative 1, neutral 0.3, positive 0). A category's priority is the sum
of its weights, so it grows with volume, negativity and urgency together.
"""

from datetime import datetime, timedelta, timezone

import pandas as pd

from store import all_feedback

URGENCY_WEIGHT = {"high": 3, "medium": 2, "low": 1}
SENTIMENT_WEIGHT = {"negative": 1.0, "neutral": 0.3, "positive": 0.0}


def priorities(rows: list[dict] | None = None, now: datetime | None = None) -> pd.DataFrame:
    """One row per category, most pressing first."""
    frame = pd.DataFrame(all_feedback() if rows is None else rows)
    if frame.empty:
        return pd.DataFrame()
    frame["weight"] = frame["urgency"].map(URGENCY_WEIGHT) * frame["sentiment"].map(SENTIMENT_WEIGHT)
    frame["created_at"] = pd.to_datetime(frame["created_at"], utc=True)
    week_ago = (now or datetime.now(timezone.utc)) - timedelta(days=7)
    negative = frame["sentiment"] == "negative"
    frame["negative_last_7_days"] = negative & (frame["created_at"] >= week_ago)
    frame["negative_7_days_before"] = negative & frame["created_at"].between(week_ago - timedelta(days=7), week_ago,
                                                                            inclusive="left")
    frame = frame.sort_values("weight", ascending=False)

    def top_actions(items: pd.Series) -> list[str]:
        """The first (most urgent) action item of each of the three weightiest pieces of feedback."""
        return list(dict.fromkeys(actions[0] for actions in items if actions))[:3]

    table = frame.groupby("category").agg(
        priority=("weight", "sum"),
        feedback=("text", "size"),
        negative_share=("sentiment", lambda s: (s == "negative").mean()),
        high_urgency=("urgency", lambda s: (s == "high").sum()),
        negative_last_7_days=("negative_last_7_days", "sum"),
        negative_7_days_before=("negative_7_days_before", "sum"),
        example=("text", "first"),
        top_actions=("action_items", top_actions),
    )
    table["priority"] = table["priority"].round(1)
    table["negative_share"] = table["negative_share"].round(2)
    return table.sort_values(["priority", "feedback"], ascending=False).reset_index()
