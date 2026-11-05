"""Compare labelers on data/gold.jsonl: Jev, ScaleDown's classify model (a CDM with a
Jev-compatible API), gpt-oss-20b, and the cascade (Jev, LLM when unsure).

Each labeler's answers are cached per item in data/eval_results.jsonl, so a rerun
only calls what is missing and every row of the report comes from the same answers.
Run: python evaluate.py   (--fresh to ask everything again)
"""

import json
import os
import statistics
import sys
import time
from pathlib import Path

import httpx
from langchain_core.callbacks import UsageMetadataCallbackHandler

import classify

DATA = Path(__file__).parent / "data"
RESULTS = DATA / "eval_results.jsonl"
PRICES = {"jev": 0.042, "llm": (0.075, 0.30)}  # USD per 1M tokens: Jev input; gpt-oss-20b input, output
SCALEDOWN_QUESTIONS = {name: q.model_dump(mode="json", exclude_none=True) for name, q in classify.QUESTIONS.items()}


def ask_jev(text: str) -> dict:
    jev = classify.label_with_jev({"text": text})
    return {"labels": jev["labels"], "confidence": jev["confidence"],
            "cost": jev["jev_tokens"] * PRICES["jev"] / 1e6}


def ask_scaledown(text: str) -> dict:
    response = httpx.post("https://api.scaledown.xyz/v1/scaledown", timeout=60,
                          headers={"x-api-key": os.environ["SCALEDOWN_API_KEY"]},
                          json={"state": {"text": text}, "questions": SCALEDOWN_QUESTIONS})
    response.raise_for_status()
    data = response.json()
    answers = data["answers"]
    return {"labels": {f: answers[f]["choice"] for f in classify.FIELDS},
            "confidence": {f: round(answers[f]["probabilities"][answers[f]["choice"]], 3) for f in classify.FIELDS},
            "cost": data["usage"]["cost"]}


def ask_llm(text: str) -> dict:
    usage = UsageMetadataCallbackHandler()
    labels = classify.second_opinion.invoke(classify.LLM_PROMPT.format(text=text), config={"callbacks": [usage]})
    tokens = next(iter(usage.usage_metadata.values()))
    return {"labels": labels.model_dump(),
            "cost": (tokens["input_tokens"] * PRICES["llm"][0] + tokens["output_tokens"] * PRICES["llm"][1]) / 1e6}


LABELERS = {"jev": ask_jev, "scaledown": ask_scaledown, "llm": ask_llm}


def score(gold: list[dict]) -> list[dict]:
    cached = {} if "--fresh" in sys.argv or not RESULTS.exists() else \
        {r["text"]: r for r in map(json.loads, RESULTS.open(encoding="utf-8"))}
    rows, failed = [], set()
    for i, item in enumerate(gold, 1):
        row = cached.get(item["text"], {}) | item
        for name, ask in LABELERS.items():
            if name in failed or (isinstance(row.get(name), dict) and "labels" in row[name]):
                continue
            started = time.perf_counter()
            try:
                row[name] = ask(item["text"]) | {"ms": (time.perf_counter() - started) * 1000}
            except httpx.HTTPError as error:  # e.g. ScaleDown answers 402 when out of credits
                print(f"{name} unavailable, leaving it out: {error}")
                failed.add(name)
        rows.append(row)
        print(f"[{i}/{len(gold)}] " + "  ".join(f"{n}={list(row[n]['labels'].values())}" for n in LABELERS if n in row))
    RESULTS.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return rows


def cascade(row: dict, sure: float) -> dict:
    """Jev's labels, with the LLM's label for every field Jev is unsure about."""
    return {f: row["llm"]["labels"][f] if row["jev"]["confidence"][f] < sure else row["jev"]["labels"][f]
            for f in classify.FIELDS}


def accuracy(rows: list[dict], predict) -> dict:
    result = {f: round(statistics.mean(predict(r)[f] == r[f] for r in rows), 3) for f in classify.FIELDS}
    result["all_three"] = round(statistics.mean(all(predict(r)[f] == r[f] for f in classify.FIELDS) for r in rows), 3)
    return result


def main():
    rows = score([json.loads(line) for line in (DATA / "gold.jsonl").open(encoding="utf-8")])
    available = [name for name in LABELERS if all(name in r for r in rows)]
    cost = {name: statistics.mean(r[name]["cost"] for r in rows) for name in available}
    report = {"items": len(rows)}
    for name in available:
        report[f"{name}_only"] = {**accuracy(rows, lambda r, n=name: r[n]["labels"]),
                                  "cost_per_1k_usd": round(1000 * cost[name], 4),
                                  "latency_ms_mean": round(statistics.mean(r[name]["ms"] for r in rows))}
    for sure in (0.5, 0.6, 0.7, 0.8, 0.9):
        sent = statistics.mean(min(r["jev"]["confidence"].values()) < sure for r in rows)
        report[f"cascade_sure_{sure}"] = {**accuracy(rows, lambda r: cascade(r, sure)), "sent_to_llm": round(sent, 3),
                                          "cost_per_1k_usd": round(1000 * (cost["jev"] + sent * cost["llm"]), 4)}
    (DATA / "eval_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
