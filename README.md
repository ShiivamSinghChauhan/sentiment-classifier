# sentiment-classifier

Turns raw customer feedback into something a team can act on. Each message gets a **sentiment**, a **category** and an **urgency**, and anything that needs a reply also gets a short summary and action items. A priority table then shows which category to fix first.

## Built with

Python 3.13 · Jev (TypeSafe) for labelling · gpt-oss-20b on Groq for second opinions · LangGraph · FastAPI · SQLite · pandas · Streamlit

## What makes it different

- **Probabilities, not guesses.** Jev answers all three labels in one request and returns a probability for each option, so the app knows when it is unsure.
- **LLM only for hard cases.** A field below 0.6 confidence is sent to the LLM for a second opinion. About 8% of messages need it.
- **Measured, not assumed.** On a 132-message hand-labelled gold set, Jev alone gets all three labels right 91.7% of the time, against 66.7% for the LLM alone. It is also about 7x faster (395 ms vs 2.7 s) and cheaper. The cascade gets 92.4%.
- **Numbers come from code.** The priority score (urgency x negativity, summed per category) is computed in pandas; the LLM only writes the report from that table.

## Run it

```bash
uv sync
# .env: TYPESAFE_API_KEY, GROQ_API_KEY
uv run uvicorn main:app          # API, docs at /docs
uv run streamlit run app.py      # dashboard
uv run python evaluate.py        # reproduce the evaluation
```

API: `POST /classify`, `POST /classify/csv`, `GET /stats`, `GET /priorities`, `GET /insights`.
