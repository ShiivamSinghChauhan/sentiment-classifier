"""Label customer feedback with sentiment, category and urgency.

Jev answers all three labels as Choice questions in one request and returns a
probability for every label. When a field's top probability is below SURE, an LLM
gives a second opinion on that field, so the LLM only sees the hard cases.

    jev --(unsure)--> review --> END
     +--(sure)----------------> END
"""

from typing import Literal, TypedDict

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel
from typesafe_sdk import Choice, TypeSafeClient

load_dotenv()

JEV_MODEL = "jev-1.14.0"
SURE = 0.6  # below this probability a field counts as unsure (tuned in evaluate.py)

SENTIMENT = {
    "positive": "The customer is mainly pleased: praise outweighs any complaint.",
    "negative": "The main point is a complaint, a problem or frustration, including a missing feature "
                "that stops the customer from buying.",
    "neutral": "A question, suggestion or plain statement with no clear satisfaction or dissatisfaction: "
               "'works as described', or a small flaw mentioned calmly next to something fine.",
}

CATEGORY = {
    "delivery": "Shipping speed, tracking, couriers, packaging, or items that arrived late, damaged, "
                "missing or wrong. Damage in transit is delivery, not product quality.",
    "bug_report": "Something in the website or app is broken: crashes, errors, outages, wrong data, "
                  "freezing, slowness, failed checkout, or text that can't be read on screen.",
    "usability": "The website or app works as designed but is confusing, hard to find or needs too many steps.",
    "feature_request": "Asks for something new in the shop, site or app, or for a higher limit.",
    "pricing": "Prices, value for money, fees, billing, charges, refunds, discounts or subscriptions.",
    "customer_service": "The experience with support staff, chat, phone or email: speed, tone, competence.",
    "product_quality": "The product itself: materials, durability, performance, safety, or the "
                       "instructions that come with it.",
    "general": "Anything else: the company in general, marketing, jobs, account or personal-data requests.",
}

URGENCY = {
    "high": "Needs action today: an outage, a security or privacy issue, data loss, money charged wrongly "
            "or a refund missing, the customer cannot buy or use the product at all, an item arrived "
            "damaged, missing or wrong, a safety hazard, support ignored them repeatedly, or they "
            "threaten to leave. One feature failing while the customer can still shop is not high.",
    "medium": "A real problem that hurts the experience but blocks nothing: one feature or page that keeps "
              "failing (a crash on upload, frequent logouts, a broken page), slowness, a confusing flow, "
              "a late delivery, a defect that shows up in use, an unhappy support contact.",
    "low": "Praise, neutral questions, suggestions, or small annoyances that make no task harder, such as "
           "a cosmetic mistake in an email or too many pop-ups.",
}

QUESTIONS = {
    "sentiment": Choice(instructions="What is the overall sentiment of this customer feedback?", criteria=SENTIMENT),
    "category": Choice(instructions="What is this customer feedback mainly about?", criteria=CATEGORY),
    "urgency": Choice(instructions="How urgently should the company act on this feedback?", criteria=URGENCY),
}
FIELDS = list(QUESTIONS)

jev = TypeSafeClient(model=JEV_MODEL)
llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0, reasoning_effort="low", max_retries=6)


class Labels(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"]
    category: Literal["delivery", "bug_report", "usability", "feature_request", "pricing",
                      "customer_service", "product_quality", "general"]
    urgency: Literal["high", "medium", "low"]


def rubric(options: dict) -> str:
    return "\n".join(f"- {name}: {meaning}" for name, meaning in options.items())


LLM_PROMPT = f"""Label this customer feedback for an online store.

Sentiment:
{rubric(SENTIMENT)}

Category (pick the main topic):
{rubric(CATEGORY)}

Urgency:
{rubric(URGENCY)}

Feedback: {{text}}"""

# strict JSON schema: without it gpt-oss-20b sometimes wraps the answer in the schema itself
second_opinion = llm.with_structured_output(Labels, method="json_schema", strict=True).with_retry(stop_after_attempt=3)


class State(TypedDict, total=False):
    text: str
    labels: dict  # field -> label
    confidence: dict  # field -> Jev's probability for the chosen label
    reviewed_by_llm: bool
    jev_tokens: int


def label_with_jev(state: State) -> dict:
    response = jev.system_one(state=state["text"], questions=QUESTIONS)
    labels = {field: response.choices[field].choice for field in FIELDS}
    confidence = {field: round(response.choices[field].probabilities[labels[field]], 3) for field in FIELDS}
    return {"labels": labels, "confidence": confidence, "reviewed_by_llm": False,
            "jev_tokens": response.usage.input_tokens or 0}


def label_with_llm(state: State) -> dict:
    llm_labels = second_opinion.invoke(LLM_PROMPT.format(text=state["text"])).model_dump()
    unsure = [field for field in FIELDS if state["confidence"][field] < SURE]
    return {"labels": state["labels"] | {field: llm_labels[field] for field in unsure}, "reviewed_by_llm": True}


def after_jev(state: State) -> str:
    if min(state["confidence"].values()) < SURE:
        return "review"
    return END


graph = (
    StateGraph(State)
    .add_node("jev", label_with_jev)
    .add_node("review", label_with_llm)
    .add_edge(START, "jev")
    .add_conditional_edges("jev", after_jev, ["review", END])
    .add_edge("review", END)
    .compile()
)


def classify(text: str) -> dict:
    state = graph.invoke({"text": text})
    return {"text": text, **state["labels"], "confidence": state["confidence"],
            "reviewed_by_llm": state["reviewed_by_llm"]}


if __name__ == "__main__":
    import json
    import sys

    print(json.dumps(classify(" ".join(sys.argv[1:]) or "My parcel arrived crushed."), indent=2))
