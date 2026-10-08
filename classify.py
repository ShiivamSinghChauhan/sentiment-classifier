"""Label customer feedback with sentiment, category and urgency.

The labels are Choice questions for Jev, TypeSafe's decision model. One request
returns a probability for every option of every question.
"""

from dotenv import load_dotenv
from typesafe_sdk import Choice, TypeSafeClient

load_dotenv()

JEV_MODEL = "jev-1.14.0"

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


def classify(text: str) -> dict:
    response = jev.system_one(state=text, questions=QUESTIONS)
    labels = {field: response.choices[field].choice for field in FIELDS}
    confidence = {field: round(response.choices[field].probabilities[labels[field]], 3) for field in FIELDS}
    return {"text": text, **labels, "confidence": confidence}


if __name__ == "__main__":
    import json
    import sys

    print(json.dumps(classify(" ".join(sys.argv[1:]) or "My parcel arrived crushed."), indent=2))
