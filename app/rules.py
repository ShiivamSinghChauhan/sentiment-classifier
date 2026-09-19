
CATEGORY_RULES = """Categories:
delivery=shipping/packaging/damaged-late-wrong arrival
bug_report=crash/error/outage/wrong result/freezing/lag
usability=works but confusing/hard to use
feature_request=add feature/increase limit
pricing=cost/billing/refund/charges
customer_service=support interaction
product_quality=intrinsic quality/effectiveness
general=otherwise"""

URGENCY_RULES = """Urgency:
high=outage/security/data loss/billing error/blocked flow/damaged arrival/ignored support/churn risk
medium=non-blocking bug/performance/usability/missing feature/repeated complaint
low=praise/minor issue/neutral"""

FEW_SHOTS = """Examples (study the reasoning):

Feedback: "My package arrived crushed and the box was ripped open."
-> {"sentiment": "negative", "category": "delivery", "urgency": "high"}
(Arrived damaged = delivery failure, not product_quality. Damaged goods = high.)

Feedback: "The dashboard freezes constantly and sometimes shows last week's numbers."
-> {"sentiment": "negative", "category": "bug_report", "urgency": "medium"}
(Malfunction = bug_report even though it feels like a usability problem. Not blocking core use = medium.)

Feedback: "I can never remember where the settings live; the menus are a maze."
-> {"sentiment": "negative", "category": "usability", "urgency": "medium"}
(Nothing is broken — it's hard to use. usability, not bug_report.)"""