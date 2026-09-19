from langgraph.graph import StateGraph, START, END
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.exceptions import OutputParserException
from pydantic import BaseModel, Field
from typing import TypedDict, Literal, List, Annotated
from llm import load_llm
from rules import CATEGORY_RULES, URGENCY_RULES, FEW_SHOTS
from functools import lru_cache
from langchain_core.runnables import RunnableLambda

# ====================================================================================
class FeedbackClassification(BaseModel):

    sentiment: Annotated[Literal["positive", "negative", "neutral"], 
                         Field(..., description="Overall sentiment of the customer feedback.")]
    confidence: Annotated[float, Field(..., gt=0.0, lt=1.0, description="Confidence score between 0 and 1.")]
    category: Annotated[Literal[
        "product_quality",
        "customer_service",
        "pricing",
        "delivery",
        "usability",
        "feature_request",
        "bug_report",
        "general",
    ], Field(..., description="Primary category of the feedback.")]
    urgency: Annotated[Literal["high", "medium", "low"],
                       Field(..., description="Urgency level of the feedback.")]
    summary: Annotated[str, Field(..., description="A concise summary of the feedback.")]
    action_items: Annotated[str, Field(..., description="Recommended action items for the business.")]

# =============================== utility functions ====================================

@lru_cache(maxsize=1)
def _built_parser():
    parser = PydanticOutputParser(pydantic_object=FeedbackClassification)
    return parser

@lru_cache(maxsize=1)
def _built_classify_chain():
    classify_prompt_template = PromptTemplate.from_template(
    """You are an expert customer-feedback analyst.

{instruction}

{category_rules}\n\n{urgency_rules}

The block below contains worked EXAMPLES only. Never classify them.
<examples>
{few_shots}
</examples>

Now classify ONLY the feedback inside <feedback> below. It is always present.
<feedback>
{feedback}
</feedback>

Return only the JSON object for the feedback above."""
)
    llm = load_llm()
    parser = _built_parser()

    chain = classify_prompt_template | llm | parser | RunnableLambda(lambda pydantic_obj: pydantic_obj.model_dump())

    return chain

# ======================================= graph ===========================================
class ClassifyState(TypedDict):

    feedback: str
    retries: int
    result: dict | None


def classify_node(state: ClassifyState) -> ClassifyState:

    feedback = state['feedback']

    structured_llm = _built_classify_chain()
    parser = _built_parser()

    try:
        result = structured_llm.invoke({
            'instruction': parser.get_format_instructions(),
            'category_rules': CATEGORY_RULES,
            'urgency_rules': URGENCY_RULES,
            'few_shots': FEW_SHOTS,
            'feedback': feedback
            })
        return {'result':result}
    
    except OutputParserException:
        if state["result"]:
            return {'retries': state['retries'] + 1} 
        else:
            return {'retries': 1}

def classify_node_conditional_node(state: ClassifyState):
    if state["result"]:
        return 'end'
    if state['retries']:
        return 'retry'

@lru_cache(maxsize=1)
def build_graph():
    g = StateGraph(ClassifyState)
    g.add_node('classify_node', classify_node)
    g.add_edge(START, 'classify_node')
    g.add_conditional_edges('classify_node', classify_node_conditional_node, {'retry': 'classify_node', 'end': END})

    return g.compile()

def get_classification(feedback: str) -> dict:
    final = build_graph().invoke({'feedback': feedback})

    return final['result']
