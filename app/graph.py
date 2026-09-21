from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.exceptions import OutputParserException
# from langchain_core.runnables import RunnableLambda
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Literal, Annotated
from pydantic import BaseModel, Field
from llm import load_llm
from rules import CATEGORY_RULES, URGENCY_RULES, FEW_SHOTS
from functools import lru_cache

# ====================================================================================
class FeedbackClassification(BaseModel):

    sentiment: Annotated[Literal["positive", "negative", "neutral"], 
                         Field(..., description="Overall sentiment of the customer feedback.")]
    confidence: Annotated[float, Field(..., ge=0.0, le=1.0, description="Confidence score between 0 and 1.")]
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
def _built_classify_chain(temprature):
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
    llm = load_llm(temprature)

    chain = classify_prompt_template | llm

    return chain

@lru_cache
def _built_retry_chain(temprature):
    retry_prompt_template = PromptTemplate.from_template(
        """Previous model returned a output which raise the error while parsing, read the schema and the error and fix the previous output. You must return valid JSON.
Schema:
{instructions}

{category_rules}\n\n{urgency_rules}
<feedback>
{feedback}
<feedback>

Original feedback being classified:

Previous output:
{bad_output}

Parser error:
{error}

Fix the output and return ONLY valid JSON."""
    )
    llm = load_llm(temprature)

    chain = retry_prompt_template | llm

    return chain
# ======================================= graph ===========================================
class ClassifyState(TypedDict):

    feedback: str
    retries: int
    last_result: str | None
    parsed_result: dict | None
    last_excep: str | None


MAX_RETRIES = 2
def classify_node(state: ClassifyState) -> ClassifyState:

    feedback = state['feedback']

    structured_llm = _built_classify_chain(temprature=0)
    parser = _built_parser()

    last_result = structured_llm.invoke({
        'instruction': parser.get_format_instructions(),
        'category_rules': CATEGORY_RULES,
        'urgency_rules': URGENCY_RULES,
        'few_shots': FEW_SHOTS,
        'feedback': feedback
        })
    return {'last_result': last_result.content}

def parser_node(state: ClassifyState) -> ClassifyState:
    last_result = state['last_result']
    parser = _built_parser()
    try:
        parsed_result = parser.invoke(last_result)
        return {'parsed_result':parsed_result.model_dump()}
    
    except OutputParserException as e:
        return {'parsed_result':None, 'last_excep':str(e)}

def retry_node(state: ClassifyState):

    feedback = state['feedback']
    last_excep = state['last_excep']
    last_result = state['last_result']
    structured_llm = _built_retry_chain(temprature=0.3)
    parser = _built_parser()

    last_result = structured_llm.invoke({
        'instructions': parser.get_format_instructions(),
        'category_rules': CATEGORY_RULES,
        'urgency_rules': URGENCY_RULES,
        'feedback': feedback,
        'bad_output': last_result,
        'error': last_excep,
        })

    return {'retries': state.get('retries', 0) + 1, 'last_result':last_result.content}


def parser_conditional_node(state: ClassifyState) -> str:
    if state.get('parsed_result', None):
        return 'end'
    if state.get('retries', 0) < MAX_RETRIES:
        return 'retry'
    return 'end'

@lru_cache(maxsize=1)
def build_graph():
    g = StateGraph(ClassifyState)
    g.add_node('classify_node', classify_node)
    g.add_node('parser_node', parser_node)
    g.add_node('retry_node', retry_node)
    g.add_edge(START, 'classify_node')
    g.add_edge('classify_node', 'parser_node')
    g.add_edge('retry_node', 'parser_node')
    g.add_conditional_edges('parser_node', parser_conditional_node, {'retry': 'retry_node', 'end': END})
    return g.compile()

def get_classification(feedback: str) -> dict | None:
    final = build_graph().invoke({'feedback': feedback, 'retries':0, 'last_result':None, 'parsed_result':None, 'last_excep':None})

    return final['parsed_result']

