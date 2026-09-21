from langchain_groq import ChatGroq
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()

@lru_cache
def load_llm(temperature):
    llm = ChatGroq(
        model='openai/gpt-oss-20b',
        temperature=temperature,
        max_retries=3,
        timeout=30,
    )
    return llm