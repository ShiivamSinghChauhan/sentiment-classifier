from langchain_groq import ChatGroq
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()

@lru_cache(maxsize=1)
def load_llm():
    llm = ChatGroq(
        model='openai/gpt-oss-20b',
        temperature=0,
        max_retries=3,
        timeout=30,
    )
    return llm