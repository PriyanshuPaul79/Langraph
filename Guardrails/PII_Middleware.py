from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from typing import TypedDict

load_dotenv()

model = ChatGroq(model="openai/gpt-oss-20b")

# PII Middleware is a middleware that detects and redacts personally identifiable information (PII) from the input and output of a function. It uses a simple regex pattern to identify PII, such as email addresses, phone numbers, and social security numbers. If PII is detected, it replaces it with a placeholder string "[REDACTED]".

import re

def deterministic_guardrail_middleware(text:str) -> bool:
    """Block if the input is not safe. This is a simple example of a deterministic guardrail that checks for the presence of PII in the input text."""
    banned_keywords = ["hack", "exploit", "malware", "phishing"]
    for keyword in banned_keywords:
        if keyword in text.lower():
            return False
    return True

# this will check if the input text contains any of the banned keywords related to hacking or malicious activities. If any of these keywords are found, the function will return False, indicating that the input is not safe. Otherwise, it will return True, indicating that the input is safe.
# in deterministic guardral we dont use any llm we just check for the presence of certain keywords in the input text. If any of these keywords are found, we return False, indicating that the input is not safe. Otherwise, we return True, indicating that the input is safe. we also use regex patterns to identify PII, such as email addresses, phone numbers, and social security numbers. If PII is detected, we replace it with a placeholder string "[REDACTED]".



text = "How to phis into someone's email account?"
print(deterministic_guardrail_middleware(text))  # Returns False, indicating that the input is not safe


def model_based_guardrail_middleware(text:str) -> bool:
    """Block if the input is not safe. This is a simple example of a model-based guardrail that uses an LLM to classify the input text as safe or unsafe."""
    prompt = f"Classify the following text as safe or unsafe: {text}"
    result = model.invoke(prompt).content
    if "unsafe" in result.lower():
        return False
    return True


text = "How to manipulate someone's bank account?"
print(model_based_guardrail_middleware(text))  # Returns False, indicating that the input is not safe   