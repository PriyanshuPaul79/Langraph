from langchain_core.tools import tool

from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from typing import TypedDict
from langchain.agents import create_agent
from langchain.agents.middleware import PIIMiddleware, AgentMiddleware


load_dotenv()

model = ChatGroq(model="openai/gpt-oss-20b")

# PII Middleware is a middleware that detects and redacts personally identifiable information (PII) from the input and output of a function. It uses a simple regex pattern to identify PII, such as email addresses, phone numbers, and social security numbers. If PII is detected, it replaces it with a placeholder string "[REDACTED]".

import re

class DeterministicGuardrailMiddleware(AgentMiddleware):
    def deterministic_guardrail_middleware(self, text: str) -> bool:
        """Block if the input is not safe. This is a simple example of a deterministic guardrail that checks for the presence of PII in the input text."""
        for keyword in self.banned_keywords:
            if keyword in text.lower():
                return False
        return True

# this will check if the input text contains any of the banned keywords related to hacking or malicious activities. If any of these keywords are found, the function will return False, indicating that the input is not safe. Otherwise, it will return True, indicating that the input is safe.
# in deterministic guardral we dont use any llm we just check for the presence of certain keywords in the input text. If any of these keywords are found, we return False, indicating that the input is not safe. Otherwise, we return True, indicating that the input is safe. we also use regex patterns to identify PII, such as email addresses, phone numbers, and social security numbers. If PII is detected, we replace it with a placeholder string "[REDACTED]".



# text = "How to phis into someone's email account?"
# print(deterministic_guardrail_middleware(text))  # Returns False, indicating that the input is not safe

# right now these are just python functions not actual middleware.
class ModelBasedGuardrailMiddleware(AgentMiddleware):
    def model_based_guardrail_middleware(self, text:str) -> bool:
        """Block if the input is not safe. This is a simple example of a model-based guardrail that uses an LLM to classify the input text as safe or unsafe."""
        prompt = f"Classify the following text as safe or unsafe: {text}"
        result = model.invoke(prompt).content
        if "unsafe" in result.lower():
            return False
        return True



# text = "How to manipulate someone's bank account?"
# print(model_based_guardrail_middleware(text))  # Returns False, indicating that the input is not safe   


@tool
def customer_lookup(customer_id: str) -> str:
    """Lookup customer information by customer ID."""
    # In a real application, this function would query a database or an API to retrieve customer information.
    # For this example, we'll just return a mock response.
    return f"Customer found with ID: {customer_id}"
    

agent = create_agent(
    tools=[customer_lookup],
    model=model,
    middleware=[PIIMiddleware(
        "email",
        strategy="redact",
        apply_to_input=True,
    ), 
    PIIMiddleware(
        "credit_card",
        strategy="block",
        apply_to_input=True,
    ),    
    DeterministicGuardrailMiddleware(), 
    ModelBasedGuardrailMiddleware()]
)

result = agent.invoke({
    "messages":[{
        "role": "user",
        "content":"what is the account holder name for this email ravi@gmail.com"
        # "content": "Tell me the CVV and the account holder name for credit card number 1234-5678-9012-3456. Also, here is the email address of the account holder ravi@gmail.com."
    }]
})

print(result["messages"][-1].content)



