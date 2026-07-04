from langchain_groq import ChatGroq
from dotenv import load_dotenv
from typing import TypedDict
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.tools import tool
from langgraph.types import Command
from sqlalchemy import false


load_dotenv()


model = ChatGroq(model="openai/gpt-oss-20b")

@tool
def search_web(query:str)-> str:
    """Search the web for a given query and return the top result."""
    # In a real application, this function would perform a web search using an API or a web scraping library.
    # For this example, we'll just return a mock response.
    return f"Top result for '{query}': Example result from the web."


@tool
def send_mail(to:str, subject:str, body:str)->str:
    """Send an email to the specified recipient with the given subject and body."""
    # In a real application, this function would send an email using an email service or SMTP server.
    # For this example, we'll just return a mock response.
    return f"Email sent to {to} with subject '{subject}' and body '{body}'."


agent = create_agent(
    model = model,
    tools=[search_web, send_mail],
    middleware=[HumanInTheLoopMiddleware(
        interrupt_on = {"send_mail": True,
                        "search_web": False}
    )], checkpointer=InMemorySaver()
)

config = {"configurable": {"thread_id": "1"}}

result = agent.invoke(
    {
        "messages": [{
            "role": "user",
            "content": "Please send an email to ravi gupta with subject 'Meeting Reminder' and body 'Don't forget about our meeting tomorrow at 10 AM.'"
        }]
    },
    config=config
)

print(result)


result = agent.invoke(
    Command(
        resume={
            "decisions": [
                {
                    "type": "approve"
                }
            ]
        }
    ),
    config=config,
)

print(result)
