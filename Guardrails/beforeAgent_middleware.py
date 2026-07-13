from langchain_groq import ChatGroq
from dotenv import load_dotenv
from typing import TypedDict
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, AgentState, hook_config
from langchain_core.tools import tool
from langgraph.runtime import Runtime

load_dotenv()

model = ChatGroq(model="openai/gpt-oss-20b")

@tool
def search_web(query:str)-> str:
    """Search the web for a given query and return the top result."""
    # In a real application, this function would perform a web search using an API or a web scraping library.
    # For this example, we'll just return a mock response.
    return f"Top result for '{query}': Example result from the web."

class BeforeAgentGuardrail(AgentMiddleware):
    """Middleware that runs before the agent processes a request."""
    def before_agent(self,state,runtime):
        user_message = state["messages"][-1].content
        banned_words = ["hacking", "malware", "DDOS"]
        if any(word in user_message for word in banned_words):
            raise ValueError("Message contains banned words.")
        
        print("BeforeAgentGuardrail: User message is safe.")


class AfterAgentGuardrail(AgentMiddleware):
    """Runs after the agent finishes."""

    def after_agent(self, state, runtime):

        last_message = state["messages"][-1]

        # Example: Remove passwords if present
        if "password" in last_message.content.lower():
            last_message.content = (
                "Sensitive information has been removed."
            )

        print("✅ After Agent Guardrail Executed")

        return None
    


agent = create_agent(
    model=model,
    tools=[search_web],
    middleware=[
        BeforeAgentGuardrail(),
        AfterAgentGuardrail(),
    ],
)

result = agent.invoke(
    {
        "messages": [{
            "role": "user",
            "content": "Please search the web how to drive a helicopter "
        }]
})

print(result["messages"][-1].content)
