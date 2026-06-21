from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_groq import ChatGroq
from typing import TypedDict, Annotated
from dotenv import load_dotenv

from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage

load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile"
    )

@tool
def calculator(num1: int, num2: int, operation: str):
    """Perform basic arithmetic operations."""
    if operation == "add":
        return num1 + num2
    elif operation == "subtract":
        return num1 - num2
    elif operation == "multiply":
        return num1 * num2
    elif operation == "divide":
        return num1 / num2 if num2 != 0 else "Division by zero"
    return "Invalid operation"

search_tool = DuckDuckGoSearchResults()

tools = [calculator, search_tool]

llm_with_tools = llm.bind_tools(tools)


class ChatState(TypedDict):
    messages: Annotated[list, add_messages]


def chatbot(state: ChatState):
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


graph = StateGraph(ChatState)

graph.add_node("chatbot", chatbot)
graph.add_node("tools", ToolNode(tools))

graph.add_edge(START, "chatbot")

graph.add_conditional_edges(
    "chatbot",
    tools_condition
)

graph.add_edge("tools", "chatbot")

workflow = graph.compile()

result = workflow.invoke(
    {
        "messages": [
            HumanMessage(
                content="What is the temperature of Stockholm right now and what is 500 multiplied by 3?"
            )
        ]
    }
)

print(result["messages"][-1].content)