from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from typing import TypedDict
from dotenv import load_dotenv

from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community_tools import GoogleSearchResults, WikipediaSummary
from langchain_core.tools import Tool

import requests
import random


load_dotenv()

llm = ChatGroq(model="llama-3.1-8b-instant", streaming=True)

@tool
def calculator(num1:int, num2:int, operation:str):
    """ A simple calculator tool that performs basic arithmetic operations."""
    if operation == "add":
        return num1 + num2
    elif operation == "subtract":
        return num1 - num2
    elif operation == "multiply":
        return num1 * num2
    elif operation == "divide":
        if num2 != 0:
            return num1 / num2
        else:
            return "Error: Division by zero"
    else:
        return "Error: Invalid operation. Please use 'add', 'subtract', 'multiply', or 'divide'."
    
search_tool = google_search_tool = ToolNode(
    tool=GoogleSearchResults(),
    name="google_search",
    description="Use this tool to search for information on the web. Input should be a search query string. The output will be the search results.",
    input_type=str,
    output_type=str,
    condition=tools_condition,
)

# make a list of all the tools present 
tools = [calculator, search_tool]
# make the llm aware that these tools are available for use. This allows the LLM to decide when to use a tool based on the input it receives.
llm_tools = llm.bind_tools(tools)


class chatState(TypedDict):
    user_input: str
    response: str

graph = StateGraph(chatState)

def llm_tools(state:chatState):
    user_input = state['user_input']
    prompt = f"You are a helpful assistant. You can use the available tools to answer the user's query. Here is the user's input: {user_input}"
    response = llm_tools.invoke(prompt).content
    return {'response': response}

tool_node = ToolNode(tools)
graph.add_node("chat_with_tools", llm_tools)