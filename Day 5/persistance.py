from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()


class jokeState(TypedDict):
    topic : str
    joke : str
    explanation : str

model = ChatGroq(model="llama-3.1-8b-instant")
graph = StateGraph(jokeState)

def create_joke(state:jokeState):
    topic = state['topic']
    prompt = f'you are given a topic your task is to create a funny joke about this topic here is the topic : {topic}'
    joke = model.invoke(prompt).content
    # state['joke'] = joke
    return {'joke':joke}


def explain(state:jokeState):
    joke = state["joke"]
    prompt = f'you task is to explain this joke to the user in easy language. here is the joke : {joke}'
    meaning = model.invoke(prompt).content
    return {'explanation':meaning}


graph.add_node("create_joke",create_joke)
graph.add_node("give_explanation", explain)

graph.add_edge(START,"create_joke")
graph.add_edge("create_joke","give_explanation")
graph.add_edge("give_explanation",END)


graph.compile
