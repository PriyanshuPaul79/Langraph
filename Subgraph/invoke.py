from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_groq import ChatGroq
from typing import TypedDict, Annotated
from dotenv import load_dotenv
from pydantic import BaseModel
load_dotenv()

class questionState(BaseModel):
    algo : Annotated[str, "Define the algorithm used to solve the question"]
    topic: Annotated[str,"is question related to coding or science"]
    code: str
    question: str
    answer: str

model = ChatGroq(model="openai/gpt-oss-20b")


#helper functions
def evaluation(State:questionState):
    print("Evalution done.")

def result(State:questionState):
    question = State["question"]
    topic = State["topic"]
    prompt = f"""
                you are a professional {topic} with 10+ years of experience answer me this {question} in short.
"""
    prompt2 = f""" if the question is related to coding then provide the code in python and if it is related to science then provide the answer in short and simple words."""
    answer = model.invoke(prompt)
    return {"answer":answer}

def science(State:questionState):
    print('this is a science question')

def code(State:questionState):
    print("this is a coding question")

def route(State: questionState) -> str:
    if State["topic"] == "science":
        return science
    return code

myGraph = StateGraph(questionState)

myGraph.add_node("eval",evaluation)
myGraph.add_node("result",result)
myGraph.add_node("science_node",science)
myGraph.add_node("code_node",code)


myGraph.add_edge(START,"eval")
myGraph.add_conditional_edges(
    "evals",
    route,
    {
        "science":"science_node",
        "code":"code_node"
    }
)

myGraph.add_edge("science_node","result")
myGraph.add_edge("code_node","result")
myGraph.add_edge("result",END)

output = myGraph.compile()
output.invoke({
    "algo":"binary search",
    "topic":"science",
    "code":"def binary_search(arr, target):\n    left, right =
