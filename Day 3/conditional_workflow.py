from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from typing import TypedDict, Literal

load_dotenv()

model = ChatGroq(model="llama-3.1-8b-instant")

class Roots(TypedDict):
    result : str
    dis : float
    equation: str
    a : int
    b : int
    c : int


def show_eqn(state:Roots):
    equation = f'{state["a"]}x² + {state["b"]}x + {state["c"]} = 0'    
    return {"equation":equation}

def cal_disc(state:Roots):
    discriminant = state["b"]**2 - (4* state["a"] * state["c"]) 
    return {"dis":discriminant}


def no_real_roots(state:Roots):
        return {"result":"No real roots"}
    
def real_roots(state:Roots):
        root1 = (-state["b"] + state["dis"]**0.5) / (2*state["a"])
        root2 = (-state["b"] - state["dis"]**0.5) / (2*state["a"])
        return {"result":f"Two real roots: {root1} and {root2}"}
    
def repeated_roots(state:Roots):
        root = -state["b"] / (2*state["a"])
        return {"result":f"One repeated root: {root}"}

def root_condition(state:Roots) -> Literal['real_roots', 'no_real_roots', 'repeated_roots']:
    if state["dis"] > 0:
        return "real_roots"
    elif state["dis"] == 0:
        return "repeated_roots"
    else:
        return "no_real_roots"


graph = StateGraph(Roots)
graph.add_node("show_eqn",show_eqn)
graph.add_node("cal_disc",cal_disc)
graph.add_node("no_real_roots",no_real_roots)
graph.add_node("real_roots",real_roots)
graph.add_node("repeated_roots",repeated_roots)

graph.add_edge(START,"show_eqn")
graph.add_edge("show_eqn","cal_disc")
graph.add_conditional_edges("cal_disc",root_condition)
graph.add_edge("real_roots", END)
graph.add_edge("no_real_roots", END)
graph.add_edge("repeated_roots", END)

workflow = graph.compile()

initial_state = {'a':1,
                 'b':4,
                 'c':4}

res = workflow.invoke(initial_state)
print(res)