from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from dotenv import load_dotenv
from langgraph.checkpoint.memory import InMemorySaver

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

checkpointer = InMemorySaver()
workflow = graph.compile(checkpointer=checkpointer)

config1 = {"configurable":{"thread_id":"1"}}
config2 = {"configurable":{"thread_id":"2"}}
result1 = workflow.invoke({"topic":"programming"}, config=config1)
result2 = workflow.invoke({"topic":"football"}, config=config2)
# print(result)

answer = list(workflow.get_state(config=config2))
# this will show all the checkpoints and there states and the final state will have the topic, joke and explanation all together this also show us the intermediate state as well as the enxt state and the checkpoint id 
print(answer)

history = list(workflow.get_state_history(config=config1))
# this will show us the history of the states that we have gone through in this workflow with the checkpoint id and the state at that checkpoint id and the timestamp of when that checkpoint was created this
print(history)