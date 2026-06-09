from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from typing import TypedDict


load_dotenv()

model = ChatGroq(model = "llama-3.1-8b-instant")

class RaceState(TypedDict):
    race_name: str
    driver_name: str
    driver_analysis: dict
    track_analysis:dict
    weather_analysis: dict
    tire_analysis: dict
    final_prediction: str


def driver_performance(state:RaceState):
    name = state["driver_name"]
    venue = state["race_name"]
    prompt = f"you are give the {name} of a f1 driver your task is to analyse his last 2 f1 season performances also give more emphasis in this {venue}"
    result = model.invoke(prompt).content
    return {"driver_analysis":result}


def track_analysis(state:RaceState):
    venue = state["race_name"]
    prompt = f"you are given the name of a f1 track your task is to analyse the track and give insights about the track and also give insights about the last 5 races held there here is the {venue} track"
    result = model.invoke(prompt).content
    return {"track_analysis":result}

def weather_analysis(state:RaceState):
    venue = state["race_name"]
    prompt = f"you are given the name of a f1 track your task is to analyse the weather conditions of the track and also give insights about the last 5 races held there here is the {venue} track"
    result = model.invoke(prompt).content
    return {"weather_analysis":result}


def tire_analysis(state:RaceState):
    venue = state["race_name"]
    weather = state["weather_analysis"]
    prompt

graph = StateGraph(RaceState)
graph.add_node("driver_performance",driver_performance)
graph.add_node("track_analysis",track_analysis)
graph.add_node("weather_analysis",weather_analysis)
graph.add_node("tire_analysis",tire_analysis)
