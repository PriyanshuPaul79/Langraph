from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from typing import TypedDict

load_dotenv()

model = ChatGroq(model="llama-3.1-8b-instant")


class RaceState(TypedDict):
    race_name: str
    driver_name: str
    driver_analysis: str
    track_analysis: str
    weather_analysis: str
    tire_analysis: str
    final_prediction: str


def driver_performance(state: RaceState):
    name = state["driver_name"]
    venue = state["race_name"]

    prompt = (
        f"You are given the name of an F1 driver ({name}). "
        f"Analyze his last 2 F1 seasons and place extra emphasis on performances at {venue}."
    )

    result = model.invoke(prompt).content
    return {"driver_analysis": result}


def track_analysis(state: RaceState):
    venue = state["race_name"]

    prompt = (
        f"You are given an F1 venue ({venue}). "
        f"Analyze the track characteristics and provide insights about recent races there."
    )

    result = model.invoke(prompt).content
    return {"track_analysis": result}


def weather_analysis(state: RaceState):
    venue = state["race_name"]

    prompt = (
        f"You are given an F1 venue ({venue}). "
        f"Analyze typical weather conditions and how they affect racing."
    )

    result = model.invoke(prompt).content
    return {"weather_analysis": result}


def tire_analysis(state: RaceState):
    venue = state["race_name"]
    weather = state["weather_analysis"]

    prompt = (
        f"Venue: {venue}\n"
        f"Weather Analysis: {weather}\n\n"
        f"Suggest an optimal tire strategy."
    )

    result = model.invoke(prompt).content
    return {"tire_analysis": result}


# JOIN NODE
def gather(state: RaceState):
    return {}


def final_prediction(state: RaceState):
    prompt = f"""
    Venue: {state['race_name']}

    Driver Analysis:
    {state['driver_analysis']}

    Track Analysis:
    {state['track_analysis']}

    Weather Analysis:
    {state['weather_analysis']}

    Tire Analysis:
    {state['tire_analysis']}

    Predict the driver's finishing position and explain why.
    """

    result = model.invoke(prompt).content
    return {"final_prediction": result}


graph = StateGraph(RaceState)

graph.add_node("driver_performance", driver_performance)
graph.add_node("track_analysis", track_analysis)
graph.add_node("weather_analysis", weather_analysis)
graph.add_node("tire_analysis", tire_analysis)
graph.add_node("gather", gather)
graph.add_node("final_prediction", final_prediction)

# Parallel branches
graph.add_edge(START, "driver_performance")
graph.add_edge(START, "track_analysis")
graph.add_edge(START, "weather_analysis")

# Weather -> Tire
graph.add_edge("weather_analysis", "tire_analysis")

# Fan-in
graph.add_edge("driver_performance", "gather")
graph.add_edge("track_analysis", "gather")
graph.add_edge("tire_analysis", "gather")

# Final prediction
graph.add_edge("gather", "final_prediction")
graph.add_edge("final_prediction", END)

workflow = graph.compile()

initial_state = {
    "race_name": "Monaco Grand Prix",
    "driver_name": "Max Verstappen"
}

prediction = workflow.invoke(initial_state)

print("\n=== FINAL PREDICTION ===\n")
print(prediction["final_prediction"])