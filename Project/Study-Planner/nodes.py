from langgraph.graph import StateGraph, START, END
from states import PlannerState, User, StudyGoal
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from IPython.display import Image, display

load_dotenv()

model = ChatGroq(model = "openai/gpt-oss-20b") 
planner = StateGraph(PlannerState)


def collectInputFunction(State: PlannerState):
    exam = input("Exam name: ")
    exam_date = input("Exam date: ")
    target_goal = input("Target goal: ")
    no_of_hours = int(input("Hours per day: "))

    return {"current_goal": {"exam": exam, "exam_date": exam_date, "target_goal": target_goal, "no_of_hours": no_of_hours}}


def genPlanFunction(State: PlannerState):
    goal = State["current_goal"]
    prompt = f'''Create a weekly study plan.

Exam: {goal['exam']}
Target: {goal['target_goal']}
Study Hours: {goal['no_of_hours']}
Exam Date: {goal['exam_date']}

Return the answer in markdown.

Include:
- Weekly schedule
- Daily tasks
- Revision strategy
- Mock test schedule 

'''
    plan = model.invoke(prompt)
    return {"study_plan": plan.content}


planner.add_node("Input", collectInputFunction)
planner.add_node("Generate Plan", genPlanFunction)
planner.add_edge(START,"Input")
planner.add_edge("Input","Generate Plan")
planner.add_edge("Generate Plan",END)

initial_graph = planner.compile()

print(initial_graph.get_graph().draw_mermaid())

result = initial_graph.invoke({
    "message": [],
    "user_profile": {},
    "current_goal": {},
    "study_plan": ""
})

print(result['study_plan'])