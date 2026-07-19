from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres import PostgresSaver
from uuid import uuid4

from states import PlannerState
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from memory import load_history, save_goal, save_plan_text
from db import pool

load_dotenv()

model = ChatGroq(model="openai/gpt-oss-20b")
planner = StateGraph(PlannerState)


def collectInputFunction(State: PlannerState):
    name = input("Your name: ")
    history, _ = load_history(name) if name else ([], {})
    if history:
        print("\nPast exams:")
        for h in history:
            plan_note = " (plan saved)" if h.get("plan_text") else ""
            print(f"  - {h['exam']} ({h['exam_date']}): {h['target_goal']}, {h['no_of_hours']}hrs/day{plan_note}")
        print()

    exam = input("Exam name: ")
    exam_date = input("Exam date: ")
    target_goal = input("Target goal: ")
    no_of_hours = int(input("Hours per day: "))

    return {
        "user_profile": {"name": name},
        "current_goal": {
            "exam": exam,
            "exam_date": exam_date,
            "target_goal": target_goal,
            "no_of_hours": no_of_hours,
        },
        "approved": False,
        "feedback": "",
    }


def genPlanFunction(State: PlannerState):
    goal = State["current_goal"]
    user_name = State["user_profile"].get("name")
    feedback = State.get("feedback", "")

    history, _ = load_history(user_name) if user_name else ([], {})

    history_prompt = ""
    if history:
        lines = ["\nPast exam history (consider this for personalization):"]
        for h in history[:5]:
            lines.append(
                f"- {h['exam']} (date: {h['exam_date']}, target: {h['target_goal']}, {h['no_of_hours']}hrs/day)"
            )
        history_prompt = "\n".join(lines)

    feedback_prompt = ""
    if feedback:
        feedback_prompt = (
            f"\nUser feedback on the plan: {feedback}\nAdjust the plan based on this feedback."
        )

    prompt = f"""Create a weekly study plan.

Exam: {goal['exam']}
Target: {goal['target_goal']}
Study Hours: {goal['no_of_hours']}
Exam Date: {goal['exam_date']}
{history_prompt}
{feedback_prompt}

Return the answer in markdown.

Include:
- Weekly schedule
- Daily tasks
- Revision strategy
- Mock test schedule
"""
    plan = model.invoke(prompt)

    if user_name:
        history_id = save_goal(user_name, goal, State["user_profile"])
        save_plan_text(history_id, user_name, plan.content)

    return {"study_plan": plan.content, "approved": True, "feedback": ""}


planner.add_node("Input", collectInputFunction)
planner.add_node("Generate Plan", genPlanFunction)
planner.add_edge(START, "Input")
planner.add_edge("Input", "Generate Plan")
planner.add_edge("Generate Plan", END)

# ponytail: could share db.py pool instead of separate conn string, but from_conn_string is cleanest
with PostgresSaver.from_conn_string(pool.conninfo) as checkpointer:
    checkpointer.setup()
    graph = planner.compile(checkpointer=checkpointer, interrupt_before=["Generate Plan"])

    config = {"configurable": {"thread_id": str(uuid4())}}

    print(graph.get_graph().draw_mermaid())

    graph.invoke(
        {
            "message": [],
            "user_profile": {},
            "current_goal": {},
            "study_plan": "",
            "approved": False,
            "feedback": "",
        },
        config=config,
    )

    state = graph.get_state(config)
    goal = state.values["current_goal"]
    print(
        f"\nGoal: {goal.get('exam')} — Target: {goal.get('target_goal')} ({goal.get('exam_date')}), {goal.get('no_of_hours')}hrs/day"
    )
    ok = input("Approve this goal? (y/n): ").strip().lower() == "y"

    if ok:
        graph.update_state(config, {"approved": True})
        graph.invoke(None, config=config)
        final = graph.get_state(config)
        print("\n" + final.values.get("study_plan", ""))
    else:
        feedback = input("What should be adjusted? ")
        graph.update_state(config, {"approved": False, "feedback": feedback})
        graph.invoke(None, config=config)
        final = graph.get_state(config)
        print("\n" + final.values.get("study_plan", ""))
