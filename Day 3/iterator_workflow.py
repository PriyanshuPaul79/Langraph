from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from typing import TypedDict, Literal

load_dotenv()

model = ChatGroq(model="llama-3.1-8b-instant")

class LinkedinPost(TypedDict):
    content: str
    topic: str
    evaluation: Literal["approved","try_again"]
    feedback: str
    iteration: int
    max_iteration: int

class EvaluationResult(TypedDict):
    evaluation: Literal["approved","try_again"]
    feedback: str


graph = StateGraph(LinkedinPost)

def create_post(state:LinkedinPost):
    topic = state["topic"]
    prompt = f'you are a professional content creator your task is to take the topic {topic} write a detailed linkedin post about this topic.'
    result = model.invoke(prompt).content
    return {"content":result}

def evaluation(state:LinkedinPost):
    content = state["content"]
    topic = state["topic"]
    prompt = f'''you are a professional evaluator whose task is to check weather the content is upto to point ot not according to the given topic here is the {topic} and here is the {content}.
    judge on the baasis of originality, humor, virality potential and format.
    Automatically reject it if exceed 280 character or if very generic. 
    respond only in structured format:
    evaluation: "approved" or "try_again"
    feedback: Give a feedback under 20 words so that onn the next try model can improve the quality mention the strenghts and weaknesses. 
    '''

    structured_response = model.with_structured_output(EvaluationResult).invoke(prompt)
    return {
        "evaluation": structured_response       ["evaluation"], 
        "feedback": structured_response["feedback"]
        }


def optimizer(state:LinkedinPost):
    topic = state["topic"]
    feedback = state["feedback"]
    content = state["content"]
    current_iteration = state["iteration"] + 1
    prompt = f'''you are a professional content creator your task is to optimize the content according to the feedback provided by the evaluator here is the topic {topic} and here is the content {content} and here is the feedback {feedback} optimize the content according to the feedback and make it more engaging, original and upto the point.'''
    result = model.invoke(prompt).content 
    return {"content":result, "iteration":current_iteration}


def route_evaluation(state:LinkedinPost):
    if state["evaluation"] == 'approved' or state["iteration"]>= state["max_iteration"]:
        return "approved"
    else :
        return "try_again"


graph.add_node("create_post",create_post)
graph.add_node("evaluation",evaluation)
graph.add_node("optimizer",optimizer)

graph.add_edge(START,"create_post")
graph.add_edge("create_post","evaluation")  
graph.add_conditional_edges("evaluation",route_evaluation,{'approved':END, 'try_again':"optimizer"})
graph.add_edge("optimizer","evaluation")


workflow = graph.compile()

inital_state = {'topic':"Jupyter notebook's role in ML development.",
                'iteration':1,
                'max_iteration':3,
                'content':'',
                'feedback':'',
                'evaluation':'',
                }

output = workflow.invoke(inital_state)
print(output)
  

