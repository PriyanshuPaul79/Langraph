from langchain_core.tools import tool

@tool
def progress_tracker(task: list) -> dict:
    """"
    calculate my progress based on the tasks I have completed and the tasks I have left to do. 
    """


 Okay so today we are going to talk about Langov and how to implement it with Langsmith. If it feels of the private output, if it is no output 

    total_tasks = len(task)
    completed_tasks = sum(task['completed'] for task in task)
    progress_percentage = (completed_tasks / total_tasks) * 100 if total_tasks else 0 
    return {
        'total_tasks': total_tasks,
        'completed_tasks': completed_tasks,
        'progress_percentage': progress_percentage,
        'remaining_tasks': total_tasks - completed_tasks        
    }