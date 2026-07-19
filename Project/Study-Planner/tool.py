from langchain_core.tools import tool
from memory import load_schedule, update_task_completion

@tool
def progress_tracker(task: list) -> dict:
    """"
    calculate my progress based on the tasks I have completed and the tasks I have left to do. 
    """

    total_tasks = len(task)
    completed_tasks = sum(task['completed'] for task in task)
    progress_percentage = (completed_tasks / total_tasks) * 100 if total_tasks else 0 
    return {
        'total_tasks': total_tasks,
        'completed_tasks': completed_tasks,
        'progress_percentage': progress_percentage,
        'remaining_tasks': total_tasks - completed_tasks        
    }


@tool
def db_progress_tracker(user_name: str, exam_history_id: int = None) -> dict:
    """
    Fetch the user's study schedule from the Supabase database and calculate their current progress.
    Returns details on total, completed, remaining tasks, and overall progress percentage.
    """
    tasks = load_schedule(user_name, exam_history_id)
    total_tasks = len(tasks)
    completed_tasks = sum(1 for t in tasks if t.get("is_completed"))
    progress_percentage = (completed_tasks / total_tasks) * 100 if total_tasks else 0
    return {
        'total_tasks': total_tasks,
        'completed_tasks': completed_tasks,
        'progress_percentage': progress_percentage,
        'remaining_tasks': total_tasks - completed_tasks
    }


@tool
def mark_task_complete(task_id: int, is_completed: bool = True) -> str:
    """
    Update a specific schedule task's completion status in Supabase database.
    """
    success = update_task_completion(task_id, is_completed)
    if success:
        return f"Successfully updated task {task_id} to completed={is_completed}"
    else:
        return f"Failed to update task {task_id}"