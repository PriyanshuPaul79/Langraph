from typing import TypedDict, Annotated



class StudyGoal(TypedDict):
    exam : str
    exam_date : str
    target_goal : str
    no_of_hours : int

class User(TypedDict):
    name: str
    preferred_study_time: str
    daily_available_hours: int
    weak_subjects: list[str]
    strong_subjects: list[str]
    revision_frequency: str

class PlannerState(TypedDict):
    message: list
    # this is for the message history 

    user_profile: User
    # long term memory 

    current_goal : StudyGoal
    study_plan : str

    approved : bool
    feedback : str


