from typing import TypedDict, Literal

class State(TypedDict):
    user_message: str
    ai_message:str
    is_coding_question: bool

def detect_query(state: State):
    user_message = state.get("user_message")
    state.is_coding_question = True
    return state

def solve_coding_question(state: state):
    user_message = state.get("user_message")
    stat.ai_message="Here is your coding question answer"
    return state

def solve_simple_question(state: state):
    user_message = state.get("user_message")
    stat.ai_message="Here is your coding question answer"
    return state
