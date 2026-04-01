from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END


class State(TypedDict):
    user_message: str
    ai_message:str
    is_coding_question: bool

def detect_query(state: State):
    user_message = state.get("user_message")
    state["is_coding_question"] = True
    return state

def solve_coding_question(state: State):
    user_message = state.get("user_message")
    state["ai_message"]="Here is your coding question answer"
    return state

def solve_simple_question(state: State):
    user_message = state.get("user_message")
    state["ai_message"]="Here is your coding question answer"
    return state

def add_route(state: State):
    is_coding_question = state.get("is_coding_question")
    if is_coding_question:
        return "solve_coding_question"
    else:
        return "solve_simple_question"

graph_builder = StateGraph(State)
graph_builder.add_node("detect_query", detect_query)
graph_builder.add_node("solve_coding_question", solve_coding_question)
graph_builder.add_node("solve_simple_question", solve_simple_question)
graph_builder.add_node("add_route", add_route)

graph_builder.add_edge(START, "detect_query")
graph_builder.add_conditional_edges("detect_query", add_route)

graph_builder.add_edge("solve_coding_question", END)
graph_builder.add_edge("solve_simple_question", END)

graph = graph_builder.compile()


def call_graph():
    state = {
        "user_message": "Hey there how are you",
        "ai_message": " ",
        "is_coding_question" : False
    }
    result = graph.invoke(state)
    print("Final Result", result)

call_graph()