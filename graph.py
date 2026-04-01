from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
from langsmith.wrappers import wrap_openai
from openai import OpenAI
from pydantic import BaseModel

load_dotenv()
client = wrap_openai(OpenAI())

# Schema
class DetectCallResponse(BaseModel):
    is_question_ai: bool

class CodingAiResponse(BaseModel):
    answer: str

class SimpleAiResponse(BaseModel):
    answer: str

class State(TypedDict):
    user_message: str
    ai_message: str
    is_coding_question: bool


def detect_query(state: State):
    user_message = state.get("user_message")
    SYSTEM_PROMPT = """
    You are an AI assistant. Your job is to find out whether the user query is related to a coding question or not.
    Return the response in specified JSON boolean only.
    """
    result = client.beta.chat.completions.parse(  
        model="gpt-4o-mini",
        response_format=DetectCallResponse,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message}
        ]
    )
    state["is_coding_question"] = result.choices[0].message.parsed.is_question_ai
    return state


def solve_coding_question(state: State):
    user_message = state.get("user_message")
    SYSTEM_PROMPT = """
    You are an AI assistant. Your job is to resolve the coding question the user is facing.
    """
    result = client.beta.chat.completions.parse( 
        model="gpt-4o-mini", 
        response_format=CodingAiResponse,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message}
        ]
    )
    state["ai_message"] = result.choices[0].message.parsed.answer
    return state


def solve_simple_question(state: State):
    user_message = state.get("user_message")
    SYSTEM_PROMPT = """
    You are an AI assistant. Your job is to resolve the simple question the user is asking.
    """
    result = client.beta.chat.completions.parse( 
        model="gpt-4o-mini",
        response_format=SimpleAiResponse,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message}
        ]
    )
    state["ai_message"] = result.choices[0].message.parsed.answer
    return state


def add_route(state: State):
    if state.get("is_coding_question"):
        return "solve_coding_question"
    else:
        return "solve_simple_question"


graph_builder = StateGraph(State)
graph_builder.add_node("detect_query", detect_query)
graph_builder.add_node("solve_coding_question", solve_coding_question)
graph_builder.add_node("solve_simple_question", solve_simple_question)

graph_builder.add_edge(START, "detect_query")
graph_builder.add_conditional_edges("detect_query", add_route)
graph_builder.add_edge("solve_coding_question", END)
graph_builder.add_edge("solve_simple_question", END)

graph = graph_builder.compile()


def call_graph():
    state = {
        "user_message": "Hey there how are you",
        "ai_message": " ",
        "is_coding_question": False
    }
    result = graph.invoke(state)
    print("Final Result", result)

call_graph()