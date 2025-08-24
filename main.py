from fastapi import FastAPI
from pydantic import BaseModel
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(model='gemini-2.0-flash', temperature=0)

app = FastAPI()

class Request(BaseModel):
    user_input: str

# --- Define agent "state" ---
from typing import TypedDict, Optional

class AgentState(TypedDict, total=False):
    user_input: str
    intent: Optional[str]
    response: Optional[str]

# --- Define nodes ---
def classify_intent(state: AgentState):
    """Classify if query is about weather or normal chat."""
    query = state["user_input"]
    response = llm.invoke([
        {"role": "system", "content": "Classify if user wants 'weather' or 'chat'. Only respond with one word."},
        {"role": "user", "content": query}
    ])
    state["intent"] = response.content.strip().lower()
    return state

def fetch_weather(state: AgentState):
    """Toy weather lookup"""
    query = state["user_input"]
    response = llm.invoke([
        {"role": "system", "content": "Extract the city name from the query."},
        {"role": "user", "content": query}
    ])
    city = response.content.strip()
    weather_data = {
        "paris": "Sunny, 25°C",
        "london": "Rainy, 18°C",
        "new york": "Cloudy, 22°C"
    }
    state["response"] = f"The weather in {city} is: {weather_data.get(city.lower(), 'Unknown')}"
    return state

def normal_chat(state: AgentState):
    query = state["user_input"]
    response = llm.invoke([
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": query}
    ])
    state["response"] = response.content.strip()
    return state

# --- Build LangGraph ---
workflow = StateGraph(AgentState)

workflow.add_node("classify", classify_intent)
workflow.add_node("weather", fetch_weather)
workflow.add_node("chat", normal_chat)

# Edges
workflow.set_entry_point("classify")
workflow.add_conditional_edges(
    "classify",
    lambda state: "weather" if "weather" in state.get("intent", "") else "chat",
    {"weather": "weather", "chat": "chat"}
)
workflow.add_edge("weather", END)
workflow.add_edge("chat", END)

agent_app = workflow.compile()

# --- FastAPI endpoint ---
@app.post("/agent")
def agent_endpoint(req: Request):
    final_state = agent_app.invoke({"user_input": req.user_input})
    return {"response": final_state["response"]}
