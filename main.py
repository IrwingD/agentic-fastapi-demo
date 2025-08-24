from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from typing import TypedDict, Optional
import logging

# --- Setup logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Initialize LLM ---
llm = ChatGoogleGenerativeAI(model='gemini-2.0-flash', temperature=0)

# --- FastAPI app ---
app = FastAPI()

# --- Enable CORS for Lovable ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://*.lovable.dev"],  # allow Lovable sandbox domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Request body model ---
class UserRequest(BaseModel):
    user_input: str

# --- Define agent "state" ---
class AgentState(TypedDict, total=False):
    user_input: str
    intent: Optional[str]
    response: Optional[str]

# --- Define nodes ---
def classify_intent(state: AgentState):
    query = state["user_input"]
    response = llm.invoke([
        {"role": "system", "content": "Classify if user wants 'weather' or 'chat'. Only respond with one word."},
        {"role": "user", "content": query}
    ])
    state["intent"] = response.content.strip().lower()
    return state

def fetch_weather(state: AgentState):
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
def agent_endpoint(req: UserRequest):
    try:
        final_state = agent_app.invoke({"user_input": req.user_input})
        return {"response": final_state["response"]}
    except Exception as e:
        logger.exception("Error while processing request")
        return {"error": str(e)}
