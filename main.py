from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from typing import TypedDict, Optional
import logging
import os
import pandas as pd
import pinecone

# --- Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- API key ---
API_KEY = os.getenv("AGENT_API_KEY")
if not API_KEY:
    raise RuntimeError("Environment variable AGENT_API_KEY not set")

# --- LLM ---
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)

# --- FastAPI ---
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://1317128d-d404-412b-a6cc-3420c76dfb42.sandbox.lovable.dev"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Request model ---
class UserRequest(BaseModel):
    user_input: str

# --- Load CSVs ---
DATA_DIR = os.path.join(os.path.dirname(__file__), "Data")
salary_df = pd.read_csv(os.path.join(DATA_DIR, "salary.csv"))
leaves_df = pd.read_csv(os.path.join(DATA_DIR, "leaves.csv"))

# --- Pinecone setup for policies ---
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")  # e.g., "us-east1-gcp"
INDEX_NAME = "policy-index"

pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
if INDEX_NAME not in pinecone.list_indexes():
    raise RuntimeError(f"Pinecone index '{INDEX_NAME}' not found")
policy_index = pinecone.Index(INDEX_NAME)

# --- Agent state ---
class AgentState(TypedDict, total=False):
    user_input: str
    intent: Optional[str]
    response: Optional[str]

# --- Helper functions ---
def get_salary(name: str):
    row = salary_df[salary_df["name"].str.lower() == name.lower()]
    return f"{row.iloc[0]['salary']}" if not row.empty else "Employee not found"

def get_leaves(name: str):
    row = leaves_df[leaves_df["name"].str.lower() == name.lower()]
    if not row.empty:
        return f"Casual: {row.iloc[0]['casual_leave']}, Sick: {row.iloc[0]['sick_leave']}, Earned: {row.iloc[0]['earned_leave']}"
    return "Employee not found"

def query_policy(text: str):
    # Toy example: simple vector search
    results = policy_index.query(vector=llm.embed(text), top_k=1)
    if results.matches:
        return results.matches[0].metadata.get("text", "Policy not found")
    return "Policy not found"

# --- Nodes ---
def classify_intent(state: AgentState):
    query = state["user_input"]
    response = llm.invoke([
        {"role": "system", "content": "Classify if user wants 'salary', 'leave', 'policy', or 'chat'. Respond with a single word."},
        {"role": "user", "content": query}
    ])
    state["intent"] = response.content.strip().lower()
    return state

def salary_node(state: AgentState):
    name = state["user_input"]  # in real case, extract name using LLM
    state["response"] = get_salary(name)
    return state

def leave_node(state: AgentState):
    name = state["user_input"]
    state["response"] = get_leaves(name)
    return state

def policy_node(state: AgentState):
    state["response"] = query_policy(state["user_input"])
    return state

def chat_node(state: AgentState):
    response = llm.invoke([
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": state["user_input"]}
    ])
    state["response"] = response.content.strip()
    return state

# --- Build LangGraph ---
workflow = StateGraph(AgentState)
workflow.add_node("classify", classify_intent)
workflow.add_node("salary", salary_node)
workflow.add_node("leave", leave_node)
workflow.add_node("policy", policy_node)
workflow.add_node("chat", chat_node)

workflow.set_entry_point("classify")
workflow.add_conditional_edges(
    "classify",
    lambda state: state.get("intent", "chat"),
    {"salary": "salary", "leave": "leave", "policy": "policy", "chat": "chat"}
)
workflow.add_edge("salary", END)
workflow.add_edge("leave", END)
workflow.add_edge("policy", END)
workflow.add_edge("chat", END)

agent_app = workflow.compile()

# --- Endpoint ---
@app.post("/agent")
def agent_endpoint(req: UserRequest, x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized: invalid API key")
    try:
        final_state = agent_app.invoke({"user_input": req.user_input})
        return {"response": final_state["response"]}
    except Exception as e:
        logger.exception("Error processing request")
        return {"error": str(e)}
