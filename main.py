from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from typing import TypedDict, Optional
import logging
import os
import pandas as pd
from pinecone import Pinecone
from google.genai import Client, types

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
print("this is just a test")

# --- Request model ---
class UserRequest(BaseModel):
    user_input: str

# --- Load CSVs ---
DATA_DIR = os.path.join(os.path.dirname(__file__), "Data")
salary_df = pd.read_csv(os.path.join(DATA_DIR, "salary.csv"))
leaves_df = pd.read_csv(os.path.join(DATA_DIR, "leaves.csv"))

# --- Pinecone setup for policies ---
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
SECTION_INDEX_NAME = "policy-sections"
CHUNK_INDEX_NAME = "policy-chunks"
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

pc = Pinecone(api_key=PINECONE_API_KEY)

# sanity check that indexes exist (optional but helpful)
existing = pc.list_indexes().names()
if SECTION_INDEX_NAME not in existing or CHUNK_INDEX_NAME not in existing:
    raise RuntimeError(
        f"Required Pinecone indexes not found. "
        f"Have: {existing}. Need: {SECTION_INDEX_NAME}, {CHUNK_INDEX_NAME}"
    )

section_index = pc.Index(SECTION_INDEX_NAME)
chunk_index = pc.Index(CHUNK_INDEX_NAME)
client = Client(api_key=GOOGLE_API_KEY)

# --- Embeddings helper (fixes previous 'embed_text' NameError) ---
# Use the same embedding model/dimension you used to build the indexes.
#embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

def embed_text(text: str):
    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=[text],
        config=types.EmbedContentConfig(output_dimensionality=768)
    )
    return result.embeddings[0].values

# --- Agent state ---
class AgentState(TypedDict, total=False):
    user_input: str
    intent: Optional[str]
    name: Optional[str]
    emp_id: Optional[int]
    response: Optional[str]

# --- Helper functions ---
def get_employee_record(identifier: str, df: pd.DataFrame):
    """Fetch employee row by name or employee_id."""
    if identifier.isdigit():
        row = df[df["employee_id"].astype(str) == identifier]
    else:
        row = df[df["name"].str.lower() == identifier.lower()]
    return row

def get_salary(identifier: str):
    row = get_employee_record(identifier, salary_df)
    if row.empty:
        return "Employee not found."
    if len(row) > 1:
        emp_list = ", ".join(
            f"{r['name']} (ID: {r['employee_id']})" for _, r in row.iterrows()
        )
        return f"Multiple employees found: {emp_list}. Please specify the employee ID."
    r = row.iloc[0]
    return f"Salary for {r['name']} (ID: {r['employee_id']}): {r['salary']}"

def get_leaves(identifier: str):
    row = get_employee_record(identifier, leaves_df)
    if row.empty:
        return "Employee not found."
    if len(row) > 1:
        emp_list = ", ".join(
            f"{r['name']} (ID: {r['employee_id']})" for _, r in row.iterrows()
        )
        return f"Multiple employees found: {emp_list}. Please specify the employee ID."
    r = row.iloc[0]
    return (
        f"Leave balance for {r['name']} (ID: {r['employee_id']}): "
        f"Casual: {r['casual_leave']}, Sick: {r['sick_leave']}, Earned: {r['earned_leave']}"
    )

# --- Parent–child policy search ---
def query_policy(text: str, top_k_sections: int = 2, top_k_chunks: int = 3):
    """
    1) Search top sections (parents) using SECTION_INDEX_NAME
    2) For each section, search top chunks (children) in CHUNK_INDEX_NAME filtered by section
    3) Return all child hits, sorted by score desc
    """

    # Embed the query
    vector = embed_text(text)


    # Step 1: search sections (parents)
    section_results = section_index.query(
        vector=vector,
        top_k=top_k_sections,
        include_metadata=True
    )

    matches = section_results.get("matches") if isinstance(section_results, dict) else section_results.matches
    if not matches:
        return [{"text": "Policy not found", "section": None, "title": None, "score": 0.0}]

    final_results = []

    for sec in matches:
        sec_meta = sec["metadata"] if isinstance(sec, dict) else sec.metadata
        sec_id = f"section-{sec_meta.get('title')}"
        sec_title = sec_meta.get("title", "")

        if not sec_id:
            # If the section id wasn't stored, skip to stay safe
            continue

        # Step 2: search chunks (children) within this section
        chunk_results = chunk_index.query(
            vector=vector,
            top_k=top_k_chunks,
            include_metadata=True,
            filter={"section": {"$eq": sec_id}}
        )
        chunk_matches = chunk_results.get("matches") if isinstance(chunk_results, dict) else chunk_results.matches

        for cm in (chunk_matches or []):
            cm_meta = cm["metadata"] if isinstance(cm, dict) else cm.metadata
            final_results.append({
                "section": sec_id,
                "title": sec_title,
                "text": cm_meta.get("text", ""),
                "score": cm["score"] if isinstance(cm, dict) else cm.score
            })

    # Sort by relevance score (desc)
    final_results.sort(key=lambda x: x.get("score", 0.0), reverse=True)

    # Fallback if no children matched
    if not final_results:
        # Return parents (sections) themselves as a last resort
        fallback = []
        for sec in matches:
            sec_meta = sec["metadata"] if isinstance(sec, dict) else sec.metadata
            fallback.append({
                "section": f"section-{sec_meta.get('title')}",
                "title": sec_meta.get("title", ""),
                "text": sec_meta.get("text", ""),  # if you stored section text in metadata
                "score": sec["score"] if isinstance(sec, dict) else sec.score
            })
        return fallback or [{"text": "Policy not found", "section": None, "title": None, "score": 0.0}]

    return final_results

# --- Classification + Name Extraction Node ---
intent_prompt = ChatPromptTemplate.from_template("""
You are an HR assistant.
User query: {query}

Classify the user's intent into ONE of these categories:

- "salary" → if the user is asking about the salary of a specific employee.
- "leave" → if the user is asking about the leave balance of a specific employee.
- "policy" → if the user is asking about company policies, rules, or entitlements in general 
   (e.g., "How many leaves per year?", "What is the work from home policy?").
- "chat" → for small talk or anything else.

Reply ONLY with the intent word: salary, leave, policy, or chat.
""")

name_prompt = ChatPromptTemplate.from_template("""
You are an HR assistant.
User query: {query}

Extract the employee name if it exists.

Respond with ONLY the name as plain text.  
If no name is found, respond with 000.
""")

id_prompt = ChatPromptTemplate.from_template("""
You are an HR assistant.
User query: {query}

Extract the employee id if it exists.

Respond with ONLY the id as plain text.  
If no id is found, respond with AAA.
""")

def classify_and_extract(state: AgentState):
    raw_name = ""
    raw_intent = ""
    raw_id = None

    chain = intent_prompt | llm
    raw_intent = chain.invoke({"query": state["user_input"]}).content.strip().lower()

    if raw_intent in ["leave", "salary"]:
        chain = name_prompt | llm
        raw_name = chain.invoke({"query": state["user_input"]}).content.strip()

        chain = id_prompt | llm
        raw_id = chain.invoke({"query": state["user_input"]}).content.strip()

    state["name"] = None if raw_name == "000" else raw_name
    state["emp_id"] = None if raw_id == "AAA" else raw_id
    state["intent"] = raw_intent.lower()
    return state

# --- Intent Nodes ---
def salary_node(state: AgentState):
    name_or_id = state["name"] or state["emp_id"] or state["user_input"]
    state["response"] = get_salary(name_or_id)
    return state

def leave_node(state: AgentState):
    name_or_id = state["name"] or state["emp_id"] or state["user_input"]
    state["response"] = get_leaves(name_or_id)
    return state

def policy_node(state: AgentState):
    # Step 1: retrieve top parent+child chunks
    hits = query_policy(state["user_input"], top_k_sections=2, top_k_chunks=3)
    
    if not hits:
        state["response"] = "Policy not found."
        return state

    # Step 2: combine top chunks into a single context string
    context = "\n".join([f"[{hit['title']}]\n{hit['text']}" for hit in hits])

    # Step 3: prompt the LLM to answer using the retrieved context
    prompt = f"""
You are an HR assistant. Use the following policy excerpts to answer the user's question.

User question: {state['user_input']}

Policy excerpts:
{context}

Answer in a concise, clear, professional way.
"""
    response = llm.invoke([
        {"role": "system", "content": "You are a helpful HR assistant."},
        {"role": "user", "content": prompt}
    ])
    state["response"] = response.content.strip()
    return state


def chat_node(state: AgentState):
    response = llm.invoke([
        {"role": "system", "content": "You are a helpful HR assistant."},
        {"role": "user", "content": state["user_input"]}
    ])
    state["response"] = response.content.strip()
    return state

# --- Build LangGraph ---
workflow = StateGraph(AgentState)
workflow.add_node("classify", classify_and_extract)
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
