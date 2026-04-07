import os
import json
import operator
from dotenv import load_dotenv
from typing import TypedDict, List, Dict, Any, Annotated, Optional

from langgraph.graph import StateGraph, START, END  # graph workflow primitives [1](https://docs.langchain.com/oss/python/langgraph/graph-api)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from service import cloudwatch
# Hugging Face (chat-style)
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace


# ----------------------------
# 0) Config
# ----------------------------
load_dotenv()  # Loads variables from .env
HF_TOKEN = os.environ.get("HUGGINGFACEHUB_API_TOKEN")
HF_MODEL_ID = os.environ.get("HF_MODEL_ID", "meta-llama/Llama-3.1-8B-Instruct")

DEFAULT_CHUNK_CHARS = 8000  # safe char-based chunk size for big Spark logs
DEFAULT_MAX_NEW_TOKENS = 600


# ----------------------------
# 1) Utilities (chunking + pre-filter)
# ----------------------------
KEYWORDS = (
    "ERROR", "Exception", "SparkException", "Caused by",
    "OutOfMemoryError", "ExecutorLostFailure", "Container killed",
    "TaskKilled", "ExitCode", "GC overhead", "FileNotFoundException",
    "AnalysisException", "AccessDenied", "Permission denied", "Timeout"
)

def filter_log_lines(log_text: str, keep_context_lines: int = 1) -> str:
    """
    Keep only high-signal lines (+ a little context) to reduce prompt size.
    """
    lines = log_text.splitlines()
    keep = [False] * len(lines)

    for i, line in enumerate(lines):
        if any(k in line for k in KEYWORDS):
            for j in range(max(0, i - keep_context_lines), min(len(lines), i + keep_context_lines + 1)):
                keep[j] = True

    filtered = [lines[i] for i in range(len(lines)) if keep[i]]
    return "\n".join(filtered) if filtered else log_text


def chunk_text(text: str, max_chars: int = DEFAULT_CHUNK_CHARS) -> List[str]:
    """
    Simple, robust char-based chunking.
    """
    return [text[i:i + max_chars] for i in range(0, len(text), max_chars)]


# ----------------------------
# 2) Build LLM / Chat Model
# ----------------------------
def build_chat_model():
    """
    Hugging Face conversational wrapper. Swap with your preferred model provider if needed.
    """
    if not HF_TOKEN:
        raise RuntimeError("HUGGINGFACEHUB_API_TOKEN is not set in environment variables.")

    endpoint_llm = HuggingFaceEndpoint(
        repo_id=HF_MODEL_ID,
        task="conversational",
        huggingfacehub_api_token=HF_TOKEN,
        max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
        temperature=0.1,
        provider="auto"
    )
    return ChatHuggingFace(llm=endpoint_llm)


chat_model = build_chat_model()
parser = StrOutputParser()


# ----------------------------
# 3) Define Agent Prompts (each node is an agent)
# ----------------------------

# (Agent-1) Chunk Summarizer
summarizer_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a Spark/Glue production support engineer. "
     "Summarize the given Spark log CHUNK.\n"
     "Rules:\n"
     "- Focus only on errors, exceptions, stage/task failures, resource issues, retries, executor loss.\n"
     "- Extract key stack trace lines or error messages as evidence.\n"
     "- Keep output concise and structured with headings: Signals, Evidence, Hypotheses.\n"
    ),
    ("human", "Spark log chunk:\n\n{chunk}\n")
])

summarizer_chain = summarizer_prompt | chat_model | parser


# (Agent-2) Error Identifier (structured JSON)
error_identifier_prompt = ChatPromptTemplate.from_messages([
    (
"system",
     "You are an expert at Spark log triage.\n"
     "Return JSON ONLY with keys:\n"
     "- primary_errors: list of objects with fields "
     "(error_type, short_message, evidence[list], likely_causes[list], severity)\n"
     "- secondary_signals: list of strings\n"
     "- missing_info_to_confirm: list of strings\n"
     "- best_guess_root_cause: string\n"
     "No markdown, no extra text."
    ),
    ("human", "Combined chunk summaries:\n\n{combined_summary}\n")
])

error_identifier_chain = error_identifier_prompt | chat_model | parser


# (Agent-3) RCA + Recommendations
rca_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a senior Spark/Glue incident manager.\n"
     "Using the extracted error JSON and any remaining context, produce a clear RCA.\n"
     "Output in this format:\n"
     "1) Incident summary (2-3 lines)(bulleted)- include incident id as a generated number starting with INC, incident type and incident description\n"
     "2) Most likely root cause\n"
     "3) Evidence(bulleted) - no secondary signals\n"
     "4) Recommended fixes (bulleted) - include code-level changes + ops actions\n"
     "give result in markdown format\n"
    ),
    ("human",
     "Error JSON:\n{error_json}\n\n"
     "Combined Summary:\n{combined_summary}\n"
    )
])

rca_chain = rca_prompt | chat_model | parser

#Agent 4 - Jira Story and log into system
jira_prompt =  ChatPromptTemplate.from_messages([
    ("system",
      "You are an experienced Production Support engineer.\n"
      "Your task is to generate a clear, professional JIRA ticket "
      "for a job failure incident using the rca and combined summary\n\n"
      "Follow JIRA best practices:\n"
        "- Concise but informative summary\n"
        "- Clear problem description\n"
        "- No assumptions or speculation\n"
        "- Professional, neutral tone\n"
        "- Do not keep any placeholder if not details available\n"
     
     "Output in this format:\n"
     "- Summary:\n"
     "- Description:\n"
     "- Error Message(bulleted): - summerize combined_summary\n"
     "- Root Cause Analysis (bulleted): summerize rca_report\n\n"
     "give result in markdown format\n"
    ),
    ("human",
          "rca_report:\n{rca_report}\n\n"
          "combined_summary:\n{combined_summary}\n\n"
    )
])

jira_chain = jira_prompt | chat_model | parser

# ----------------------------
# 4) LangGraph State
# ----------------------------
class LogRCAState(TypedDict):
    # input
    raw_log: str
    chunk_chars: int

    # derived
    filtered_log: str
    chunks: List[str]

    # results
    chunk_summaries: Annotated[List[str], operator.add]   # reducer-friendly list aggregation
    combined_summary: str

    error_json_raw: str
    error_json: Dict[str, Any]

    rca_report: str
    jira_ticket: str


# ----------------------------
# 5) Agent Nodes
# ----------------------------

def summarizer_agent(state: LogRCAState) -> Dict[str, Any]:
    """
    Agent 1: filter -> chunk -> summarize each chunk.
    """
    raw_log = state["raw_log"]
    chunk_chars = state.get("chunk_chars", DEFAULT_CHUNK_CHARS)

    filtered = filter_log_lines(raw_log, keep_context_lines=1)
    chunks = chunk_text(filtered, max_chars=chunk_chars)

    summaries = []
    for idx, chunk in enumerate(chunks, start=1):
        out = summarizer_chain.invoke({"chunk": chunk})
        summaries.append(f"=== Chunk {idx}/{len(chunks)} Summary ===\n{out}")

    combined = "\n\n".join(summaries)

    return {
        "filtered_log": filtered,
        "chunks": chunks,
        "chunk_summaries": summaries,
        "combined_summary": combined
    }


def error_identifier_agent(state: LogRCAState) -> Dict[str, Any]:
    """
    Agent 2: extract structured errors from summaries.
    """
    combined_summary = state["combined_summary"]
    raw = error_identifier_chain.invoke({"combined_summary": combined_summary})

    # Robust JSON parse (LLMs sometimes add stray text)
    parsed: Dict[str, Any] = {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        # Attempt to salvage JSON between first { and last }
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            parsed = json.loads(raw[start:end+1])
        else:
            parsed = {"primary_errors": [], "secondary_signals": [], "missing_info_to_confirm": [],
                      "best_guess_root_cause": "Could not parse JSON output reliably."}

    return {
        "error_json_raw": raw,
        "error_json": parsed
    }


def rca_agent(state: LogRCAState) -> Dict[str, Any]:
    """
    Agent 3: RCA + recommendations.
    """
    combined_summary = state["combined_summary"]
    error_json = state["error_json"]

    rca = rca_chain.invoke({
        "error_json": json.dumps(error_json, indent=2),
        "combined_summary": combined_summary
    })

    return {"rca_report": rca}


def jira_agent(state: LogRCAState) -> Dict[str, Any]:
  """
  Agent 4: Jira Description.
  """
  combined_summary = state["combined_summary"]
  rca_report = state["rca_report"]
  
  jira = jira_chain.invoke({
    "rca_report": json.dumps(rca_report, indent=2),
    "combined_summary": combined_summary
  })
  
  return {"jira_ticket": jira}

# ----------------------------
# 6) Build the Graph (3 agents in sequence)
# ----------------------------
graph = StateGraph(LogRCAState)  # StateGraph: state + nodes + edges [1](https://docs.langchain.com/oss/python/langgraph/graph-api)

graph.add_node("summarizer_agent", summarizer_agent)
graph.add_node("error_identifier_agent", error_identifier_agent)
graph.add_node("rca_agent", rca_agent)
graph.add_node("jira_agent", jira_agent)

graph.add_edge(START, "summarizer_agent")
graph.add_edge("summarizer_agent", "error_identifier_agent")
graph.add_edge("error_identifier_agent", "rca_agent")
graph.add_edge("rca_agent", "jira_agent")
graph.add_edge("jira_agent", END)

app = graph.compile()  # required compile step [1](https://docs.langchain.com/oss/python/langgraph/graph-api)


# ----------------------------
# 7) Run It
# ----------------------------
# if __name__ == "__main__":

def get_rca(response_type, job_id):
    # Example: replace with CloudWatch or file-read log text
    cloudwatch.get_log_details(job_id)
    spark_log_text = open("./data/spark_log.txt", "r", encoding="utf-8", errors="ignore").read()

    result = app.invoke({
        "raw_log": spark_log_text,
        "chunk_chars": 8000,
        "filtered_log": "",
        "chunks": [],
        "chunk_summaries": [],
        "combined_summary": "",
        "error_json_raw": "",
        "error_json": {},
        "rca_report": "",
        "jira_ticket": ""
    })

    # print("\n================ RCA REPORT ================\n")
    # print(result["rca_report"])
    #
    # print("\n================ JIRA TICKET ================\n")
    # print(result["jira_ticket"])
    
    return result[response_type]

# print(get_rca("jira_ticket"))