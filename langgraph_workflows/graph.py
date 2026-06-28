from langgraph_workflows.state import GraphState
from langgraph.graph import StateGraph,END
from agents.translation_normalization_agent import translation_agent 
from agents.completenss_agent import completeness_agent
from agents.human_review import human_review
from agents.ehr_validation_agent import ehr_validation_agent
from langgraph.checkpoint.memory import InMemorySaver
import sqlite3

from langgraph.checkpoint.sqlite import SqliteSaver

builder = StateGraph(GraphState)

def route_after_completeness(state:GraphState):
    print("in route function ",state['missing_fields'])
    if len(state['missing_fields']) > 0:
        return "human_review"
    return "ehr_validation"

builder.add_node("translation",translation_agent)
builder.add_node("completeness",completeness_agent)
builder.add_node("human_review",human_review)
builder.add_node("ehr_validation",ehr_validation_agent)
builder.set_entry_point("translation")
builder.add_edge("translation","completeness")
# builder.add_edge("completeness",END)
builder.add_conditional_edges(
    "completeness",
    route_after_completeness,
    {
        "human_review": "human_review",
        "ehr_validation": "ehr_validation"
    }
)
builder.add_edge("human_review","ehr_validation")
builder.add_edge("ehr_validation",END)
# graph = builder.compile(checkpointer = InMemorySaver())
# conn = SqliteSaver.from_conn_string("hospital_graph.db")

# graph = builder.compile(
#     checkpointer=conn
# )
conn = sqlite3.connect(
    "hospital_graph.db",
    check_same_thread=False
)

checkpointer = SqliteSaver(conn)

graph = builder.compile(
    checkpointer=checkpointer
)