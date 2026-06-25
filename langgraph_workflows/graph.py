from langgraph_workflows.state import GraphState
from langgraph.graph import StateGraph,END
from agents.translation_normalization_agent import translation_agent 

builder = StateGraph(GraphState)

builder.add_node("translation",translation_agent)
builder.set_entry_point("translation")
builder.add_edge("translation",END)
graph = builder.compile()