from langgraph.graph import StateGraph, START, END
from src.agent.state import AgentState
from src.agent.nodes.router import router_node, route_decision
from src.agent.nodes.retriever import retriever_node
from src.agent.nodes.generator import generator_node

# 1. Initialize the StateGraph
workflow = StateGraph(AgentState)

# 2. Add all Nodes
workflow.add_node("router", router_node)
workflow.add_node("retriever", retriever_node)
workflow.add_node("generator", generator_node)

# 3. Define the Flow
# Start with the router
workflow.add_edge(START, "router")

# Add the branching logic from the router
workflow.add_conditional_edges(
    "router",
    route_decision,
    {
        "retriever": "retriever", # If True -> Search docs
        "generator": "generator"  # If False -> General chat
    }
)

# After retrieval, move to the generator to synthesize the answer
workflow.add_edge("retriever", "generator")

# The generator is the end of the turn
workflow.add_edge("generator", END)

# 4. Compile the Graph
# checkpointer allows for persistence/memory (optional but recommended)
app = workflow.compile()

