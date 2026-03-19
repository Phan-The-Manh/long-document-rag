from typing import TypedDict, Annotated, Dict, Any, List
from langchain.messages import AnyMessage
import operator

class AgentState(TypedDict):
    messages : Annotated[List[AnyMessage], operator.add]
    query: str
    documents: List[dict]
    is_search_required: bool