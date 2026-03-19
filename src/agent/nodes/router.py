from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from src.agent.state import AgentState
from dotenv import load_dotenv
load_dotenv()

class RetrievalRoute(BaseModel):
    """Determine if the user's request requires searching documents."""
    reasoning: str = Field(description="Brief explanation of why retrieval is or isn't needed.")
    is_search_required: bool = Field(description="True if we need to search docs, False if it's general chat.")
    query: str = Field(description="The rewritten search query incorporating chat history.")

class FollowUpRoute(BaseModel):
    """Determine if the user's message is a follow-up that relies on previous context based on pronouns or underspecified terms."""
    is_follow_up: bool = Field(description="True if this is a follow-up, False if it's a new topic.")

NOT_FOLLOW_UP_PROMPT = ChatPromptTemplate.from_template("""
### ROLE
You are a Topic Classifier. The user has started a new thread of thought or this is the first message.

### INPUT
- USER_INPUT: {user_input}

### MISSION:
Decide if this specific user input requires information from the loaded document.

### RULES:
- Set 'is_search_required' to TRUE if the user asks for a summary, specific content, or a technical explanation.
- Set 'is_search_required' to FALSE if the user is greeting you, asking for your name or say goodbye to you.
- 'query': Extract the core entities and keywords from the user's message to create a clean search query.
""")

FOLLOW_UP_PROMPT = ChatPromptTemplate.from_template("""
### ROLE
You are a Context-Aware Topic Classifier. 

### INPUTS
- CONTEXT: {context}
- USER_INPUT: {user_input}

### MISSION
1. REWRITE: Analyze the 'USER_INPUT'. If it contains pronouns or underspecified terms (e.g., "it", "that", "this", "the paper", "the results"), use the 'CONTEXT' to rewrite it into a standalone, explicit search query.
2. EVALUATE: Check if the 'CONTEXT' provided above already contains the specific, detailed answer. If not, set 'is_search_required' to TRUE.

### RULES
- is_search_required = TRUE: If the user asks for content, data, or explanations NOT fully detailed in the current CONTEXT.
- is_search_required = FALSE: If the answer is already visible in the CONTEXT, or if the user is just saying "thanks", "hello", or "goodbye".
""")

llm = ChatOpenAI(model="gpt-4o", temperature=0)
retrieval_llm = llm.with_structured_output(RetrievalRoute)
follow_up_checker = llm.with_structured_output(FollowUpRoute)

follow_up_router_chain = FOLLOW_UP_PROMPT | retrieval_llm
not_follow_up_router_chain = NOT_FOLLOW_UP_PROMPT | retrieval_llm

# Ensure your model definitions match your LLM outputs
class FollowUpRoute(BaseModel):
    is_follow_up: bool = Field(description="True if this is a follow-up, False if it's a new topic.")

class RetrievalRoute(BaseModel):
    is_search_required: bool = Field(description="Whether a document search is needed")
    query: str = Field(description="The standalone search query")

def router_node(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1].content
    print(f"--- ROUTER NODE RECEIVED: {last_message} ---")
    # --- STEP 1: Determine if we should treat this as a follow-up ---
    # We check if there's history AND use the LLM to confirm the intent
    is_follow_up_intent = False
    if len(messages) > 1:
        # Check for referential pronouns as a pre-filter or pass directly to LLM
        check_result = follow_up_checker.invoke(f"User: {last_message}")
        is_follow_up_intent = check_result.is_follow_up

    print(f"Follow-up Intent: {is_follow_up_intent}")

    # --- STEP 2: Execute the appropriate classification chain ---
    if is_follow_up_intent:
        prev_msg = messages[-2]
        context_str = f"{prev_msg.type}: {prev_msg.content}"
        decision = follow_up_router_chain.invoke({
            "context": context_str,
            "user_input": last_message
        })
    else:
        decision = not_follow_up_router_chain.invoke({
            "user_input": last_message
        })

    print(f"Router Decision: {decision.dict()}")
    # --- STEP 3: Return the result to the graph ---
    if decision.is_search_required:
        return {"query": decision.query, "is_search_required": True}
    
    return {"query": "", "is_search_required": False}

def route_decision(state: AgentState):
    """
    Conditional edge logic to decide the next path.
    """
    if state.get("is_search_required"):
        print("Routing to Retriever Node...\n")
        return "retriever"
    print("Routing to Generator Node (general chat)...\n")
    return "generator"