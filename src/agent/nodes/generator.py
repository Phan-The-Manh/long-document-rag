from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage
from src.agent.state import AgentState

# 1. Initialize the Model
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# 2. Define Prompts
gen_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Answer the question based on your general knowledge and the conversation history."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])

gen_prompt_with_context = ChatPromptTemplate.from_messages([
    ("system", """You are a strictly grounded assistant. 
    1. Use the Context below to answer the user's question, including metadata.
    2. If the Context is empty, or if the answer is not contained within the Context, 
       respond exactly with: "I am sorry, I do not have information about that in my documents."
    3. Do NOT use your general knowledge to answer. 
    4. Stay focused on the provided facts.

    Context:
    {context}"""),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])

# 3. The Pipe
# This returns a clean string thanks to StrOutputParser()
generator_with_retrieval_chain = gen_prompt_with_context | llm | StrOutputParser()
general_generator_chain = gen_prompt | llm | StrOutputParser()

def generator_node(state: AgentState):
    """
    Generator node using 'is_search_required' to switch between 
    Grounded RAG and General Knowledge.
    """
    # 1. Pull data from state
    messages = state["messages"]
    docs = state.get("documents", [])
    # Using your recovered boolean flag
    is_search_required = state.get("is_search_required", False) 
    
    # 2. Prepare common inputs
    # 'history' is everything except the latest message
    # 'input' is the latest user message
    inputs = {
        "history": messages[:-1],
        "input": messages[-1].content
    }

    # 3. Logic Branching
    if is_search_required:
        if docs:
            context_parts = []
            for d in docs:
                # Extract your specific metadata keys
                source = d.metadata.get("source", "Unknown Document")
                path = d.metadata.get("section_path", "General Section")
                pages = d.metadata.get("pages_label", "N/A")
                content = d.page_content
                
                # Create a structured header for each chunk
                header = f"[Source: {source} | Section: {path} | Page(s): {pages}]"
                chunk_text = f"{header}\nContent: {content}"
                
                context_parts.append(chunk_text)
            
            context_str = "\n\n---\n\n".join(context_parts)
        else:
            # Triggers the "I do not know" logic in your prompt
            context_str = ""

        inputs["context"] = context_str
        answer_text = generator_with_retrieval_chain.invoke(inputs)
        
    else:
        # --- PATH: GENERAL GENERATOR ---
        # is_search_required is False
        answer_text = general_generator_chain.invoke(inputs)

    # 4. Return update
    return {"messages": [AIMessage(content=answer_text)]}