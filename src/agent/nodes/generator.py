from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage
from src.agent.state import AgentState
from dotenv import load_dotenv
load_dotenv()

# 1. Initialize the Model
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# 2. Define Prompts
gen_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Answer the question based on your general knowledge."),
    ("human", "{input}"),
])

gen_prompt_with_context = ChatPromptTemplate.from_messages([
    ("system", """
    You are a highly accurate AI assistant designed for question answering over provided context.
    
    ### CONTEXT STRUCTURE:
    Each piece of context is preceded by metadata in brackets: `[Source: XXX | Section: YYY | Page(s): ZZZ]`.
    
    ### RULES:
    1. Use ONLY the provided context to answer. No prior knowledge.
    2. MANDATORY CITATION: Every fact or value you provide must be followed by an inline citation in the format: (Source, Page X). 
    3. If multiple sources support a fact, cite all of them.
    4. If the metadata says "Page(s): Unknown", use the Source and Section name instead.
    5. If the answer is not in the context, say: "I don't have enough information to answer this."
    6. For tables/numbers: Extract precisely and cite the specific table/section source.
    7. Be concise but complete.

    ### OUTPUT FORMAT:
    - [Direct Answer with inline citations]
    - [Brief Reasoning/Calculation if needed]
    - [Sources List: A bulleted list of unique sources/sections used]
    """),

    ("human", """
    Context:
    {context}

    Question: 
    {input}
    """)
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
        "input": messages[-1].content
    }

    # 3. Logic Branching
    if is_search_required:
        print("Generating with context!\n")
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

        formatted_prompt = gen_prompt_with_context.format(
            context=context_str, 
            input=messages[-1].content
        )
        print("Formatted Prompt for Generator with Context:\n")
        print(formatted_prompt) # If this shows "Context: " followed by nothing, that's your bug!

        answer_text = generator_with_retrieval_chain.invoke(inputs)
        
    else:
        print("Generating without context (general knowledge)!\n")
        # --- PATH: GENERAL GENERATOR ---
        # is_search_required is False
        answer_text = general_generator_chain.invoke(inputs)

    # 4. Return update
    return {"messages": [AIMessage(content=answer_text)]}