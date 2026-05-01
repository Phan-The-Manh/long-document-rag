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
    2. Do NOT include inline citations in the answer text.
    3. If the answer is not in the context, say: "I don't have enough information to answer this."
    4. For tables/numbers: Extract precisely from the context.
    5. Be concise but complete.

    ### OUTPUT FORMAT:
    Write a direct prose answer with no inline citations. Add brief reasoning only if needed for clarity. Then end with exactly this block (no square brackets, plain text only):

    Sources List:
    - Source: X | Section: Y | Page: Z
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
        if not docs:
            answer_text = "I don't have enough information to answer this."
        else:
            context_parts = []
            for d in docs:
                source = d.metadata.get("source", "Unknown Document")
                path = d.metadata.get("section_path", "General Section")
                pages = d.metadata.get("pages_label", "N/A")
                content = d.page_content
                header = f"[Source: {source} | Section: {path} | Page(s): {pages}]"
                context_parts.append(f"{header}\nContent: {content}")

            context_str = "\n\n---\n\n".join(context_parts)
            inputs["context"] = context_str

            formatted_prompt = gen_prompt_with_context.format(
                context=context_str,
                input=messages[-1].content
            )
            print("Formatted Prompt for Generator with Context:\n")
            print(formatted_prompt)

            answer_text = generator_with_retrieval_chain.invoke(inputs)
        
    else:
        print("Generating without context (general knowledge)!\n")
        # --- PATH: GENERAL GENERATOR ---
        # is_search_required is False
        answer_text = general_generator_chain.invoke(inputs)

    # 4. Return update
    return {"messages": [AIMessage(content=answer_text)]}