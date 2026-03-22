import sys
import time
import uuid
from langchain_core.messages import HumanMessage
from src.doc_processing.source_to_chroma import process_and_save_full_pipeline
from src.agent.graph import app 

def main():
    print("--- LONG_DOC_AGENT ACTIVE ---")
    print("(Press Enter to skip if you don't have a document link)")
    
    # 1. Optional Document Ingestion
    doc_link = input("Document link/path: ").strip()
    source_name = "user_upload"
    
    if doc_link:
        print(f"\n[SYSTEM] Starting background processing for: {doc_link}")
        try:
            start_time = time.time()
            process_and_save_full_pipeline(doc_link, source_name)
            duration = round(time.time() - start_time, 2)
            print(f"[SUCCESS] Document indexed in {duration}s.\n")
        except Exception as e:
            print(f"[ERROR] Failed to process document: {e}")
            print("[SYSTEM] Continuing in general chat mode...\n")
    else:
        print("[SYSTEM] No document provided. Operating in general chat mode.\n")

    # 2. Continuous Chat Loop
    print("Type 'exit' or 'quit' to stop.")
    
    # thread_id ensures LangGraph keeps track of the conversation history
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    
    while True:
        user_input = input("\nUser: ").strip()
        
        if user_input.lower() in ['quit', 'exit']:
            print("Shutting down. Goodbye!")
            break
        
        if not user_input:
            continue

        print("Thinking...")
        try:
            # Wrap user input as a HumanMessage for consistency
            # Invoke the graph
            final_state = app.invoke(
                {"messages": [HumanMessage(content=user_input)]}, 
                config=config
            )
            
            # Extract the content from the last message (which should be an AIMessage)
            response = final_state["messages"][-1].content
            print(f"\nAgent: {response}")
            
        except Exception as e:
            print(f"Agent Error: {e}")
            # Printing the full error for debugging
            import traceback
            traceback.print_exc()
        
        print("-" * 30)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[SYSTEM] Interrupted. Closing safely...")
        sys.exit(0)