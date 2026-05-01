import sys
import time
import uuid
from langchain_core.messages import HumanMessage

# Import both the pipeline AND the clear function from your source file
from src.doc_processing.source_to_chroma import (
    process_and_save_full_pipeline, 
    clear_chroma_database
)
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
    
    # Generate a unique thread_id for this session's conversation history
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    
    while True:
        user_input = input("\nUser: ").strip()
        
        if user_input.lower() in ['quit', 'exit']:
            print("Shutting down...")
            # --- CLEAR FUNCTION CALLED ON EXIT ---
            clear_chroma_database() 
            print("Goodbye!")
            break
        
        if not user_input:
            continue

        print("Thinking...")
        try:
            # Invoke the graph with history tracking
            final_state = app.invoke(
                {"messages": [HumanMessage(content=user_input)]}, 
                config=config
            )
            
            response = final_state["messages"][-1].content
            print(f"\nAgent: {response}")
            
        except Exception as e:
            print(f"Agent Error: {e}")
            import traceback
            traceback.print_exc()
        
        print("-" * 30)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        # Ensure cleanup even if the user hits Ctrl+C
        print("\n[SYSTEM] Interrupted. Cleaning up...")
        clear_chroma_database()
        sys.exit(0)