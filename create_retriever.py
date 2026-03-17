from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
load_dotenv()

def load_retriever_from_faiss(save_dir: str = "faiss_store", k: int = 5):
    """
    Loads a FAISS index from a local directory and returns it as a LangChain retriever.
    """
    # 1. Initialize the same embedding model used during storage
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # 2. Load the local FAISS index
    # allow_dangerous_deserialization=True is required to load the local .pkl file
    vectorstore = FAISS.load_local(
        folder_path=save_dir,
        embeddings=embeddings,
        allow_dangerous_deserialization=True
    )

    # 3. Convert the vector store into a retriever
    # 'search_kwargs' allows you to define how many chunks (k) to return
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    
    print(f"✅ Retriever loaded successfully from {save_dir}")
    return retriever

# --- Usage ---
if __name__ == "__main__":
    my_retriever = load_retriever_from_faiss("faiss_store")
    
    print(my_retriever.invoke("part 4, performance evaluation"))
    # Test the retriever
    # query = "What is the main topic of the document?"
    # results = my_retriever.invoke(query)
    # print(results[0].page_content)