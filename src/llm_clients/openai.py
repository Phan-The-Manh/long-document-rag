import os
from openai import OpenAI, AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

def get_openai_client(is_async=False):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found.")
    
    # Return Async client if requested, otherwise Sync
    if is_async:
        return AsyncOpenAI(api_key=api_key)
    return OpenAI(api_key=api_key)

# Export both instances
client = get_openai_client(is_async=False)
async_client = get_openai_client(is_async=True)