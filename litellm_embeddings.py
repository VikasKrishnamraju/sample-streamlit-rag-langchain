"""
Custom LiteLLM Embeddings wrapper for LangChain
"""
import os
from typing import List
import litellm
from langchain_core.embeddings import Embeddings
import environ

env = environ.Env()
#reading .env file
environ.Env.read_env()


class LiteLLMEmbeddings(Embeddings):
    """Custom LiteLLM embeddings using internal endpoint"""
    
    def __init__(self, model: str = "text-embedding-3-small"):
        # Set up environment for LiteLLM
        os.environ["OPENAI_API_KEY"] = env("LITELLM_API_KEY")
        self.model = model
        self.api_base = "https://litellm.int.thomsonreuters.com"
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents"""
        try:
            response = litellm.embedding(
                model=self.model,
                input=texts,
                api_base=self.api_base
            )
            embeddings = [item['embedding'] for item in response['data']]
            print(f"Successfully created embeddings for {len(texts)} documents")
            return embeddings
        except Exception as e:
            print(f"ERROR in embed_documents: {e}")
            print(f"Model: {self.model}, API Base: {self.api_base}")
            # Fallback: return dummy embeddings with consistent dimensions
            print("WARNING: Using dummy embeddings - RAG functionality will be impaired")
            return [[0.0] * 1536 for _ in texts]  # 1536 for text-embedding-3-small
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query"""
        try:
            response = litellm.embedding(
                model=self.model,
                input=[text],
                api_base=self.api_base
            )
            embedding = response['data'][0]['embedding']
            print(f"Successfully created query embedding")
            return embedding
        except Exception as e:
            print(f"ERROR in embed_query: {e}")
            print(f"Model: {self.model}, API Base: {self.api_base}")
            # Fallback: return dummy embedding
            print("WARNING: Using dummy query embedding - RAG functionality will be impaired")
            return [0.0] * 1536  # 1536 for text-embedding-3-small