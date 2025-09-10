from ragas import evaluate
from datasets import Dataset
from vector_functions import get_lite_llm_model
from litellm_embeddings import LiteLLMEmbeddings
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall, context_entity_recall
import environ
from langfuse import Langfuse
import os
import pandas as pd

env = environ.Env()
environ.Env.read_env()

def eval_thr_ragas(query, answer, retrieved_contexts):
    """
    Evaluate RAG performance using RAGAS metrics.
    
    Args:
        query (str): The user's question
        answer (str): The generated answer
        retrieved_contexts (list): List of retrieved context strings
    """

    # Ensure all inputs are in the correct format for RAGAS
    questions = [query] if isinstance(query, str) else query
    answers = [answer] if isinstance(answer, str) else answer
    
    # Contexts should be a list of lists - each question gets a list of context strings
    if isinstance(retrieved_contexts, str):
        contexts = [[retrieved_contexts]]
    elif isinstance(retrieved_contexts, list):
        if len(retrieved_contexts) > 0 and isinstance(retrieved_contexts[0], str):
            contexts = [retrieved_contexts]  # Single question with multiple contexts
        else:
            contexts = retrieved_contexts  # Already in correct format
    else:
        contexts = [[]]  # Empty context if none provided
    
    data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts
    }

    try:
        langchain_llm = get_lite_llm_model() # any langchain LLM instance
        langchain_embeddings =  LiteLLMEmbeddings(model="text-embedding-3-small")
        dataset = Dataset.from_dict(data)
        print(f"Dataset created successfully with {len(dataset)} samples")
        #print(f"Sample data: {data}")
        
        score = evaluate(
            dataset=dataset,
            metrics=[faithfulness, answer_relevancy],
            llm=langchain_llm,
            embeddings=langchain_embeddings
            )
        score_df = score.to_pandas()
        score_df.to_csv("EvaluationScores.csv", encoding="utf-8", index=False)
        
        print("RAGAS evaluation completed successfully!")
        print(f"Scores: {score}")
        
        # Re-enable Langfuse integration
        print("Attempting Langfuse integration...")
        langfuse_integration(score_df)
        
        return score
        
    except Exception as e:
        print(f"Error during RAGAS evaluation: {e}")
        return None
    

def langfuse_integration(df):
    try:
        # Configure SSL certificate bundle for enterprise environments
        cert_bundle_path = "/Users/a0144076/sample-streamlit-rag-langchain/corp-bundle-final-complete.pem"
        if os.path.exists(cert_bundle_path):
            os.environ['SSL_CERT_FILE'] = cert_bundle_path
            os.environ['REQUESTS_CA_BUNDLE'] = cert_bundle_path
            os.environ['CURL_CA_BUNDLE'] = cert_bundle_path
            os.environ['HTTPX_VERIFY'] = cert_bundle_path
            print(f"Using enterprise certificate bundle: {cert_bundle_path}")
        
        # Get credentials from environment
        public_key = env("LANGFUSE_PUBLIC_KEY", default="")
        secret_key = env("LANGFUSE_SECRET_KEY", default="")
        host = env("LANGFUSE_HOST", default="https://cloud.langfuse.com")
        
        # Check if langfuse is properly configured
        if not public_key or not secret_key:
            print("Langfuse not configured - skipping integration")
            return

        print(f"Initializing Langfuse client with host: {host}")
        
        # Initialize Langfuse client with explicit credentials
        langfuse = Langfuse(
            public_key=public_key,
            secret_key=secret_key,
            host=host
        )

        print("Langfuse client initialized successfully!")
        
        # Test basic connectivity first
        print("Testing Langfuse connectivity...")
        
        # Create a simple trace without complex operations
        trace = langfuse.trace(
            name="ragas_evaluation_test",
            metadata={"source": "streamlit_app"}
        )
        
        print(f"Created trace with ID: {trace.id}")
        
        # Log a simple event instead of scores initially
        langfuse.event(
            trace_id=trace.id,
            name="ragas_evaluation_started"
        )
        
        print("Basic Langfuse operations successful!")
        
        # Now try to log the actual scores
        score_count = 0
        for _, row in df.iterrows():
            for metric_name in ["faithfulness", "answer_relevancy"]:
                if metric_name in row and not pd.isna(row[metric_name]):
                    print(f"Logging {metric_name} = {row[metric_name]}")
                    # Simplified score logging
                    langfuse.event(
                        trace_id=trace.id,
                        name=f"metric_{metric_name}",
                        metadata={
                            "metric": metric_name,
                            "value": float(row[metric_name])
                        }
                    )
                    score_count += 1
        
        # Ensure all data is sent to Langfuse
        print("Flushing Langfuse data...")
        langfuse.flush()
        print(f"Langfuse integration completed successfully! Logged {score_count} metrics.")
        
    except Exception as e:
        print(f"Langfuse integration failed with error: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        print("Evaluation results still saved locally to EvaluationScores.csv")