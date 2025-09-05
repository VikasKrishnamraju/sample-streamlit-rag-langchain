from ragas import evaluate
from datasets import Dataset
from vector_functions import get_lite_llm_model
from litellm_embeddings import LiteLLMEmbeddings
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall, context_entity_recall

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
        return score
        
    except Exception as e:
        print(f"Error during RAGAS evaluation: {e}")
        return None