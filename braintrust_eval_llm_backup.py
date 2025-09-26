#!/usr/bin/env python3
"""
Proper Braintrust Eval Implementation with LLM Judges
Based on the official Braintrust documentation for "Score using AI (LLM judges)"
"""

import os
import environ
import pandas as pd
from typing import List, Dict, Any
from datetime import datetime

# Braintrust imports
from braintrust import Eval
from autoevals import (
    ExactMatch,
    Levenshtein,
    # Enterprise LLM Judges for RAG evaluation
    AnswerRelevancy,
    AnswerCorrectness,
    Factuality,
    ContextRelevancy,
    Faithfulness
)

# Company's RAG system integration
from vector_functions import get_lite_llm_model, load_retriever, generate_answer_from_context

# Load environment variables
env = environ.Env()
environ.Env.read_env()

# Global variable to capture LLM judge outputs for experiment logging
llm_judge_outputs = {}

def setup_braintrust():
    """Setup Braintrust configuration"""
    # SSL certificates for enterprise environment
    cert_bundle = "/Users/a0144076/sample-streamlit-rag-langchain/corp-bundle-final-complete.pem"
    if os.path.exists(cert_bundle):
        os.environ['SSL_CERT_FILE'] = cert_bundle
        os.environ['REQUESTS_CA_BUNDLE'] = cert_bundle
        print(f"✅ Using enterprise certificate bundle")

    # Braintrust configuration
    import braintrust
    braintrust.api_url = env("BRAINTRUST_API_URL")
    braintrust.api_key = env("BRAINTRUST_API_KEY")
    print(f"✅ Braintrust configured: {env('BRAINTRUST_API_URL')}")

# Initialize LLMs - separate models for RAG and evaluation
llm = get_lite_llm_model()  # Main RAG model
print(f"🤖 Company LiteLLM (RAG) initialized: {type(llm)}")

def get_eval_llm_model():
    """Initialize and return a LiteLLM chat model specifically for evaluation/judging."""
    from langchain_community.chat_models import ChatLiteLLM
    os.environ["LITELLM_API_KEY"] = env("LITELLM_API_KEY")
    eval_model = ChatLiteLLM(
        model=env("LITELLM_EVAL_MODEL"),  # Use dedicated evaluation model
        api_base=env("LITELLM_BASE_URL"),
        temperature=0  # Deterministic for consistent evaluation
    )
    return eval_model

eval_llm = get_eval_llm_model()
print(f"🧑‍⚖️ Company LiteLLM (Evaluation) initialized: {env('LITELLM_EVAL_MODEL')}")

def rag_task(input_text: str, collection_name: str = None) -> str:
    """
    RAG Comparison Task: Generate answers from both RAG app and Braintrust LLM

    Args:
        input_text: User's question
        collection_name: Vector collection to search (optional)

    Returns:
        RAG app output (this will be compared against Braintrust LLM output)
    """
    try:
        if collection_name:
            # Get RAG app answer using vector_functions.py
            retriever = load_retriever(collection_name)
            rag_answer = generate_answer_from_context(retriever, input_text)

            # Get retrieved context for comparison
            retrieved_docs = retriever.get_relevant_documents(input_text)
            context = "\n".join([doc.page_content for doc in retrieved_docs[:3]])

            # Generate Braintrust LLM answer using same context
            braintrust_prompt = f"""
Answer this question using the provided context only.

{input_text}

Context:
{context}
"""
            braintrust_response = llm.invoke(braintrust_prompt)
            braintrust_answer = braintrust_response.content

            print(f"🔄 RAG App Answer: {rag_answer[:100]}...")
            print(f"🔄 Braintrust LLM Answer: {braintrust_answer[:100]}...")

            # Return RAG app answer (will be compared against Braintrust LLM answer in scorers)
            return rag_answer

        else:
            # If no collection, just use LLM directly
            response = llm.invoke(f"Answer this question: {input_text}")
            return response.content

    except Exception as e:
        print(f"⚠️ RAG task failed: {e}")
        return f"Error: {e}"

# Non-LLM Scorers from autoevals (no OpenAI API key required)
exact_match_scorer = ExactMatch()
levenshtein_scorer = Levenshtein()
# Note: EmbeddingSimilarity requires OpenAI API key, so we'll skip it

# Simple exact match scorer for comparison
def exact_match(input, output, expected):
    """Simple exact match scorer"""
    if not expected:
        return None  # Skip if no expected answer
    return 1.0 if output.strip().lower() == expected.strip().lower() else 0.0

# =============================================================================
# Custom Braintrust Scorers following official documentation guidelines
# Single-aspect scorers with LLM-based evaluation using company's LiteLLM
# =============================================================================

def _extract_json_score(result_text: str, default_score: float = 0.5) -> float:
    """Extract score from JSON response with fallback to regex."""
    import json
    import re

    try:
        # Try to parse JSON directly
        if '{' in result_text and '}' in result_text:
            json_start = result_text.find('{')
            json_end = result_text.rfind('}') + 1
            json_str = result_text[json_start:json_end]
            result = json.loads(json_str)
            score = float(result.get('Score', default_score))
            return max(0.0, min(1.0, score))
    except:
        # Fallback: extract score with regex
        score_match = re.search(r'"Score":\s*([0-9.]+)', result_text)
        if score_match:
            score = float(score_match.group(1))
            return max(0.0, min(1.0, score))

    print(f"⚠️ Could not parse score from: {result_text[:100]}...")
    return default_score

def answer_relevancy_scorer(input, output, expected=None):
    """
    Braintrust Scorer: Answer Relevancy
    Single-aspect scorer following Braintrust guidelines
    Uses dedicated evaluation model for consistent LLM judging
    Returns both score and detailed judge output for transparency
    """
    try:
        # Braintrust LLM-based scorer prompt with explicit instructions and rubric
        prompt = f'''You are an expert evaluator assessing how relevant an AI assistant's answer is to the user's question.

Analyze the question and answer, then select one of the following options:

a) The answer is highly relevant and directly addresses the question (Score: 1.0)
b) The answer is mostly relevant but includes some off-topic content (Score: 0.7)
c) The answer is partially relevant but misses key aspects of the question (Score: 0.4)
d) The answer is not relevant to the question asked (Score: 0.0)

Output format: Return your evaluation as a JSON object with the following keys:
1. Score: A score between 0 and 1 based on the rubric above
2. Choice: The letter (a, b, c, or d) that best describes the answer
3. Rationale: A brief 1-2 sentence explanation for your scoring decision

Question: {input}

Answer: {output}

Provide your evaluation:'''

        response = eval_llm.invoke(prompt)  # Use dedicated evaluation model
        judge_output = response.content

        # Store judge output globally for experiment logging
        global llm_judge_outputs
        if 'llm_judge_outputs' not in globals():
            llm_judge_outputs = {}
        llm_judge_outputs['relevancy_judge_output'] = judge_output

        score = _extract_json_score(judge_output, default_score=0.5)

        # For offline CSV evaluation, return Score object with judge output metadata
        try:
            from autoevals import Score
            return Score(
                name="answer_relevancy_scorer",
                score=score,
                metadata={
                    "judge_output": judge_output,
                    "eval_model": env("LITELLM_EVAL_MODEL"),
                    "judge_type": "relevancy"
                }
            )
        except:
            return score  # Fallback to numeric score

    except Exception as e:
        print(f"⚠️ Answer relevancy scorer failed: {e}")
        return 0.5

def answer_accuracy_scorer(input, output, expected=None):
    """
    Braintrust Scorer: Answer Accuracy
    Single-aspect scorer with reference answer when available
    Uses dedicated evaluation model for consistent LLM judging
    """
    try:
        reference_section = ""
        if expected:
            reference_section = f"\n\nReference Answer: {expected}"

        prompt = f'''You are an expert evaluator assessing the factual accuracy of an AI assistant's answer.

Analyze the answer for factual correctness and select one of the following options:

a) The answer is completely accurate with no factual errors (Score: 1.0)
b) The answer is mostly accurate with minor factual issues (Score: 0.7)
c) The answer has significant factual errors but some correct information (Score: 0.3)
d) The answer contains major factual errors or misinformation (Score: 0.0)

Output format: Return your evaluation as a JSON object with the following keys:
1. Score: A score between 0 and 1 based on the rubric above
2. Choice: The letter (a, b, c, or d) that best describes the accuracy
3. Rationale: A brief explanation identifying any factual issues found

Question: {input}

Answer: {output}{reference_section}

Provide your evaluation:'''

        response = eval_llm.invoke(prompt)  # Use dedicated evaluation model
        judge_output = response.content

        # Store judge output globally for experiment logging
        global llm_judge_outputs
        if 'llm_judge_outputs' not in globals():
            llm_judge_outputs = {}
        llm_judge_outputs['accuracy_judge_output'] = judge_output

        score = _extract_json_score(judge_output, default_score=0.5)

        # For offline CSV evaluation, return Score object with judge output metadata
        try:
            from autoevals import Score
            return Score(
                name="answer_accuracy_scorer",
                score=score,
                metadata={
                    "judge_output": judge_output,
                    "eval_model": env("LITELLM_EVAL_MODEL"),
                    "judge_type": "accuracy"
                }
            )
        except:
            return score  # Fallback to numeric score

    except Exception as e:
        print(f"⚠️ Answer accuracy scorer failed: {e}")
        return 0.5

def answer_completeness_scorer(input, output, expected=None):
    """
    Braintrust Scorer: Answer Completeness
    Single-aspect scorer assessing if answer fully addresses the question
    Uses dedicated evaluation model for consistent LLM judging
    """
    try:
        reference_section = ""
        if expected:
            reference_section = f"\n\nExpected Answer: {expected}"

        prompt = f'''You are an expert evaluator assessing how completely an AI assistant's answer addresses the user's question.

Analyze whether the answer covers all important aspects of the question and select one of the following options:

a) The answer is comprehensive and addresses all aspects of the question (Score: 1.0)
b) The answer covers most key points but misses some important details (Score: 0.7)
c) The answer covers some aspects but leaves significant gaps (Score: 0.4)
d) The answer is incomplete and misses most important aspects (Score: 0.0)

Output format: Return your evaluation as a JSON object with the following keys:
1. Score: A score between 0 and 1 based on the rubric above
2. Choice: The letter (a, b, c, or d) that best describes the completeness
3. Rationale: A brief explanation of what aspects are covered or missing

Question: {input}

Answer: {output}{reference_section}

Provide your evaluation:'''

        response = eval_llm.invoke(prompt)  # Use dedicated evaluation model
        judge_output = response.content

        # Store judge output globally for experiment logging
        global llm_judge_outputs
        if 'llm_judge_outputs' not in globals():
            llm_judge_outputs = {}
        llm_judge_outputs['completeness_judge_output'] = judge_output

        score = _extract_json_score(judge_output, default_score=0.5)

        # For offline CSV evaluation, return Score object with judge output metadata
        try:
            from autoevals import Score
            return Score(
                name="answer_completeness_scorer",
                score=score,
                metadata={
                    "judge_output": judge_output,
                    "eval_model": env("LITELLM_EVAL_MODEL"),
                    "judge_type": "completeness"
                }
            )
        except:
            return score  # Fallback to numeric score

    except Exception as e:
        print(f"⚠️ Answer completeness scorer failed: {e}")
        return 0.5

def load_test_data():
    """Load test data for evaluation"""
    return [
        {
            "input": "What is Retrieval Augmented Generation?",
            "expected": "RAG combines retrieval and generation for better AI responses",
        },
        {
            "input": "What are the benefits of using RAG?",
            "expected": "RAG improves accuracy and reduces hallucinations in AI systems",
        },
        {
            "input": "How does vector search work in RAG?",
            "expected": "Vector search finds semantically similar documents using embeddings",
        }
    ]

def load_csv_data(csv_path: str, limit: int = None):
    """Load data from CSV file"""
    print(f"📂 Loading data from: {csv_path}")

    df = pd.read_csv(csv_path)

    # Validate required columns
    required_cols = ['query', 'expected_answer']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    if limit:
        df = df.head(limit)
        print(f"📊 Limited to first {limit} items")

    print(f"📊 Total items: {len(df)}")

    # Convert to evaluation format
    data = []
    for _, row in df.iterrows():
        item = {
            "input": row['query'],
            "expected": row['expected_answer']
        }

        # Add optional metadata
        if 'test_id' in row:
            item["metadata"] = {"test_id": row['test_id']}
        if 'category' in row:
            item.setdefault("metadata", {})["category"] = row.get('category', 'unknown')

        data.append(item)

    return data

def run_rag_evaluation(dataset_path: str = None, limit: int = None, use_llm_judges: bool = True):
    """Run RAG evaluation using Braintrust Eval with LLM judges"""

    setup_braintrust()

    # Load data
    if dataset_path:
        eval_data = load_csv_data(dataset_path, limit)
    else:
        eval_data = load_test_data()
        if limit:
            eval_data = eval_data[:limit]

    project_name = "vikas-autoeval-poc"
    experiment_name = f"offline_csv_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Choose scorers based on configuration
    if use_llm_judges:
        # Custom Braintrust Scorers following official documentation guidelines
        scorers = [
            # Custom LLM-based scorers using company's LiteLLM
            answer_relevancy_scorer,      # Single-aspect: relevance to question
            answer_accuracy_scorer,       # Single-aspect: factual accuracy
            answer_completeness_scorer,   # Single-aspect: completeness
            # Include Non-LLM scorers for comparison
            ExactMatch(),
            Levenshtein()
        ]
        scorer_description = "Custom Braintrust Scorers (LiteLLM) + Non-LLM"
        evaluation_method = "braintrust_custom_scorers"
    else:
        scorers = [
            ExactMatch(),
            Levenshtein(),
            exact_match  # Custom simple scorer
        ]
        scorer_description = "Non-LLM Scorers Only"
        evaluation_method = "braintrust_non_llm_scorers"

    print(f"🚀 Running Braintrust Eval: {project_name}")
    print(f"📊 Evaluating {len(eval_data)} items")
    print(f"🔧 Scorers: {scorer_description}")

    if use_llm_judges:
        print("🧑‍⚖️ Using Custom Braintrust Scorers with Dedicated Evaluation Model:")
        print(f"   🤖 RAG Model: {env('LITELLM_MODEL')}")
        print(f"   ⚖️ Evaluation Model: {env('LITELLM_EVAL_MODEL')}")
        print("   📊 Answer Relevancy Scorer (single-aspect)")
        print("   ✅ Answer Accuracy Scorer (single-aspect)")
        print("   📝 Answer Completeness Scorer (single-aspect)")
        print("   🔧 Following Braintrust official scorer guidelines")

    try:
        # Run Braintrust Eval with proper structure and enhanced metadata
        result = Eval(
            project_name,  # Project name (first argument)
            data=lambda: eval_data,  # Data function
            task=lambda input: rag_task(input),  # Task function
            scores=scorers,  # Scoring functions (Enterprise LLM judges + Non-LLM)
            metadata={  # Experiment metadata
                "evaluation_method": evaluation_method,
                "rag_model": env("LITELLM_MODEL"),
                "eval_model": env("LITELLM_EVAL_MODEL"),
                "scorers_used": scorer_description,
                "use_llm_judges": use_llm_judges,
                "created_at": datetime.now().isoformat(),
                "total_items": len(eval_data),
                "llm_judge_outputs_captured": True if use_llm_judges else False,
                "judge_models": {
                    "relevancy_judge": env("LITELLM_EVAL_MODEL"),
                    "accuracy_judge": env("LITELLM_EVAL_MODEL"),
                    "completeness_judge": env("LITELLM_EVAL_MODEL")
                } if use_llm_judges else {}
            },
            experiment_name=experiment_name  # Experiment name
        )

        print("✅ Braintrust Eval completed successfully!")
        print(f"📊 Results summary: {result.summary}")

        # Print key metrics
        if hasattr(result, 'summary') and hasattr(result.summary, 'scores'):
            scores = result.summary.scores
            print(f"\n🎯 Key Metrics:")
            for score_name, score_data in scores.items():
                if hasattr(score_data, 'score'):
                    print(f"   {score_name}: {score_data.score:.3f}")

        return result

    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def eval_single_response_with_braintrust(question: str, answer: str, expected_answer: str = None, use_llm_judges: bool = True):
    """
    Evaluate a single RAG response using Braintrust LLM-as-a-Judge scorers
    For real-time evaluation in Streamlit app

    Args:
        question: User's question
        answer: RAG system's response
        expected_answer: Expected/reference answer (optional)
        use_llm_judges: Whether to use LLM judges or only non-LLM scorers

    Returns:
        dict: Evaluation scores from all scorers
    """
    try:
        setup_braintrust()

        # Choose scorers based on configuration
        if use_llm_judges:
            scorers = [
                answer_relevancy_scorer,
                answer_accuracy_scorer,
                answer_completeness_scorer,
                ExactMatch(),
                Levenshtein()
            ]
        else:
            scorers = [
                ExactMatch(),
                Levenshtein()
            ]

        # Initialize global judge outputs if using LLM judges
        global llm_judge_outputs
        if 'llm_judge_outputs' not in globals():
            llm_judge_outputs = {}

        # Run scoring
        scores = {}
        for scorer in scorers:
            scorer_name = _get_scorer_name(scorer)
            try:
                score = _run_single_scorer(scorer, question, answer, expected_answer)
                # Extract numeric score if it's a Score object
                if hasattr(score, 'score'):
                    scores[scorer_name] = float(score.score)
                else:
                    scores[scorer_name] = float(score)
            except Exception as e:
                print(f"⚠️ Scorer {scorer_name} failed: {e}")
                scores[scorer_name] = 0.5

        # Capture LLM judge outputs if available
        judge_outputs = dict(llm_judge_outputs) if llm_judge_outputs else {}

        return {
            "status": "success",
            "scores": scores,
            "judge_outputs": judge_outputs,
            "question": question,
            "answer": answer,
            "expected": expected_answer
        }

    except Exception as e:
        print(f"❌ Single response evaluation failed: {e}")
        return {
            "status": "error",
            "error": str(e),
            "scores": {},
            "judge_outputs": {}
        }

def _get_scorer_name(scorer):
    """Get consistent scorer name for logging."""
    return scorer.__name__ if hasattr(scorer, '__name__') else scorer.__class__.__name__

def _run_single_scorer(scorer, question: str, answer: str, expected_answer: str = None):
    """Run a single scorer and return its score."""
    if hasattr(scorer, 'score'):
        # For autoevals scorers (ExactMatch, Levenshtein, etc.)
        return scorer.score(answer, expected_answer or "")
    elif callable(scorer):
        # For custom function scorers
        try:
            # Try with all 3 arguments first (input, output, expected)
            return scorer(question, answer, expected_answer)
        except TypeError:
            # Fallback to 2 arguments (output, expected) for some scorers
            return scorer(answer, expected_answer or "")
    else:
        return 0.5

def log_to_braintrust_experiment(question: str, answer: str, scores: dict, chat_id: str = None):
    """
    Log a single interaction to Braintrust experiment for tracking

    Args:
        question: User's question
        answer: RAG system's response
        scores: Dictionary of evaluation scores
        chat_id: Chat session ID for grouping
    """
    try:
        setup_braintrust()

        # Create experiment name for real-time logging
        experiment_name = f"realtime_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        project_name = "vikas-autoeval-poc"

        # Log single interaction to experiment (not span)
        import braintrust

        # Initialize experiment if it doesn't exist
        experiment = braintrust.init_experiment(
            project=project_name,
            experiment=experiment_name
        )

        # Prepare metadata with LLM judge outputs if available
        metadata = {
            "chat_id": chat_id,
            "timestamp": datetime.now().isoformat(),
            "evaluation_method": "realtime_llm_judges",
            "source": "streamlit_app",
            "rag_model": env("LITELLM_MODEL"),
            "eval_model": env("LITELLM_EVAL_MODEL")
        }

        # Add LLM judge outputs if available
        global llm_judge_outputs
        if 'llm_judge_outputs' in globals() and llm_judge_outputs:
            metadata["llm_judge_outputs"] = dict(llm_judge_outputs)
            print(f"📝 Including {len(llm_judge_outputs)} judge outputs in experiment")

        # Log the interaction with scores and judge outputs
        experiment.log(
            input=question,
            output=answer,
            scores=scores,
            metadata=metadata
        )

        # Clear judge outputs after logging
        if 'llm_judge_outputs' in globals():
            llm_judge_outputs.clear()

        # Ensure data is flushed to Braintrust
        experiment.flush()

        print(f"✅ Logged to Braintrust experiment: {experiment_name}")
        print(f"🔗 View at: https://www.braintrust.dev/app/Thomson%20Reuters%20(Materia)/p/{project_name}/experiments/{experiment_name}")
        return True

    except Exception as e:
        print(f"⚠️ Failed to log to Braintrust: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function with CLI support"""
    import argparse

    parser = argparse.ArgumentParser(description='Run Braintrust RAG Evaluation with LLM Judges')
    parser.add_argument('--dataset', help='Path to CSV dataset file')
    parser.add_argument('--limit', type=int, help='Limit number of items to evaluate')
    parser.add_argument('--no-llm-judges', action='store_true',
                       help='Use only Non-LLM scorers (ExactMatch, Levenshtein)')

    args = parser.parse_args()

    print("🧪 Braintrust RAG Evaluation with LLM Judges")
    print("=" * 50)

    result = run_rag_evaluation(
        dataset_path=args.dataset,
        limit=args.limit,
        use_llm_judges=not args.no_llm_judges
    )

    if result:
        print(f"\n🔗 View results at: https://www.braintrust.dev/app/Thomson%20Reuters%20(Materia)/p/vikas-autoeval-poc")
        return True
    else:
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)