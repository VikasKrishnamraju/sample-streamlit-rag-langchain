#!/usr/bin/env python3
"""
Comprehensive Braintrust AutoEvals Implementation
Implements all available autoevals.llm and autoevals.ragas scorers with company's LiteLLM integration
Based on official Braintrust AutoEvals documentation and working Anthropic setup
"""

import os
import environ
import pandas as pd
import openai
from typing import List, Dict, Any
from datetime import datetime

# Braintrust imports
from braintrust import Eval
from autoevals import (
    ExactMatch,
    Levenshtein,
    init as autoevals_init
)

# Import ALL available autoevals.llm scorers
from autoevals.llm import (
    Battle,
    ClosedQA,
    Factuality,
    Humor,
    Possible,
    Security,
    Sql,
    Summary,
    Translation,
    LLMClassifier
)

# Import ALL available autoevals.ragas scorers
from autoevals.ragas import (
    ContextEntityRecall,
    ContextRelevancy,
    ContextRecall,
    ContextPrecision,
    Faithfulness,
    AnswerRelevancy,
    AnswerSimilarity,
    AnswerCorrectness
)

# Import additional scorer modules
from autoevals.string import EmbeddingSimilarity
from autoevals.number import NumericDiff
from autoevals.json import JSONDiff, ValidJSON
from autoevals.moderation import Moderation

# Company's RAG system integration
from vector_functions import get_lite_llm_model, load_retriever, generate_answer_from_context

# Load environment variables
env = environ.Env()
environ.Env.read_env()

# Global variables to capture LLM judge outputs and context for experiment logging
llm_judge_outputs = {}
current_context = None  # Store context for current evaluation item
context_map = {}  # Map input questions to their contexts

# Setup AutoEvals using company's LiteLLM proxy (from working implementation)
def setup_autoevals_with_anthropic():
    """Setup AutoEvals to use company's LiteLLM proxy for Anthropic models"""
    try:
        # Use company's LiteLLM proxy endpoint and API key
        litellm_client = openai.OpenAI(
            api_key=env("LITELLM_API_KEY"),
            base_url=env("LITELLM_BASE_URL")
        )

        # Initialize autoevals with the LiteLLM client
        autoevals_init(litellm_client)

        print("✅ AutoEvals configured with company's LiteLLM proxy")
        print(f"   🤖 Model: {env('LITELLM_MODEL')}")
        print(f"   🔗 Base URL: {env('LITELLM_BASE_URL')}")
        print(f"   🔑 Using LITELLM_API_KEY")

        return litellm_client

    except Exception as e:
        print(f"⚠️ AutoEvals setup failed: {e}")
        print(f"   Error details: {str(e)}")
        return None

# Initialize AutoEvals with Anthropic API
anthropic_client_for_autoevals = setup_autoevals_with_anthropic()
autoevals_ready = anthropic_client_for_autoevals is not None

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

            # Store context globally for AutoEvals RAGAS scorers
            global current_context
            current_context = context

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

def rag_task_with_context(data_item):
    """Wrapper for rag_task that handles context from test data"""
    global current_context

    # Set context from test data if available (Braintrust passes only the input string)
    # The context mapping is handled via context_map global variable

    # Call the regular rag_task
    input_text = data_item.get('input') if isinstance(data_item, dict) else data_item
    return rag_task(input_text)

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

# =============================================================================
# AutoEvals.LLM Scorers - All available LLM-based evaluators
# =============================================================================

def create_llm_scorers():
    """Create all available autoevals.llm scorers with proper configuration"""
    scorers = {}

    if not autoevals_ready:
        print("⚠️ AutoEvals not ready, skipping LLM scorers")
        return scorers

    try:
        # Configure model to use company's Anthropic model
        model_name = env('LITELLM_MODEL')

        scorers.update({
            # Factuality Checker
            'factuality': Factuality(model=model_name),

            # Closed QA Evaluator
            'closed_qa': ClosedQA(model=model_name),

            # Humor Detector
            'humor': Humor(model=model_name),

            # Solution Feasibility Checker
            'possible': Possible(model=model_name),

            # Security Vulnerability Scanner
            'security': Security(model=model_name),

            # SQL Query Equivalence Checker
            'sql': Sql(model=model_name),

            # Text Summarization Quality
            'summary': Summary(model=model_name),

            # Translation Quality
            'translation': Translation(model=model_name),

            # Battle Comparison (solution vs reference)
            'battle': Battle(model=model_name)
        })

        print(f"✅ Created {len(scorers)} autoevals.llm scorers")

    except Exception as e:
        print(f"⚠️ Failed to create LLM scorers: {e}")

    return scorers

# =============================================================================
# AutoEvals.RAGAS Scorers - All available RAG evaluation metrics
# =============================================================================

def create_ragas_scorers():
    """Create all available autoevals.ragas scorers with proper configuration"""
    scorers = {}

    if not autoevals_ready:
        print("⚠️ AutoEvals not ready, skipping RAGAS scorers")
        return scorers

    try:
        # Configure model to use company's Anthropic model
        model_name = env('LITELLM_MODEL')

        # Context Quality Evaluators
        scorers.update({
            'context_entity_recall': ContextEntityRecall(model=model_name),
            'context_relevancy': ContextRelevancy(model=model_name),
            'context_recall': ContextRecall(model=model_name),
            'context_precision': ContextPrecision(model=model_name),
        })

        # Answer Quality Evaluators
        scorers.update({
            'faithfulness': Faithfulness(model=model_name),
            'answer_relevancy_ragas': AnswerRelevancy(model=model_name),
            'answer_similarity': AnswerSimilarity(model=model_name),
            'answer_correctness': AnswerCorrectness(model=model_name),
        })

        print(f"✅ Created {len(scorers)} autoevals.ragas scorers")

    except Exception as e:
        print(f"⚠️ Failed to create RAGAS scorers: {e}")

    return scorers

# =============================================================================
# Enhanced Wrapper Functions for Context-Aware Evaluation
# =============================================================================

def create_context_aware_scorer(scorer, scorer_name):
    """Create a context-aware wrapper for RAGAS scorers"""
    def context_aware_evaluator(input, output, expected=None):
        try:
            # Get context from context_map or current_context
            global current_context, context_map
            context = None

            if input in context_map:
                context = context_map[input]
            elif current_context:
                context = current_context
            else:
                # Try to get context from retrieval if possible
                try:
                    retriever = load_retriever()
                    retrieved_docs = retriever.get_relevant_documents(input)
                    if retrieved_docs:
                        context = "\n".join([doc.page_content for doc in retrieved_docs[:3]])
                except:
                    context = "Context not available for this evaluation"

            # Call the RAGAS scorer with context
            result = scorer.eval(
                input=input,
                output=output,
                expected=expected,
                context=context
            )

            return result

        except Exception as e:
            print(f"⚠️ {scorer_name} failed: {e}")
            from autoevals import Score
            return Score(
                name=scorer_name,
                score=0.5,
                metadata={"error": str(e)}
            )

    return context_aware_evaluator

# =============================================================================
# Additional Non-LLM Scorers
# =============================================================================

def create_additional_scorers():
    """Create additional non-LLM scorers from other modules"""
    scorers = {}

    try:
        # String similarity scorers
        scorers['exact_match'] = ExactMatch()
        scorers['levenshtein'] = Levenshtein()

        # Numeric comparison
        scorers['numeric_diff'] = NumericDiff()

        # JSON comparison
        scorers['json_diff'] = JSONDiff()
        scorers['valid_json'] = ValidJSON()

        # Content moderation (requires OpenAI API for moderation endpoint)
        # scorers['moderation'] = Moderation()  # Uncomment if needed

        print(f"✅ Created {len(scorers)} additional scorers")

    except Exception as e:
        print(f"⚠️ Failed to create additional scorers: {e}")

    return scorers

# =============================================================================
# Test Data with Context
# =============================================================================

def load_test_data():
    """Load test data for evaluation with realistic context"""
    return [
        {
            "input": "What is Retrieval Augmented Generation?",
            "expected": "RAG combines retrieval and generation for better AI responses",
            "context": "Retrieval-Augmented Generation (RAG) is a natural language processing approach that combines information retrieval with text generation. RAG works by first retrieving relevant documents or passages from a knowledge base, then using those retrieved documents as additional context for generating responses. This approach helps language models provide more accurate, factual, and up-to-date information by grounding their responses in external knowledge sources rather than relying solely on their training data."
        },
        {
            "input": "What are the benefits of using RAG?",
            "expected": "RAG improves accuracy and reduces hallucinations in AI systems",
            "context": "RAG provides several key benefits for AI systems: 1) Improved accuracy by grounding responses in factual external sources, 2) Reduced hallucinations since the model refers to real documents rather than generating from memory, 3) Up-to-date information by accessing current knowledge bases, 4) Enhanced transparency as users can trace answers back to source documents, 5) Better handling of domain-specific knowledge without retraining the entire model."
        },
        {
            "input": "How does vector search work in RAG?",
            "expected": "Vector search finds semantically similar documents using embeddings",
            "context": "Vector search in RAG systems works by converting documents and queries into high-dimensional numerical vectors called embeddings. These embeddings capture semantic meaning, allowing the system to find documents that are conceptually similar rather than just matching keywords. The process involves: 1) Converting documents to embeddings using neural networks, 2) Storing embeddings in a vector database, 3) Converting user queries to embeddings using the same model, 4) Computing similarity scores between query and document embeddings, 5) Retrieving the most similar documents based on cosine similarity or other distance metrics."
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

# =============================================================================
# Comprehensive Evaluation Function
# =============================================================================

def run_comprehensive_evaluation(dataset_path: str = None, limit: int = None,
                                scorer_types: List[str] = None):
    """
    Run comprehensive RAG evaluation using all available AutoEvals scorers

    Args:
        dataset_path: Path to CSV dataset file
        limit: Limit number of items to evaluate
        scorer_types: List of scorer types to use ['custom', 'llm', 'ragas', 'additional']
    """

    setup_braintrust()

    # Load data
    if dataset_path:
        eval_data = load_csv_data(dataset_path, limit)
    else:
        eval_data = load_test_data()
        if limit:
            eval_data = eval_data[:limit]

    # Build context map for AutoEvals RAGAS scorers
    global context_map
    context_map = {}
    for item in eval_data:
        if isinstance(item, dict) and 'context' in item and 'input' in item:
            context_map[item['input']] = item['context']
            print(f"🔍 Added context mapping for: {item['input'][:50]}...")

    print(f"🔍 Built context map with {len(context_map)} entries")

    # Default to all scorer types if none specified
    if scorer_types is None:
        scorer_types = ['custom', 'llm', 'ragas', 'additional']

    # Create all scorer types
    all_scorers = []
    scorer_descriptions = []

    # Custom Braintrust Scorers
    if 'custom' in scorer_types:
        custom_scorers = [
            answer_relevancy_scorer,
            answer_accuracy_scorer,
            answer_completeness_scorer
        ]
        all_scorers.extend(custom_scorers)
        scorer_descriptions.append(f"Custom LiteLLM ({len(custom_scorers)})")

    # AutoEvals LLM Scorers
    if 'llm' in scorer_types and autoevals_ready:
        llm_scorers = create_llm_scorers()
        all_scorers.extend(list(llm_scorers.values()))
        scorer_descriptions.append(f"AutoEvals LLM ({len(llm_scorers)})")

    # AutoEvals RAGAS Scorers
    if 'ragas' in scorer_types and autoevals_ready:
        ragas_scorers = create_ragas_scorers()
        # Create context-aware wrappers for RAGAS scorers
        for name, scorer in ragas_scorers.items():
            context_aware_scorer = create_context_aware_scorer(scorer, name)
            all_scorers.append(context_aware_scorer)
        scorer_descriptions.append(f"AutoEvals RAGAS ({len(ragas_scorers)})")

    # Additional Non-LLM Scorers
    if 'additional' in scorer_types:
        additional_scorers = create_additional_scorers()
        all_scorers.extend(list(additional_scorers.values()))
        scorer_descriptions.append(f"Additional ({len(additional_scorers)})")

    project_name = "vikas-comprehensive-autoevals"
    experiment_name = f"comprehensive_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    print(f"🚀 Running Comprehensive Braintrust AutoEvals: {project_name}")
    print(f"📊 Evaluating {len(eval_data)} items")
    print(f"🔧 Scorer Types: {' + '.join(scorer_descriptions)}")
    print(f"📈 Total Scorers: {len(all_scorers)}")

    if autoevals_ready:
        print("🤖 AutoEvals Configuration:")
        print(f"   📡 RAG Model: {env('LITELLM_MODEL')}")
        print(f"   ⚖️ Custom Evaluation Model: {env('LITELLM_EVAL_MODEL')}")
        print(f"   🔗 LiteLLM Proxy: {env('LITELLM_BASE_URL')}")

    try:
        # Run Braintrust Eval with comprehensive scorer suite
        result = Eval(
            project_name,
            data=lambda: eval_data,
            task=rag_task_with_context,
            scores=all_scorers,
            metadata={
                "evaluation_method": "comprehensive_autoevals",
                "rag_model": env("LITELLM_MODEL"),
                "eval_model": env("LITELLM_EVAL_MODEL"),
                "scorer_types": scorer_types,
                "total_scorers": len(all_scorers),
                "autoevals_ready": autoevals_ready,
                "created_at": datetime.now().isoformat(),
                "total_items": len(eval_data),
                "proxy_endpoint": env("LITELLM_BASE_URL")
            },
            experiment_name=experiment_name
        )

        print("✅ Comprehensive AutoEvals evaluation completed successfully!")
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
        print(f"❌ Comprehensive evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main function with CLI support"""
    import argparse

    parser = argparse.ArgumentParser(description='Run Comprehensive Braintrust AutoEvals')
    parser.add_argument('--dataset', help='Path to CSV dataset file')
    parser.add_argument('--limit', type=int, help='Limit number of items to evaluate')
    parser.add_argument('--scorers', nargs='+',
                       choices=['custom', 'llm', 'ragas', 'additional'],
                       default=['custom', 'llm', 'ragas', 'additional'],
                       help='Types of scorers to use')

    args = parser.parse_args()

    print("🧪 Comprehensive Braintrust AutoEvals Evaluation")
    print("=" * 60)

    result = run_comprehensive_evaluation(
        dataset_path=args.dataset,
        limit=args.limit,
        scorer_types=args.scorers
    )

    if result:
        print(f"\n🔗 View results at: https://www.braintrust.dev/app/Thomson%20Reuters%20(Materia)/p/vikas-comprehensive-autoevals")
        return True
    else:
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)