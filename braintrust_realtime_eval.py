"""
Braintrust Real-time Evaluation using Eval() Framework
======================================================

This module provides real-time RAG evaluation using Braintrust's Eval() framework
to create individual traces for each evaluation (same structure as offline CSV evaluations).

Key differences from braintrust_eval_llm.py:
- Uses Eval() framework for real-time evaluations instead of experiment.log()
- Creates individual traces for each evaluation item
- Maintains same metadata structure as offline evaluations
- Compatible with Streamlit integration

Author: Enhanced for dual model architecture with judge outputs
"""

import os
import json
import ssl
import certifi
from datetime import datetime
from typing import Dict, Any, Optional

# Braintrust and Evaluation Dependencies
from braintrust import Eval
import braintrust

# LangChain and LLM Dependencies
from langchain.prompts import ChatPromptTemplate
import environ

# AutoEvals Scorers (for Score objects)
# Note: ExactMatch and Levenshtein not needed for real-time evaluation
# as they require expected values which are unavailable in real-time chats

# Company's RAG system integration
from vector_functions import get_lite_llm_model, load_retriever, generate_answer_from_context

# Import the working evaluation model function from the main file
from braintrust_eval_llm import get_eval_llm_model, setup_braintrust

# Load environment variables
env = environ.Env()
environ.Env.read_env()

# Global variable to capture LLM judge outputs for experiment logging
llm_judge_outputs = {}

# Initialize models once using the same working function
eval_llm = get_eval_llm_model()

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
    Braintrust Scorer: Answer Relevancy for Real-time Evaluation
    Single-aspect scorer following Braintrust guidelines
    Uses dedicated evaluation model for consistent LLM judging
    Returns Score object with judge output metadata
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

        # Return Score object with judge output metadata
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

    except Exception as e:
        print(f"Answer relevancy scorer failed: {e}")
        from autoevals import Score
        return Score(
            name="answer_relevancy_scorer",
            score=0.5,
            metadata={"error": str(e)}
        )

def answer_accuracy_scorer(input, output, expected=None):
    """
    Braintrust Scorer: Answer Accuracy for Real-time Evaluation
    Single-aspect scorer following Braintrust guidelines
    Uses dedicated evaluation model for consistent LLM judging
    Returns Score object with judge output metadata
    """
    try:
        prompt = f'''You are an expert evaluator assessing the factual accuracy of an AI assistant's answer.

Analyze the answer for factual correctness, then select one of the following options:
a) The answer is completely accurate with no factual errors (Score: 1.0)
b) The answer is mostly accurate with minor factual issues (Score: 0.7)
c) The answer has some accurate information but contains factual errors (Score: 0.4)
d) The answer contains significant factual errors or misinformation (Score: 0.0)

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
        llm_judge_outputs['accuracy_judge_output'] = judge_output

        score = _extract_json_score(judge_output, default_score=0.5)

        # Return Score object with judge output metadata
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

    except Exception as e:
        print(f"Answer accuracy scorer failed: {e}")
        from autoevals import Score
        return Score(
            name="answer_accuracy_scorer",
            score=0.5,
            metadata={"error": str(e)}
        )

def answer_completeness_scorer(input, output, expected=None):
    """
    Braintrust Scorer: Answer Completeness for Real-time Evaluation
    Single-aspect scorer following Braintrust guidelines
    Uses dedicated evaluation model for consistent LLM judging
    Returns Score object with judge output metadata
    """
    try:
        prompt = f'''You are an expert evaluator assessing how completely an AI assistant's answer addresses the user's question.

Analyze the completeness of the answer, then select one of the following options:
a) The answer completely addresses all aspects of the question (Score: 1.0)
b) The answer addresses most aspects but misses some minor details (Score: 0.7)
c) The answer addresses some aspects but leaves significant gaps (Score: 0.4)
d) The answer barely addresses the question or is very incomplete (Score: 0.0)

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
        llm_judge_outputs['completeness_judge_output'] = judge_output

        score = _extract_json_score(judge_output, default_score=0.5)

        # Return Score object with judge output metadata
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

    except Exception as e:
        print(f"Answer completeness scorer failed: {e}")
        from autoevals import Score
        return Score(
            name="answer_completeness_scorer",
            score=0.5,
            metadata={"error": str(e)}
        )

def rag_task_realtime(input_data) -> str:
    """
    Real-time RAG task function for single question evaluation
    Compatible with Braintrust Eval() framework
    """
    try:
        # Handle both dict and string inputs
        if isinstance(input_data, dict):
            question = input_data.get('input', input_data.get('question', str(input_data)))
            collection_name = input_data.get('collection_name', None)
        else:
            question = str(input_data)
            collection_name = None

        # Generate RAG response using company's system
        if collection_name:
            retriever = load_retriever(collection_name)
            response = generate_answer_from_context(retriever, question)
        else:
            # Direct LLM call for general questions
            llm = get_lite_llm_model()
            response = llm.invoke(question).content

        return response

    except Exception as e:
        print(f"RAG task failed: {e}")
        return "Sorry, I encountered an error processing your question."

def eval_single_realtime_with_braintrust_eval(question: str, answer: str = None, collection_name: str = None, chat_id: str = None) -> Dict[str, Any]:
    """
    Evaluate a single real-time interaction using Braintrust Eval() framework
    Creates individual traces similar to offline CSV evaluation

    Args:
        question: User's question
        answer: Pre-generated answer (optional, will generate if not provided)
        collection_name: Vector collection for RAG (optional)
        chat_id: Chat session ID for grouping

    Returns:
        dict: Evaluation results with experiment URL
    """
    try:
        setup_braintrust()

        # Create experiment name for real-time logging
        experiment_name = f"realtime_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        project_name = "vikas-autoeval-poc"

        # Prepare evaluation data (Braintrust requires 'input' field)
        eval_data = [{
            "input": question,  # Required by Braintrust
            "collection_name": collection_name,
            "chat_id": chat_id
        }]

        # If answer is provided, use direct evaluation; otherwise use RAG task
        if answer:
            def direct_task(input_data):
                return answer
            task_function = direct_task
        else:
            task_function = rag_task_realtime

        # Set up scorers for real-time evaluation (only LLM judges - no reference answer needed)
        scorers = [
            answer_relevancy_scorer,
            answer_accuracy_scorer,
            answer_completeness_scorer,
            # Note: ExactMatch and Levenshtein excluded - they require expected values
            # which are not available in real-time chat interactions
        ]

        print(f"🔄 Running real-time Braintrust Eval: {experiment_name}")
        print(f"📊 Using {len(scorers)} scorers with dual model architecture")
        print(f"🤖 RAG Model: {env('LITELLM_MODEL')}")
        print(f"⚖️ Evaluation Model: {env('LITELLM_EVAL_MODEL')}")

        # Run Braintrust Eval with proper structure
        result = Eval(
            project_name,  # Project name (first argument)
            data=lambda: eval_data,  # Data function
            task=task_function,  # Task function
            scores=scorers,  # Scoring functions (LLM judges + Non-LLM)
            metadata={  # Experiment metadata
                "evaluation_method": "realtime_llm_judges",
                "rag_model": env("LITELLM_MODEL"),
                "eval_model": env("LITELLM_EVAL_MODEL"),
                "chat_id": chat_id,
                "use_llm_judges": True,
                "created_at": datetime.now().isoformat(),
                "judge_models": {
                    "relevancy_judge": env("LITELLM_EVAL_MODEL"),
                    "accuracy_judge": env("LITELLM_EVAL_MODEL"),
                    "completeness_judge": env("LITELLM_EVAL_MODEL")
                },
                "source": "streamlit_realtime"
            },
            experiment_name=experiment_name  # Experiment name
        )

        print(f"✅ Real-time Braintrust Eval completed successfully!")

        # Extract results
        experiment_url = f"https://www.braintrust.dev/app/Thomson%20Reuters%20(Materia)/p/{project_name}/experiments/{experiment_name}"

        # Get scores from result
        scores = {}
        if hasattr(result, 'summary') and hasattr(result.summary, 'scores'):
            for score_name, score_data in result.summary.scores.items():
                if hasattr(score_data, 'score'):
                    scores[score_name] = score_data.score

        return {
            "status": "success",
            "experiment_name": experiment_name,
            "experiment_url": experiment_url,
            "scores": scores,
            "judge_outputs": dict(llm_judge_outputs) if llm_judge_outputs else {},
            "question": question,
            "answer": answer or "Generated by RAG task"
        }

    except Exception as e:
        print(f"❌ Real-time evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            "status": "error",
            "error": str(e),
            "scores": {},
            "judge_outputs": {}
        }

def test_realtime_eval():
    """Test function for the real-time evaluation"""
    print("🧪 Testing Real-time Braintrust Eval Framework")
    print("=" * 50)

    result = eval_single_realtime_with_braintrust_eval(
        question="What is artificial intelligence?",
        answer="Artificial intelligence (AI) is a branch of computer science focused on creating systems that can perform tasks typically requiring human intelligence.",
        chat_id="test_realtime_eval"
    )

    if result["status"] == "success":
        print(f"✅ Test successful!")
        print(f"📊 Scores: {result['scores']}")
        print(f"🔗 View at: {result['experiment_url']}")
        print(f"📝 Judge outputs: {len(result['judge_outputs'])} judges")
    else:
        print(f"❌ Test failed: {result['error']}")

# Streamlit Integration Functions
# =================================

def eval_and_log_realtime_response(question: str, answer: str, chat_id: str = None) -> Dict[str, Any]:
    """
    Simple wrapper for Streamlit integration
    Evaluates a question-answer pair using Braintrust Eval() framework

    Args:
        question: User's question
        answer: RAG system's response
        chat_id: Chat session ID

    Returns:
        dict: Evaluation results with scores and experiment info
    """
    return eval_single_realtime_with_braintrust_eval(
        question=question,
        answer=answer,
        chat_id=chat_id
    )

if __name__ == "__main__":
    test_realtime_eval()