# Braintrust LLM-as-a-Judge Evaluation System

## 📋 Overview

This document provides comprehensive documentation for the Braintrust evaluation system implemented in this RAG application. The system uses custom LLM-as-a-Judge scorers to evaluate RAG response quality in real-time and batch processing modes.

## 🏗️ Architecture

### System Components

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           BRAINTRUST EVALUATION SYSTEM                          │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌─────────────────┐    ┌──────────────────┐    ┌─────────────────────────────┐  │
│  │  Streamlit UI   │ -> │  RAG System      │ -> │  LLM-as-a-Judge Evaluation │  │
│  │  (chats.py)     │    │ (vector_funcs.py)│    │  (braintrust_eval_llm.py)  │  │
│  └─────────────────┘    └──────────────────┘    └─────────────────────────────┘  │
│           │                       │                            │                 │
│           v                       v                            v                 │
│  ┌─────────────────┐    ┌──────────────────┐    ┌─────────────────────────────┐  │
│  │   User Input    │    │   RAG Response   │    │    Evaluation Scores       │  │
│  │   Questions     │    │   Generation     │    │    - Relevancy             │  │
│  └─────────────────┘    └──────────────────┘    │    - Accuracy              │  │
│                                                 │    - Completeness          │  │
│                                                 │    - ExactMatch            │  │
│                                                 │    - Levenshtein           │  │
│                                                 └─────────────────────────────┘  │
│                                                            │                     │
│                                                            v                     │
│                                               ┌─────────────────────────────┐    │
│                                               │    Braintrust Dashboard     │    │
│                                               │    - Experiments            │    │
│                                               │    - Score Tracking         │    │
│                                               │    - Performance Analytics  │    │
│                                               └─────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## 📁 File Structure and Responsibilities

### Core Files

#### 1. `braintrust_eval_llm.py` - Main Evaluation Engine
**Purpose**: Core evaluation system with custom LLM judges and Braintrust integration

**Key Components**:
- **Custom LLM Scorers**: Three single-aspect judges following Braintrust guidelines
- **Non-LLM Scorers**: ExactMatch and Levenshtein for baseline comparison
- **Batch Evaluation**: Full dataset evaluation with CSV support
- **Real-time Evaluation**: Single response evaluation for live interactions
- **Braintrust Logging**: Experiment tracking and score persistence

**Main Functions**:
```python
# Batch evaluation for datasets
run_rag_evaluation(dataset_path, limit, use_llm_judges)

# Single response evaluation for real-time
eval_single_response_with_braintrust(question, answer, expected_answer, use_llm_judges)

# Experiment logging
log_to_braintrust_experiment(question, answer, scores, chat_id)

# RAG answer generation
rag_task(input_text, collection_name)
```

#### 2. `chats.py` - Streamlit Integration
**Purpose**: User interface with real-time evaluation integration

**Key Integration Points**:
- **Evaluation Trigger**: After RAG response generation
- **Score Display**: Sidebar metrics for user visibility
- **Session Management**: Score persistence across chat sessions

#### 3. `vector_functions.py` - RAG System
**Purpose**: Document retrieval and answer generation using company's LiteLLM

**Key Functions**:
```python
# LLM initialization
get_lite_llm_model()

# Document processing
load_document(file_path)
create_collection(collection_name, documents)

# RAG pipeline
generate_answer_from_context(retriever, question)
```

## 🧑‍⚖️ LLM-as-a-Judge Implementation

### Custom Braintrust Scorers

Following official Braintrust scorer guidelines, we implemented three single-aspect LLM judges:

#### 1. Answer Relevancy Scorer (`answer_relevancy_scorer`)
**Purpose**: Evaluates how relevant the RAG answer is to the user's question

**Evaluation Criteria**:
- a) Highly relevant and directly addresses the question (Score: 1.0)
- b) Mostly relevant but includes some off-topic content (Score: 0.7)
- c) Partially relevant but misses key aspects (Score: 0.4)
- d) Not relevant to the question (Score: 0.0)

**Implementation**:
```python
def answer_relevancy_scorer(input, output, expected=None):
    # Uses structured JSON prompt with clear rubric
    # Returns normalized score 0-1
    # Includes chain-of-thought reasoning
```

#### 2. Answer Accuracy Scorer (`answer_accuracy_scorer`)
**Purpose**: Evaluates factual correctness of the RAG answer

**Evaluation Criteria**:
- a) Completely accurate with no factual errors (Score: 1.0)
- b) Mostly accurate with minor factual issues (Score: 0.7)
- c) Significant factual errors but some correct information (Score: 0.3)
- d) Major factual errors or misinformation (Score: 0.0)

**Features**:
- Reference answer support when available
- Factual error identification
- Confidence-based scoring

#### 3. Answer Completeness Scorer (`answer_completeness_scorer`)
**Purpose**: Evaluates if the answer fully addresses all aspects of the question

**Evaluation Criteria**:
- a) Comprehensive and addresses all aspects (Score: 1.0)
- b) Covers most key points but misses some details (Score: 0.7)
- c) Covers some aspects but leaves significant gaps (Score: 0.4)
- d) Incomplete and misses most important aspects (Score: 0.0)

### Non-LLM Baseline Scorers

#### 1. ExactMatch Scorer
- **Purpose**: Binary exact string matching
- **Use Case**: Baseline comparison, factual accuracy validation
- **Score**: 1.0 for exact match, 0.0 otherwise

#### 2. Levenshtein Scorer
- **Purpose**: Edit distance similarity measurement
- **Use Case**: Fuzzy matching, semantic similarity proxy
- **Score**: Normalized edit distance (0.0-1.0)

## 🎯 Evaluation Modes

### 1. Real-time Evaluation (Streamlit Integration)

**Trigger**: Automatically after each RAG response in chat
**Function**: `eval_single_response_with_braintrust()`
**Output**:
- Scores displayed in Streamlit sidebar
- Logs sent to Braintrust experiments
- Console debug output

**Flow**:
```
User Question -> RAG Response -> LLM Judges -> Scores -> UI + Braintrust
```

### 2. Batch Evaluation (Dataset Processing)

**Trigger**: Command-line execution with dataset
**Function**: `run_rag_evaluation()`
**Input**: CSV files with columns: `query`, `expected_answer`
**Output**:
- Braintrust experiment with all results
- Aggregate performance metrics
- Detailed scoring breakdown

**Usage**:
```bash
# Evaluate full dataset
python braintrust_eval_llm.py --dataset test_golden_dataset.csv

# Evaluate subset with LLM judges
python braintrust_eval_llm.py --dataset test_golden_dataset.csv --limit 10

# Use only non-LLM scorers for speed
python braintrust_eval_llm.py --dataset test_golden_dataset.csv --no-llm-judges
```

## 🔧 Configuration and Setup

### Environment Variables (.env)
```bash
# Company LiteLLM Configuration
LITELLM_API_KEY='your-litellm-key'
LITELLM_BASE_URL='https://litellm.int.thomsonreuters.com'
LITELLM_MODEL='anthropic/claude-sonnet-4-20250514'

# Braintrust Configuration
BRAINTRUST_API_KEY='your-braintrust-key'
BRAINTRUST_API_URL='https://api.braintrust.dev'

# Enterprise SSL Certificates
SSL_CERT_FILE='corp-bundle-final-complete.pem'
REQUESTS_CA_BUNDLE='corp-bundle-final-complete.pem'
```

### Dependencies
```bash
# Core evaluation framework
pip install braintrust-core autoevals

# RAG system components
pip install langchain langchain-community langchain-chroma

# Company-specific integrations
pip install litellm_embeddings

# UI and utilities
pip install streamlit pandas environ
```

## 📊 Braintrust Integration

### Project Structure
- **Project Name**: `vikas-autoeval-poc`
- **Real-time Experiments**: Named `realtime_YYYYMMDD`
- **Batch Experiments**: Named `eval_YYYYMMDD_HHMMSS`

### Experiment Logging

**Data Structure**:
```json
{
  "input": "User's question",
  "output": "RAG system response",
  "scores": {
    "answer_relevancy_scorer": 0.9,
    "answer_accuracy_scorer": 0.8,
    "answer_completeness_scorer": 0.85,
    "ExactMatch": 0.0,
    "Levenshtein": 0.23
  },
  "metadata": {
    "chat_id": "chat_session_id",
    "timestamp": "2025-09-23T12:34:56",
    "evaluation_method": "realtime_llm_judges",
    "source": "streamlit_app"
  }
}
```

### Dashboard Access
**URL**: https://www.braintrust.dev/app/Thomson%20Reuters%20(Materia)/p/vikas-autoeval-poc

**Key Metrics Available**:
- Individual scorer performance over time
- Aggregate evaluation trends
- Response quality distribution
- Scorer correlation analysis

## 🚀 Usage Patterns

### For Development and Testing
1. **Quick Quality Check**: Use Streamlit app for individual responses
2. **Batch Validation**: Run evaluation on test datasets
3. **Performance Monitoring**: Check Braintrust dashboard regularly
4. **Scorer Tuning**: Adjust prompts based on results

### For Production Monitoring
1. **Real-time Quality Assurance**: Every user interaction gets evaluated
2. **Quality Degradation Detection**: Monitor score trends
3. **A/B Testing Support**: Compare different RAG configurations
4. **User Experience Optimization**: Identify low-quality responses

## 🛠️ Best Practices Implemented

### 1. Braintrust Scorer Guidelines Compliance
- **Single-aspect scorers**: Each judge evaluates one specific criterion
- **Clear scoring rubrics**: 4-point scales with explicit criteria descriptions
- **Structured JSON output**: Consistent score extraction and rationale
- **Chain-of-thought prompting**: Detailed reasoning before scoring

### 2. Code Organization
- **Separation of concerns**: UI, evaluation, and RAG logic separated
- **Error handling**: Graceful fallbacks and detailed logging
- **Modularity**: Easy to add/remove/modify scorers
- **Documentation**: Comprehensive inline documentation

### 3. Performance Optimization
- **Async scoring potential**: Architecture supports parallel evaluation
- **Efficient score extraction**: Centralized JSON parsing with regex fallback
- **Caching-ready**: Setup supports future caching implementations
- **Resource management**: Proper experiment cleanup and flushing

## ⚡ Performance Characteristics

### Evaluation Speed
- **Single Response**: ~3-5 seconds (3 LLM judges + 2 non-LLM scorers)
- **Batch Processing**: ~15-20 seconds per item (depends on LLM response time)
- **Non-LLM Only**: <1 second per response

### Accuracy and Reliability
- **Score Consistency**: High inter-evaluation reliability due to structured prompts
- **Error Handling**: Graceful degradation with default scores
- **Validation**: JSON schema validation with regex fallback

### Cost Considerations
- **LLM Usage**: 3 LLM calls per evaluation (relevancy, accuracy, completeness)
- **Token Efficiency**: Optimized prompts for minimal token usage
- **Cost Monitoring**: Tracked via company's LiteLLM infrastructure

## 🔍 Debugging and Troubleshooting

### Common Issues and Solutions

#### 1. Scores Not Appearing in Braintrust
**Symptoms**: Scores show in Streamlit sidebar but not in Braintrust dashboard
**Solution**: Check console logs for experiment logging success/failure
**Debug**: Look for "✅ Logged to Braintrust experiment" messages

#### 2. LLM Judge Failures
**Symptoms**: Default scores (0.5) appearing consistently
**Solution**: Check LiteLLM connectivity and API key validity
**Debug**: Enable detailed error logging in scorer functions

#### 3. SSL Certificate Issues
**Symptoms**: Connection errors to Braintrust API
**Solution**: Verify enterprise certificate bundle path and contents
**Debug**: Check certificate loading messages during startup

### Monitoring and Alerting
- **Score Distribution**: Monitor for unusual score patterns
- **Response Times**: Track evaluation latency
- **Error Rates**: Monitor scorer failure rates
- **API Usage**: Track LLM call volumes and costs

## 🔮 Future Enhancements

### Planned Improvements
1. **OpenAI Integration**: Support for autoevals when API key available
2. **Advanced Scorers**: Domain-specific evaluation criteria
3. **Caching Layer**: Response and score caching for performance
4. **Analytics Dashboard**: Custom metrics and reporting
5. **A/B Testing Framework**: Systematic comparison of RAG variations

### Extensibility Points
- **Custom Scorers**: Easy addition of new evaluation criteria
- **Multi-model Support**: Support for different LLM providers
- **Advanced Prompting**: Few-shot and fine-tuned evaluation models
- **Integration APIs**: Webhook support for external systems

## 📈 Success Metrics

### Evaluation Quality Indicators
- **Score Correlation**: Agreement between different judges
- **User Satisfaction**: Correlation with actual user feedback
- **Response Improvement**: Quality trends over time
- **Coverage**: Comprehensive evaluation across all use cases

### System Performance Metrics
- **Evaluation Latency**: Time from response to scores
- **Reliability**: Uptime and error rates
- **Scalability**: Performance under load
- **Cost Efficiency**: Evaluation cost per response

---

## 🎯 Conclusion

The Braintrust LLM-as-a-Judge evaluation system provides comprehensive, automated quality assessment for RAG responses. By combining custom LLM judges with proven non-LLM baselines, the system offers both nuanced evaluation and reliable benchmarking. The integration with Braintrust experiments enables continuous monitoring and improvement of RAG system performance.

The implementation follows industry best practices for LLM evaluation while being tailored to the specific needs of enterprise RAG applications. The modular architecture ensures easy maintenance and extensibility as requirements evolve.