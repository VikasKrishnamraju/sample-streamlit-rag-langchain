# Sample Streamlit RAG LangChain Application 🧙‍♂️

A comprehensive Retrieval-Augmented Generation (RAG) chat application built with Streamlit, LangChain, and LiteLLM. This application allows users to create chat sessions, upload documents, add web links, and engage in AI-powered conversations with contextual understanding from their uploaded content.

## 🚀 Features

- **Multi-Chat Interface**: Create and manage multiple chat sessions with custom titles
- **Document Processing**: Support for multiple file formats (PDF, DOCX, TXT, CSV, HTML, MD)
- **Web Content Integration**: Extract and process content from web links
- **Vector Search**: Powered by ChromaDB for efficient similarity search
- **Streaming Responses**: Real-time AI responses with typing animation
- **Persistent Storage**: SQLite database for chat history and metadata
- **Paginated UI**: Clean, organized interface with pagination support

## 📋 Table of Contents

- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [RAGAS Evaluation](#ragas-evaluation)
- [File Structure](#file-structure)
- [Implementation Guide](#implementation-guide)
- [API Reference](#api-reference)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

## 🛠 Installation

### Prerequisites

- Python 3.8+
- pip package manager
- Git

### Setup Steps

1. **Clone the repository:**
   ```bash
   git clone https://github.com/VikasKrishnamraju/sample-streamlit-rag-langchain.git
   cd sample-streamlit-rag-langchain
   ```

2. **Create and activate virtual environment:**
   ```bash
   python -m venv .venv
   
   # On Windows
   .venv\Scripts\activate
   
   # On macOS/Linux
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables:**
   Create a `.env` file in the root directory:
   ```bash
   LITELLM_API_KEY=your_api_key_here
   LITELLM_BASE_URL=https://your-litellm-endpoint.com
   LITELLM_MODEL=anthropic/claude-sonnet-4-20250514
   LITELLM_EMBEDDING_MODEL=text-embedding-3-small
   ```

5. **Initialize the database:**
   ```bash
   python create_relational_db.py
   ```

6. **Run the application:**
   ```bash
   streamlit run chats.py
   ```

## ⚙️ Configuration

### Environment Variables

The application requires the following environment variables in your `.env` file:

- `LITELLM_API_KEY`: Your LiteLLM API key for accessing the chat model and embeddings
- `LITELLM_BASE_URL`: The base URL for your LiteLLM endpoint
- `LITELLM_MODEL`: The chat model to use (default: anthropic/claude-sonnet-4-20250514)
- `LITELLM_EMBEDDING_MODEL`: The embedding model to use (default: text-embedding-3-small)

### Model Configuration

The application is configured to use:
- **Chat Model**: `anthropic/claude-sonnet-4-20250514` via LiteLLM
- **Embeddings**: `text-embedding-3-small` via LiteLLM
- **Vector Store**: ChromaDB with persistence
- **Text Splitting**: CharacterTextSplitter (1000 chunk size, 0 overlap)

## 📖 Usage

### Starting a New Chat

1. Launch the application with `streamlit run chats.py`
2. Enter a chat title in the input field
3. Click "Create Chat" to start a new conversation
4. The application will redirect you to the chat interface

### Uploading Documents

1. In the chat sidebar, find the "📑 Documents" section
2. Use the file uploader to select your document
3. Supported formats: PDF, DOCX, TXT, CSV, HTML, MD
4. The document will be processed and added to the chat's vector store

### Adding Web Links

1. In the chat sidebar, find the "🔗 Links" section
2. Enter a URL in the text input field
3. Click "Add Link" to fetch and process the web content
4. The extracted content will be added to the chat's knowledge base

### Chatting with AI

1. Type your question in the chat input at the bottom
2. The AI will use the uploaded documents and links as context
3. Responses are streamed in real-time with a typing effect
4. All conversations are automatically saved

## 📊 RAGAS Evaluation

RAGAS (Retrieval-Augmented Generation Assessment) provides comprehensive evaluation metrics for RAG system performance. The implementation offers objective assessment of retrieval quality and response generation effectiveness.

### Core Metrics

- **Faithfulness**: Measures answer grounding in retrieved contexts
- **Answer Relevancy**: Evaluates question-answer alignment  
- **Context Precision**: Assesses retrieval ranking quality
- **Context Recall**: Measures information completeness
- **Context Entity Recall**: Evaluates entity-level coverage

### Evaluation Flow

```
Input: Query + Generated Answer + Retrieved Contexts
    ↓
RAGAS Metric Calculation (using LiteLLM models)
    ↓  
Score Generation (0-1 range for each metric)
    ↓
CSV Export (EvaluationScores.csv)
```

### Implementation

**File**: `ragas_eval.py`  
**Function**: `eval_thr_ragas(query, answer, retrieved_contexts)`  
**Output**: Dictionary with metric scores + CSV file  
**Configuration**: Uses same LiteLLM setup as main application

### Usage Scenarios

1. **Single Evaluation**: Test individual query-answer pairs
2. **Batch Processing**: Evaluate multiple conversations at once
3. **Performance Monitoring**: Track RAG system quality over time
4. **A/B Testing**: Compare different retrieval or generation strategies

---

## 🧪 Test Data Generation

Automated generation of high-quality test datasets for RAG system evaluation using RAGAS framework. Multiple approaches available for different use cases and frameworks.

### Available Generators

| Generator | Framework | Status | Best For |
|-----------|-----------|--------|----------|
| `generate_testdata_correct.py` | Pure RAGAS | ✅ Stable | Production use, most reliable |
| `generate_testdata_langchain.py` | LangChain + RAGAS | ✅ Stable | LangChain workflows |
| `generate_testdata_llama_index.py` | LlamaIndex + RAGAS | ⚠️ Dependency conflicts | Advanced document parsing |
| `generate_simple_testdata.py` | Custom | ✅ Simple | Quick prototyping |

### Test Data Generation Flow

```
PDF Documents (pdf/ directory)
    ↓
Document Loading & Preprocessing
    ↓
Text Chunking & Splitting
    ↓
Knowledge Graph Construction
    ↓
Persona Generation (user types/roles)
    ↓
Scenario Creation (question contexts)
    ↓
Question-Answer Pair Generation
    ↓
Quality Filtering & Validation
    ↓
CSV Export (TestData_*.csv)
```

### Generation Process Details

#### 1. Document Processing
- **Input**: PDF files from `pdf/` directory
- **Processing**: Text extraction, cleaning, chunking
- **Chunking Strategy**: Recursive character splitting with overlap
- **Optimization**: Chunk size 800 chars, overlap 50 chars

#### 2. Knowledge Graph Construction
- **Purpose**: Understanding document relationships and entities
- **Components**: Entity extraction, theme identification, relationship mapping
- **Output**: Structured knowledge representation for question generation

#### 3. Persona Development
- **Concept**: Different user types who would ask questions about the content
- **Examples**: Domain experts, beginners, analysts, decision-makers
- **Impact**: Ensures diverse question styles and complexity levels

#### 4. Scenario Generation
- **Process**: Creating realistic contexts where questions would arise
- **Variety**: Different use cases, urgency levels, information needs
- **Quality**: Scenarios ground questions in realistic user interactions

#### 5. Question-Answer Synthesis
- **Question Types**: Factual, analytical, comparative, summarization
- **Answer Generation**: Grounded in document content with proper citations
- **Quality Control**: Relevance filtering, coherence checking

### Output Format

Generated test datasets include:
- **user_input**: Natural language questions
- **reference_contexts**: Relevant document excerpts
- **reference**: Ground truth answers
- **synthesizer_name**: Generation method identifier

### Configuration & Setup

#### Prerequisites
```bash
# System dependencies (macOS)
brew install libmagic

# Python packages
pip install ragas==0.3.2
pip install "unstructured[pdf]==0.16.4"
```

#### Environment Requirements
- LiteLLM API access configured in `.env`
- PDF documents in `pdf/` directory
- Sufficient API quota for generation process

### Best Practices

#### Document Selection
- **Quality**: Use well-structured, informative PDFs
- **Diversity**: Include different content types and complexity levels
- **Size**: Start with 2-3 documents for initial testing

#### Generation Parameters
- **Test Size**: Begin with 2-3 test cases, scale gradually
- **Chunk Size**: Adjust based on document complexity
- **Error Handling**: Always enable `raise_exceptions=False`

#### Quality Assurance
- **Manual Review**: Inspect generated questions for coherence
- **Diversity Check**: Ensure varied question types and difficulty
- **Context Validation**: Verify answers are grounded in provided contexts

### Troubleshooting

#### Common Issues
- **Dependency Conflicts**: Use compatible package versions
- **API Timeouts**: Reduce test generation size
- **Document Processing**: Verify PDF accessibility and format
- **Memory Issues**: Process smaller document batches

#### Performance Optimization
- **Batch Size**: Generate 2-5 test cases per run initially  
- **Chunking**: Optimize chunk size for your document types
- **Caching**: Reuse processed documents when possible
- **Monitoring**: Track API usage and processing time

### Integration with Evaluation

The generated test data integrates seamlessly with the RAGAS evaluation system:

```
Generated Test Data → RAG System Processing → RAGAS Evaluation → Performance Metrics
```

This creates a complete testing and evaluation pipeline for continuous RAG system improvement.

## 📁 File Structure

```
sample-streamlit-rag-langchain/
│
├── chats.py                         # Main Streamlit application
├── db.py                           # Database operations and SQLite management
├── vector_functions.py             # Vector store and RAG functionality
├── litellm_embeddings.py           # Custom LiteLLM embeddings wrapper
├── create_relational_db.py         # Database initialization script
├── ragas_eval.py                   # RAGAS evaluation implementation
├── g_eval.py                       # G-Eval implementation (placeholder)
│
├── generate_testdata_correct.py    # ✅ Primary RAGAS test generator
├── generate_testdata_langchain.py  # ✅ LangChain-based test generator  
├── generate_testdata_llama_index.py # ⚠️ LlamaIndex test generator
├── generate_simple_testdata.py     # ✅ Simple CSV test generator
│
├── requirements.txt                # Python dependencies
├── .gitignore                      # Git ignore patterns
├── .env                           # Environment variables (not committed)
├── README.md                      # Project documentation
│
├── EvaluationScores.csv           # RAGAS evaluation results
├── TestData_From_Ragas_*.csv      # Generated test datasets
│
├── temp_files/                    # Temporary file storage
├── persist/                       # ChromaDB persistence directory
├── pdf/                           # PDF storage directory (place your PDFs here)
└── __pycache__/                   # Python cache files
```

## 🔧 Implementation Guide

### Core Components

#### 1. Main Application (`chats.py`)

**Purpose**: Primary Streamlit interface handling routing, UI components, and user interactions.

**Key Functions**:
- `main()`: Entry point that routes between chat list and individual chats
- `chats_home()`: Renders the main page with chat creation and management
- `chat_page(chat_id)`: Displays individual chat interface with sidebar controls
- `stream_response(response)`: Creates typing animation effect for AI responses

**Implementation Details**:
- Uses Streamlit's query parameters for navigation
- Implements pagination for chat history
- Handles file uploads and web link processing
- Manages real-time chat interactions

#### 2. Database Operations (`db.py`)

**Purpose**: SQLite database management with CRUD operations for chats, messages, and sources.

**Key Functions**:
- `connect_db()`: Creates database connection
- `create_chat(title)`: Creates new chat session
- `create_message(chat_id, sender, content)`: Stores chat messages
- `create_source(name, source_text, chat_id, source_type)`: Saves document/link metadata

**Database Schema**:
```sql
-- Chat sessions
CREATE TABLE chat (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Document and link sources
CREATE TABLE sources (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    source_text TEXT,
    type TEXT DEFAULT "document",
    chat_id INTEGER,
    FOREIGN KEY (chat_id) REFERENCES chat(id)
);

-- Chat messages
CREATE TABLE messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id INTEGER NOT NULL,
    sender TEXT NOT NULL,
    content TEXT NOT NULL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(chat_id) REFERENCES chat(id)
);
```

#### 3. Vector Operations (`vector_functions.py`)

**Purpose**: Handles document processing, vector storage, and RAG functionality using LangChain and ChromaDB.

**Key Functions**:
- `load_document(file_path)`: Processes various document formats
- `create_collection(collection_name, documents)`: Creates new ChromaDB collection
- `load_retriever(collection_name, score_threshold)`: Creates similarity-based retriever
- `generate_answer_from_context(retriever, question)`: Performs RAG query

**Supported Document Types**:
- **PDF**: Using PyPDFLoader
- **DOCX**: Using Docx2txtLoader  
- **TXT**: Using TextLoader
- **CSV**: Using CSVLoader
- **HTML**: Using UnstructuredHTMLLoader
- **MD**: Using UnstructuredMarkdownLoader

**Vector Search Configuration**:
- Similarity score threshold: 0.3 (configurable)
- Text chunking: 1000 characters with 0 overlap
- Embedding model: text-embedding-3-small

#### 4. Custom Embeddings (`litellm_embeddings.py`)

**Purpose**: Custom LangChain embeddings wrapper for LiteLLM integration.

**Key Features**:
- Implements LangChain's Embeddings interface
- Handles both document and query embeddings
- Includes error handling with fallback dummy embeddings
- Configurable API endpoint and model selection

**Usage Example**:
```python
from litellm_embeddings import LiteLLMEmbeddings

embeddings = LiteLLMEmbeddings(model="text-embedding-3-small")
doc_embeddings = embeddings.embed_documents(["Hello world"])
query_embedding = embeddings.embed_query("Hello")
```

#### 5. Database Setup (`create_relational_db.py`)

**Purpose**: Database initialization script that creates required tables.

**Execution**:
```bash
python create_relational_db.py
```

Creates SQLite database `doc_sage.sqlite` with proper schema and relationships.

#### 6. RAGAS Evaluation (`ragas_eval.py`)

**Purpose**: Comprehensive evaluation framework for RAG system performance using multiple metrics.

**Key Functions**:
- `eval_thr_ragas(query, answer, retrieved_contexts)`: Main evaluation function
- Input validation and format conversion for RAGAS compatibility
- Integration with LiteLLM models and embeddings
- CSV export functionality for result analysis

**Implementation Details**:
- Supports both single and batch evaluations
- Handles various input formats (strings, lists)
- Robust error handling with detailed logging
- Automatic result persistence to CSV files
- Compatible with existing LiteLLM configuration

#### 7. Test Data Generators

**Purpose**: Automated generation of high-quality test datasets for RAG system evaluation.

**Available Generators**:

**`generate_testdata_correct.py` (Recommended)**
- Pure RAGAS implementation with maximum stability
- Full pipeline: document processing → knowledge graphs → persona generation → test synthesis
- Output: `TestData_From_Ragas_Correct.csv`

**`generate_testdata_langchain.py`**
- LangChain-based document processing with RAGAS generation
- Optimized for LangChain workflows and existing integrations
- Output: `TestData_From_Ragas_Langchain.csv`

**`generate_testdata_llama_index.py`**
- LlamaIndex document parsing with advanced node processing
- Currently has dependency conflicts (Pydantic version issues)
- Output: `TestData_From_Ragas_Llama.csv`

**`generate_simple_testdata.py`**
- Custom implementation for quick prototyping
- Manual question-answer generation from document content
- Output: `TestData_Simple.csv`

**Common Features**:
- PDF document processing from `pdf/` directory
- Configurable test dataset size and complexity
- Integration with existing LiteLLM configuration
- Comprehensive error handling and logging
- CSV export with standardized format

**Usage Integration**:
```python
from ragas_eval import eval_thr_ragas
from vector_functions import generate_answer_from_context

# Generate answer using RAG
answer = generate_answer_from_context(retriever, query)
contexts = [doc.page_content for doc in retriever.get_relevant_documents(query)]

# Evaluate with RAGAS
scores = eval_thr_ragas(query, answer, contexts)
```

### Data Flow

1. **Document Upload**:
   ```
   File Upload → load_document() → Text Splitting → Vector Embedding → ChromaDB Storage → Metadata in SQLite
   ```

2. **Web Link Processing**:
   ```
   URL Input → BeautifulSoup Extraction → Document Creation → Vector Processing → ChromaDB Storage
   ```

3. **Chat Query**:
   ```
   User Question → Vector Similarity Search → Context Retrieval → LLM Processing → Streamed Response
   ```

4. **RAGAS Evaluation**:
   ```
   Query + Answer + Contexts → RAGAS Metrics Calculation → Score Generation → CSV Export
   ```

### RAG Implementation

The application implements a sophisticated RAG pipeline:

1. **Document Ingestion**: Multiple loaders handle different file formats
2. **Text Processing**: CharacterTextSplitter creates manageable chunks
3. **Vectorization**: Custom LiteLLM embeddings create vector representations
4. **Storage**: ChromaDB provides efficient similarity search
5. **Retrieval**: Similarity-based retriever with configurable thresholds
6. **Generation**: Claude model generates contextual responses

## 📚 API Reference

### Database Functions

```python
# Chat management
create_chat(title: str) -> int
read_chat(chat_id: int) -> tuple
update_chat(chat_id: int, new_title: str) -> None
delete_chat(chat_id: int) -> None
list_chats() -> list[tuple]

# Message management  
create_message(chat_id: int, sender: str, content: str) -> None
get_messages(chat_id: int) -> list[tuple]
delete_messages(chat_id: int) -> None

# Source management
create_source(name: str, source_text: str, chat_id: int, source_type: str) -> None
list_sources(chat_id: int, source_type: str = None) -> list[tuple]
delete_source(source_id: int) -> None
```

### RAGAS Evaluation Functions

```python
# Main evaluation function
eval_thr_ragas(query: str, answer: str, retrieved_contexts: list) -> dict

# Input format examples
query = "What is machine learning?"
answer = "Machine learning is a subset of artificial intelligence..."
contexts = ["Machine learning (ML) is a field of study...", "AI encompasses..."]

# Returns evaluation scores
scores = {
    'faithfulness': 0.85,
    'answer_relevancy': 0.92,
    'context_precision': 0.78,
    'context_recall': 0.88
}
```

### Vector Functions

```python
# Document processing
load_document(file_path: str) -> list[Document]

# Collection management
create_collection(collection_name: str, documents: list) -> Chroma
load_collection(collection_name: str) -> Chroma
add_documents_to_collection(vectordb: Chroma, documents: list) -> Chroma

# Retrieval and generation
load_retriever(collection_name: str, score_threshold: float = 0.3) -> Retriever
generate_answer_from_context(retriever: Retriever, question: str) -> str
```

### Custom Embeddings

```python
class LiteLLMEmbeddings(Embeddings):
    def __init__(self, model: str = "text-embedding-3-small")
    def embed_documents(self, texts: List[str]) -> List[List[float]]
    def embed_query(self, text: str) -> List[float]
```

## 🔍 Troubleshooting

### Common Issues

1. **API Key Error**:
   ```
   Error: Invalid API key
   Solution: Verify LITELLM_API_KEY in .env file
   ```

2. **Database Not Found**:
   ```
   Error: no such table: chat
   Solution: Run python create_relational_db.py
   ```

3. **File Upload Failed**:
   ```
   Error: Unsupported file type
   Solution: Check supported formats (PDF, DOCX, TXT, CSV, HTML, MD)
   ```

4. **Vector Store Error**:
   ```
   Error: Collection not found
   Solution: Upload documents first to create collection
   ```

5. **Web Link Processing Failed**:
   ```
   Error: Failed to fetch content
   Solution: Check URL accessibility and network connection
   ```

### Performance Optimization

- **Large Documents**: Consider reducing chunk_size in text_splitter
- **Memory Usage**: Monitor ChromaDB persistence directory size
- **Response Speed**: Adjust similarity score threshold for faster/more accurate results
- **Database Performance**: Consider indexing for large chat histories

### Security Considerations

- **API Keys**: Never commit .env files to version control
- **Web Scraping**: Uses user-agent headers and SSL verification disabled (configure as needed)
- **File Uploads**: Temporary files are cleaned up after processing
- **Database**: SQLite provides basic security; consider PostgreSQL for production

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

1. Follow installation steps
2. Install development dependencies:
   ```bash
   pip install -r requirements-dev.txt  # If available
   ```
3. Run tests:
   ```bash
   pytest  # If tests are implemented
   ```

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- **LangChain**: Framework for building LLM applications
- **Streamlit**: Web application framework
- **ChromaDB**: Vector database for embeddings
- **LiteLLM**: Unified API for multiple LLM providers
- **BeautifulSoup**: Web scraping library

## 📞 Support

For questions, issues, or contributions:
- Create an issue in the GitHub repository
- Review existing documentation
- Check troubleshooting section

---

**Happy Chatting! 🧙‍♂️✨**