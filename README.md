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

RAGAS (Retrieval-Augmented Generation Assessment) is integrated into the application for comprehensive evaluation of RAG performance. The implementation provides objective metrics to assess the quality of retrievals and generated responses.

### Features

- **Faithfulness**: Measures whether the generated answer is faithful to the retrieved contexts
- **Answer Relevancy**: Evaluates how relevant the generated answer is to the user's question
- **Context Precision**: Assesses the precision of the retrieved contexts
- **Context Recall**: Measures the recall of the retrieved contexts
- **Context Entity Recall**: Evaluates entity-level recall in retrieved contexts

### Implementation

The RAGAS evaluation is implemented in `ragas_eval.py` with the following key components:

#### Core Function: `eval_thr_ragas()`

```python
def eval_thr_ragas(query, answer, retrieved_contexts):
    """
    Evaluate RAG performance using RAGAS metrics.
    
    Args:
        query (str): The user's question
        answer (str): The generated answer
        retrieved_contexts (list): List of retrieved context strings
        
    Returns:
        dict: RAGAS evaluation scores
    """
```

#### Supported Metrics

- **Faithfulness**: Ensures generated answers are grounded in retrieved contexts
- **Answer Relevancy**: Measures how well answers address the specific question
- **Context Precision**: Evaluates retrieval quality and ranking
- **Context Recall**: Assesses completeness of retrieved information
- **Context Entity Recall**: Entity-level evaluation of retrieved contexts

### Usage Example

```python
from ragas_eval import eval_thr_ragas

# Example evaluation
query = "What is the capital of France?"
answer = "The capital of France is Paris."
contexts = ["Paris is the capital and largest city of France."]

scores = eval_thr_ragas(query, answer, contexts)
print(f"Evaluation scores: {scores}")
```

### Configuration

The RAGAS evaluation uses:
- **LLM**: Same LiteLLM model configured for the main application
- **Embeddings**: text-embedding-3-small via LiteLLM
- **Output**: Results are saved to `EvaluationScores.csv`

### Metrics Interpretation

- **Faithfulness (0-1)**: Higher scores indicate answers are more faithful to contexts
- **Answer Relevancy (0-1)**: Higher scores indicate more relevant answers
- **Context Precision (0-1)**: Higher scores indicate better context ranking
- **Context Recall (0-1)**: Higher scores indicate more complete context retrieval
- **Context Entity Recall (0-1)**: Higher scores indicate better entity coverage

### Running Evaluations

1. **Single Evaluation**:
   ```python
   from ragas_eval import eval_thr_ragas
   scores = eval_thr_ragas(query, answer, contexts)
   ```

2. **Batch Evaluation**:
   The function supports batch processing with lists of queries, answers, and contexts.

3. **Output Analysis**:
   Results are automatically saved to `EvaluationScores.csv` for further analysis.

### Error Handling

The implementation includes robust error handling:
- Input validation and format conversion
- Fallback mechanisms for API failures
- Detailed error logging for debugging

## 📁 File Structure

```
sample-streamlit-rag-langchain/
│
├── chats.py                    # Main Streamlit application
├── db.py                       # Database operations and SQLite management
├── vector_functions.py         # Vector store and RAG functionality
├── litellm_embeddings.py       # Custom LiteLLM embeddings wrapper
├── create_relational_db.py     # Database initialization script
├── ragas_eval.py               # RAGAS evaluation implementation
├── g_eval.py                   # G-Eval implementation (placeholder)
├── requirements.txt            # Python dependencies
├── .gitignore                  # Git ignore patterns
├── .env                        # Environment variables (not committed)
├── EvaluationScores.csv        # RAGAS evaluation results output
│
├── temp_files/                 # Temporary file storage
├── persist/                    # ChromaDB persistence directory
├── pdf/                        # PDF storage directory
└── __pycache__/               # Python cache files
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