import os
from langchain_community.chat_models import ChatLiteLLM
from litellm_embeddings import LiteLLMEmbeddings
from langchain_chroma import Chroma
#from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import (
    TextLoader,
    CSVLoader,
    PyPDFLoader,
    Docx2txtLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
)
import environ

env = environ.Env()
#reading .env file
environ.Env.read_env()

def get_lite_llm_model():
    os.environ["ANTHROPIC_API_KEY"] = env("LITELLM_API_KEY")
    model = ChatLiteLLM(
            model="anthropic/claude-sonnet-4-20250514",  # Provider must be specified
            api_base="https://litellm.int.thomsonreuters.com",  # optional, for custom endpoints
            temperature=0
    )
    return model

embeddings = LiteLLMEmbeddings(model="text-embedding-3-small")

# llm = ChatOpenAI(
#     model="gpt-4o-mini",
#     api_key=env("OPENAI_API_KEY"),
# )

# embeddings = OpenAIEmbeddings(
#     api_key=env("OPENAI_API_KEY"),
# )

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)


def load_document(file_path: str) -> list[Document]:
    """
    Load a document from a file path.
    Supports .txt, .pdf, .docx, .csv, .html, and .md files.

    Args:
    file_path (str): Path to the document file.

    Returns:
    list[Document]: A list of Document objects.

    Raises:
    ValueError: If the file type is not supported.
    """
    _, file_extension = os.path.splitext(file_path)

    if file_extension == ".txt":
        loader = TextLoader(file_path)
    elif file_extension == ".pdf":
        loader = PyPDFLoader(file_path)
    elif file_extension == ".docx":
        loader = Docx2txtLoader(file_path)
    elif file_extension == ".csv":
        loader = CSVLoader(file_path)
    elif file_extension == ".html":
        loader = UnstructuredHTMLLoader(file_path)
    elif file_extension == ".md":
        loader = UnstructuredMarkdownLoader(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_extension}")

    return loader.load()


def create_collection(collection_name, documents):
    """
    Create a new Chroma collection from the given documents.

    Args:
    collection_name (str): The name of the collection to create.
    documents (list): A list of documents to add to the collection.

    Returns:
    None

    This function splits the documents into texts, creates a new Chroma collection,
    and persists it to disk.
    """
    # Split the documents into smaller text chunks
    texts = text_splitter.split_documents(documents)
    persist_directory = "./persist"

    # Create a new Chroma collection from the text chunks
    try:
        vectordb = Chroma.from_documents(
            documents=texts,
            embedding=embeddings,
            persist_directory=persist_directory,
            collection_name=collection_name,
        )
    except Exception as e:
        print(f"Error creating collection: {e}")
        return None

    return vectordb


def load_collection(collection_name):
    """
    Load an existing Chroma collection.

    Args:
    collection_name (str): The name of the collection to load.

    Returns:
    Chroma: The loaded Chroma collection.

    This function loads a previously created Chroma collection from disk.
    """
    persist_directory = "./persist"
    # Load the Chroma collection from the specified directory
    vectordb = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings,
        collection_name=collection_name,
    )

    return vectordb


def load_retriever(collection_name, score_threshold: float = 0.3):
    """
    Create a retriever from a Chroma collection with a similarity score threshold.

    Args:
    collection_name (str): The name of the collection to use.
    score_threshold (float): The minimum similarity score threshold for retrieving documents.
                           Documents with scores below this threshold will be filtered out.
                           Defaults to 0.3.

    Returns:
    Retriever: A retriever object that can be used to query the collection with similarity
              score filtering.

    This function loads a Chroma collection and creates a retriever from it that will only
    return documents meeting the specified similarity score threshold.
    """
    # Load the Chroma collection
    vectordb = load_collection(collection_name)
    # Create a retriever from the collection with specified search parameters
    retriever = vectordb.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"score_threshold": score_threshold},
    )
    return retriever


def generate_answer_from_context(retriever, question: str):
    """
    Ask a question and get an answer based on the provided context.

    Args:
        retriever: A retriever object to fetch relevant context.
        question (str): The question to be answered.

    Returns:
        str: The answer to the question based on the retrieved context.
    """
    # First, test what documents the retriever finds
    try:
        retrieved_docs = retriever.get_relevant_documents(question)
        print(f"Retrieved {len(retrieved_docs)} documents for question: {question[:50]}...")
        
        if not retrieved_docs:
            return "I couldn't find any relevant information in the uploaded documents to answer your question. Please make sure your document contains content related to your query."
        
        # Print first few docs for debugging
        for i, doc in enumerate(retrieved_docs[:2]):
            print(f"Doc {i+1} content preview: {doc.page_content[:100]}...")
            
    except Exception as e:
        print(f"Error during retrieval: {e}")
        return f"Error retrieving relevant documents: {e}"
    
    # Define the message template for the prompt
    message = """
    Answer this question using the provided context only.

    {question}

    Context:
    {context}
    """

    # Create a chat prompt template from the message
    prompt = ChatPromptTemplate.from_messages([("human", message)])

    # Create a RAG (Retrieval-Augmented Generation) chain
    # This chain retrieves context, passes through the question,
    # formats the prompt, and generates an answer using the language model
    rag_chain = {"context": retriever, "question": RunnablePassthrough()} | prompt | get_lite_llm_model()

    # Invoke the RAG chain with the question and return the generated content
    return rag_chain.invoke(question).content


def add_documents_to_collection(vectordb, documents):
    """
    Add documents to the vector database collection.

    Args:
        vectordb: The vector database object to add documents to.
        documents: A list of documents to be added to the collection.

    This function splits the documents into smaller chunks, adds them to the
    vector database, and persists the changes.
    """

    # Split the documents into smaller text chunks
    texts = text_splitter.split_documents(documents)

    # Add the text chunks to the vector database
    vectordb.add_documents(texts)

    return vectordb