import streamlit as st
import os, time
import requests
import certifi
import ssl
from requests.adapters import HTTPAdapter
from urllib3.util.ssl_ import create_urllib3_context
from bs4 import BeautifulSoup
from langchain_core.documents import Document
import math
# from ragas_eval import eval_thr_ragas  # Commented out - using Braintrust LLM-as-a-Judge instead
from braintrust_realtime_eval import eval_and_log_realtime_response 
from db import (
    read_chat,
    create_chat,
    list_chats,
    delete_chat,
    create_message,
    get_messages,
    create_source,
    list_sources,
    delete_source,
)
from vector_functions import (
    load_document,
    create_collection,
    load_retriever,
    generate_answer_from_context,
    add_documents_to_collection,
    load_collection,
)


def _run_evaluations(prompt: str, response: str, retriever, chat_id: str):
    """Run RAGAS and Braintrust LLM-as-a-Judge evaluations on the response."""
    try:
        retrieved_docs = retriever.get_relevant_documents(prompt)
        retrieved_contexts = [doc.page_content for doc in retrieved_docs]

        # RAGAS Evaluation (COMMENTED OUT - using Braintrust LLM-as-a-Judge instead)
        # if len(retrieved_contexts) > 0:
        #     try:
        #         ragas_eval_ans = eval_thr_ragas(prompt, response, retrieved_contexts)
        #         if ragas_eval_ans is not None:
        #             st.session_state[f"answer_relevancy_{chat_id}"] = f"{ragas_eval_ans['answer_relevancy'][0]:.2f}"
        #             st.session_state[f"faithfulness_{chat_id}"] = f"{ragas_eval_ans['faithfulness'][0]:.2f}"
        #     except Exception as e:
        #         print(f"Error during RAGAS evaluation: {e}")
        # else:
        #     print("No contexts retrieved - skipping RAGAS evaluation")

        # Braintrust Real-time Evaluation with Individual Traces
        try:
            print(f"🔄 Running real-time Braintrust evaluation...")
            realtime_results = eval_and_log_realtime_response(
                question=prompt,
                answer=response,
                chat_id=str(chat_id)
            )

            if realtime_results["status"] == "success":
                scores = realtime_results["scores"]
                # Update session state with scores
                _update_session_scores(scores, chat_id)

                print(f"✅ Real-time evaluation successful!")
                print(f"📊 Scores: {scores}")
                print(f"📝 Judge outputs: {len(realtime_results.get('judge_outputs', {}))} judges")
                print(f"🔗 Experiment: {realtime_results['experiment_name']}")
                print(f"🌐 View at: {realtime_results['experiment_url']}")
            else:
                print(f"⚠️ Real-time evaluation failed: {realtime_results.get('error', 'Unknown error')}")

        except Exception as e:
            print(f"Error during Braintrust real-time evaluation: {e}")

    except Exception as e:
        print(f"Error during document retrieval: {e}")

def _update_session_scores(scores: dict, chat_id: str):
    """Update session state with evaluation scores."""
    score_mappings = {
        "answer_relevancy_scorer": f"llm_judge_relevancy_{chat_id}",
        "answer_accuracy_scorer": f"llm_judge_accuracy_{chat_id}",
        "answer_completeness_scorer": f"llm_judge_completeness_{chat_id}",
        "ExactMatch": f"exact_match_score_{chat_id}",
        "Levenshtein": f"levenshtein_score_{chat_id}"
    }

    for score_key, session_key in score_mappings.items():
        if score_key in scores:
            st.session_state[session_key] = f"{scores[score_key]:.2f}"

def create_secure_session():
    """Create a requests session with enterprise SSL certificate configuration"""
    session = requests.Session()
    
    # Use final complete certificate bundle with all certificates in chain
    cert_bundle_path = "corp-bundle-final-complete.pem"
    
    # Create SSL context with enterprise certificates
    ctx = create_urllib3_context()
    ctx.check_hostname = True
    ctx.verify_mode = ssl.CERT_REQUIRED
    
    # Load enterprise certificate bundle and system certificates
    if os.path.exists(cert_bundle_path):
        ctx.load_verify_locations(cert_bundle_path)
        # Also load system keychain certificates (macOS)
        ctx.load_default_certs()
        print(f"Using enterprise certificate bundle + system keychain: {cert_bundle_path}")
    else:
        ctx.load_verify_locations(certifi.where())
        ctx.load_default_certs()
        print("Using default certificate bundle + system keychain")
    
    # Custom adapter
    class SSLAdapter(HTTPAdapter):
        def init_poolmanager(self, *args, **kwargs):
            kwargs['ssl_context'] = ctx
            return super().init_poolmanager(*args, **kwargs)
    
    session.mount('https://', SSLAdapter())
    return session


def chats_home():
    """
    Renders the main chats page where users can:
    - Create new chats with titles
    - View and manage previous chats
    - Navigate through paginated chat history

    The page displays a header, chat creation form, and list of existing chats
    with options to open each chat.
    """

    st.markdown(
        "<h1 style='text-align: center;'>Sample-Streamlit-Rag-Langchain-App🧙‍♂️</h1>", unsafe_allow_html=True
    )

    with st.container(border=True):
        col1, col2 = st.columns([0.8, 0.2])

        with col1:
            chat_title = st.text_input(
                "Chat Title", placeholder="Enter Chat Title", key="chat_title"
            )

        with col2:
            st.markdown("<br>", unsafe_allow_html=True)  # Add vertical space
            if st.button("Create Chat", type="primary"):
                if chat_title:
                    chat_id = create_chat(chat_title)
                    st.success(f"Created new chat: {chat_title}")
                    st.query_params.from_dict({"chat_id": chat_id})
                    st.rerun()
                else:
                    st.warning("Please enter a chat title")

    with st.container(border=True):
        st.subheader("Previous Chats")

        # get previous chats from db
        previous_chats = list_chats()

        # Pagination settings
        chats_per_page = 5
        total_pages = math.ceil(len(previous_chats) / chats_per_page)

        # Get current page from session state
        if "current_page" not in st.session_state:
            st.session_state.current_page = 1

        # Calculate start and end indices for the current page
        start_idx = (st.session_state.current_page - 1) * chats_per_page
        end_idx = start_idx + chats_per_page

        # Display chats for the current page
        for chat in previous_chats[start_idx:end_idx]:
            chat_id, chat_title = chat[0], chat[1]
            with st.container(border=True):
                col1, col2, col3 = st.columns([0.6, 0.2, 0.2])
                with col1:
                    st.markdown(f"**{chat_title}**")
                with col2:
                    if st.button("📂 Open", key=f"open_{chat_id}"):
                        st.query_params.from_dict({"chat_id": chat_id})
                        st.rerun()
                with col3:
                    if st.button("🗑️ Delete", key=f"delete_{chat_id}"):
                        delete_chat(chat_id)
                        st.success(f"Deleted chat: {chat_title}")
                        st.rerun()

        # Pagination controls
        col1, col2, col3 = st.columns([1, 2, 1])
        with col1:
            if st.button("Previous") and st.session_state.current_page > 1:
                st.session_state.current_page -= 1
                st.rerun()
        with col2:
            st.write(f"Page {st.session_state.current_page} of {total_pages}")
        with col3:
            if st.button("Next") and st.session_state.current_page < total_pages:
                st.session_state.current_page += 1
                st.rerun()


def stream_response(response):
    """
    Stream a response word by word with a delay between each word.

    Args:
        response (str): The text response to stream

    Yields:
        str: Individual words from the response with a space appended

    Note:
        Adds a 50ms delay between each word to create a typing effect
    """
    # Split response into words and stream each one
    for word in response.split():
        # Yield the word with a space and pause briefly
        yield word + " "
        time.sleep(0.05)


def chat_page(chat_id):
    """
    Display the chat page for a specific chat ID.

    This function handles displaying and managing an individual chat conversation, including:
    - Showing the chat history
    - Allowing users to send new messages
    - Streaming AI responses
    - Managing chat context through a vector store retriever

    Args:
        chat_id (int): The ID of the chat to display

    Returns:
        None
    """
    # Initialize RAGAS scores in session state if not exists (COMMENTED OUT - using Braintrust instead)
    # if f"answer_relevancy_{chat_id}" not in st.session_state:
    #     st.session_state[f"answer_relevancy_{chat_id}"] = None
    # if f"faithfulness_{chat_id}" not in st.session_state:
    #     st.session_state[f"faithfulness_{chat_id}"] = None

    # Initialize Braintrust LLM-as-a-Judge scores in session state if not exists
    if f"llm_judge_relevancy_{chat_id}" not in st.session_state:
        st.session_state[f"llm_judge_relevancy_{chat_id}"] = None
    if f"llm_judge_accuracy_{chat_id}" not in st.session_state:
        st.session_state[f"llm_judge_accuracy_{chat_id}"] = None
    if f"llm_judge_completeness_{chat_id}" not in st.session_state:
        st.session_state[f"llm_judge_completeness_{chat_id}"] = None
    if f"exact_match_score_{chat_id}" not in st.session_state:
        st.session_state[f"exact_match_score_{chat_id}"] = None
    if f"levenshtein_score_{chat_id}" not in st.session_state:
        st.session_state[f"levenshtein_score_{chat_id}"] = None
    
    chat = read_chat(chat_id)
    if not chat:
        st.error("Chat not found")
        return

    # Retrieve messages from DB
    messages = get_messages(chat_id)

    # Display messages
    if messages:
        for sender, content in messages:
            if sender == "user":
                with st.chat_message("user"):
                    st.markdown(content)
            elif sender == "ai":
                with st.chat_message("assistant"):
                    st.markdown(content)
    else:
        st.write("No messages yet. Start the conversation!")

    # Add a text input for new messages
    prompt = st.chat_input("Type your message here...")
    if prompt:

        # Save user message
        create_message(chat_id, "user", prompt)
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        # Get AI response

        # Load retriever for the chat context
        collection_name = f"chat_{chat_id}"
        try:
            # Check if collection exists by attempting to load it
            vectordb = load_collection(collection_name)
            # Test if collection has documents
            if vectordb._collection.count() > 0:
                retriever = load_retriever(collection_name=collection_name)
            else:
                retriever = None
        except Exception as e:
            print(f"Error loading collection {collection_name}: {e}")
            retriever = None

        # Ask question using the retriever
        if retriever:
            response = generate_answer_from_context(retriever, prompt)
            # Save AI response
            create_message(chat_id, "ai", response)
            # Display AI response
            with st.chat_message("assistant"):
                 st.write_stream(stream_response(response)) 

            # Run evaluations on the response
            _run_evaluations(prompt, response, retriever, chat_id)
        else:
            response = "I need some context to answer that question."
        st.rerun()

    # Sidebar for context
    with st.sidebar:
        # Button to return to the main chats page
        if st.button("Back to Chats"):
            st.query_params.clear()
            st.rerun()

        st.subheader(f"{chat[1]}")

        # RAGAS Scores (COMMENTED OUT - using Braintrust LLM-as-a-Judge instead)
        # st.sidebar.subheader("RAGAS Scores")
        # if st.session_state[f"answer_relevancy_{chat_id}"] is not None:
        #     st.sidebar.metric("Answer Relevancy ↗", st.session_state[f"answer_relevancy_{chat_id}"])
        # if st.session_state[f"faithfulness_{chat_id}"] is not None:
        #     st.sidebar.metric("Faithfulness ↗", st.session_state[f"faithfulness_{chat_id}"])

        st.sidebar.subheader("🧑‍⚖️ LLM-as-a-Judge Scores")
        if st.session_state[f"llm_judge_relevancy_{chat_id}"] is not None:
            st.sidebar.metric("Relevancy ↗", st.session_state[f"llm_judge_relevancy_{chat_id}"])
        if st.session_state[f"llm_judge_accuracy_{chat_id}"] is not None:
            st.sidebar.metric("Accuracy ↗", st.session_state[f"llm_judge_accuracy_{chat_id}"])
        if st.session_state[f"llm_judge_completeness_{chat_id}"] is not None:
            st.sidebar.metric("Completeness ↗", st.session_state[f"llm_judge_completeness_{chat_id}"])

        st.sidebar.subheader("📊 Non-LLM Scores")
        if st.session_state[f"exact_match_score_{chat_id}"] is not None:
            st.sidebar.metric("Exact Match ↗", st.session_state[f"exact_match_score_{chat_id}"])
        if st.session_state[f"levenshtein_score_{chat_id}"] is not None:
            st.sidebar.metric("Levenshtein ↗", st.session_state[f"levenshtein_score_{chat_id}"])

        # Documents Section
        st.subheader("📑 Documents")
        # Display list of documents
        documents = list_sources(chat_id, source_type="document")
        if documents:
            for doc in documents:
                doc_id = doc[0]
                doc_name = doc[1]
                col1, col2 = st.columns([0.8, 0.2])
                with col1:
                    st.write(doc_name)
                with col2:
                    if st.button("❌", key=f"delete_doc_{doc_id}"):
                        delete_source(doc_id)
                        st.success(f"Deleted document: {doc_name}")
                        st.rerun()
        else:
            st.write("No documents uploaded.")

        uploaded_file = st.file_uploader("Upload Document", key="file_uploader")

        if uploaded_file:
            # Save document content to database
            with st.spinner("Processing document..."):
                temp_dir = "temp_files"
                os.makedirs(temp_dir, exist_ok=True)
                temp_file_path = os.path.join(temp_dir, uploaded_file.name)
                with open(temp_file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # Load document
                document = load_document(temp_file_path)
                # Create or update collection for this chat
                collection_name = f"chat_{chat_id}"
                try:
                    # Try to load existing collection
                    vectordb = load_collection(collection_name)
                    vectordb = add_documents_to_collection(vectordb, document)
                    st.success(f"Added document to existing collection: {uploaded_file.name}")
                except Exception:
                    # Create new collection if it doesn't exist
                    vectordb = create_collection(collection_name, document)
                    st.success(f"Created new collection with document: {uploaded_file.name}")
                # Save source to database
                create_source(uploaded_file.name, "", chat_id, source_type="document")
                # Remove temp file
                os.remove(temp_file_path)

                del st.session_state["file_uploader"]

                st.rerun()

        # Links Section
        st.subheader("🔗 Links")
        # Display list of links
        links = list_sources(chat_id, source_type="link")
        if links:
            for link in links:
                link_id = link[0]
                link_url = link[1]
                col1, col2 = st.columns([0.8, 0.2])
                with col1:
                    st.markdown(f"[{link_url}]({link_url})")
                with col2:
                    if st.button("❌    ", key=f"delete_link_{link_id}"):
                        delete_source(link_id)
                        st.success(f"Deleted link: {link_url}")
                        st.rerun()
        else:
            st.write("No links added.")

        # Add new link
        new_link = st.text_input("Add a link", key="new_link")
        if st.button("Add Link", key="add_link_btn"):
            if new_link:
                with st.spinner("Processing link..."):
                    # Fetch content from the link
                    try:
                        headers = {
                            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.111 Safari/537.36"
                        }
                        # Use secure session with enterprise SSL configuration
                        session = create_secure_session()
                        response = session.get(new_link, headers=headers, timeout=30)
                        soup = BeautifulSoup(response.text, "html.parser")

                        # Check if the content was successfully retrieved
                        if response.status_code == 200 and soup.text.strip():
                            link_content = soup.get_text(separator="\n")
                        else:
                            st.toast(
                                "Unable to retrieve content from the link. It may be empty or inaccessible.",
                                icon="🚨",
                            )
                            return

                        # Save link content to vector store
                        documents = [
                            Document(
                                page_content=link_content, metadata={"source": new_link}
                            )
                        ]
                        collection_name = f"chat_{chat_id}"
                        if not os.path.exists(f"./persist"):
                            create_collection(collection_name, documents)
                        else:
                            vectordb = load_collection(collection_name)
                            add_documents_to_collection(vectordb, documents)

                        # Save link to database
                        create_source(new_link, "", chat_id, source_type="link")
                        st.success(f"Added link: {new_link}")
                        del st.session_state["add_link_btn"]
                        st.rerun()
                    except Exception as e:
                        st.toast(
                            f"Failed to fetch content from the link: {e}", icon="⚠️"
                        )
            else:
                st.toast("Please enter a link", icon="❗")


def main():
    """
    Main entry point for the chat application.

    Handles routing between the chats list page and individual chat pages:
    - If a chat_id is present in URL parameters, displays that specific chat
    - Otherwise shows the main chats listing page

    The function uses Streamlit query parameters to maintain state between page loads
    and determine which view to display.
    """
    query_params = st.query_params
    if "chat_id" in query_params:
        chat_id = query_params["chat_id"]
        chat_page(chat_id)
    else:
        chats_home()


if __name__ == "__main__":
    main()