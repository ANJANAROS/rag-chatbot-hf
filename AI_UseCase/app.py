import streamlit as st
import os
import sys
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))


from models.llm import get_hf_llm
from utils.rag import load_documents, create_vector_store, search
from utils.web_search import ddg_search


def get_chat_response(chat_model, messages, system_prompt):
    """Get response from the chat model"""
    try:
        # Prepare messages for the model
        formatted_messages = [SystemMessage(content=system_prompt)]


        # Add conversation history
        for msg in messages:
            if msg["role"] == "user":
                formatted_messages.append(HumanMessage(content=msg["content"]))
            else:
                formatted_messages.append(AIMessage(content=msg["content"]))


        # Invoke the model via its simple API
        # For HuggingFaceHub, we use `.call` or simply call with a prompt depending on wrapper
        # We'll use the `call`/`__call__` convention used by LangChain LLM wrappers
        prompt_text = "\n".join([m.content for m in formatted_messages])
        response = chat_model(prompt_text)


        # Many wrappers return a string directly
        if isinstance(response, str):
            return response


        # If the wrapper returns an object with `.content` attribute
        if hasattr(response, "content"):
            return response.content
        # Fallback: convert to string
        return str(response)


    except Exception as e:
        return f"Error getting response: {str(e)}"

def instructions_page():
    st.title("The Chatbot Blueprint")
    st.markdown("Welcome! Follow these instructions to set up and use the chatbot.")
    st.markdown("""
    - Make sure you set `HF_API_KEY` environment variable.
    - Add `.txt` files in the `docs/` folder.
    - Run `streamlit run app.py`.
    """)

def chat_page():
    st.title("🤖 AI ChatBot (HuggingFace)")


    # Load RAG vector store once
    if "vector_store" not in st.session_state:
        st.info("Loading documents for RAG...")
        docs = load_documents("docs")
        st.session_state.vector_store = create_vector_store(docs)


    # Get model instance
    try:
        chat_model = get_hf_llm()
    except Exception as e:
        st.error(str(e))
        return

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Type your message here..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Getting response..."):
                # 1. RAG: Retrieve document context
                relevant_docs = search(prompt, st.session_state.vector_store)
                context = "\n\n".join([f"Source: {d['source']}\n{d['text']}" for d in relevant_docs])

                # 2. System prompt with document context
                system_prompt = (
                    "You are an intelligent assistant. Use the provided context to answer. "
                    "If context is insufficient, use web search results too.\n\n"
                    f"=== DOCUMENT CONTEXT ===\n{context}\n"
                    "=========================\n"
                )

                # 3. Web Search (DuckDuckGo)
                web_results = ddg_search(prompt)
                system_prompt += f"\n=== WEB SEARCH RESULTS ===\n{web_results}\n"

                # 4. Generate response from LLM
                response = get_chat_response(
                    chat_model,
                    st.session_state.messages,
                    system_prompt
                )

                st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})


def main():
    st.set_page_config(
        page_title="LangChain Multi-Provider ChatBot (HuggingFace)",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Sidebar
    with st.sidebar:
        st.title("Navigation")
        page = st.radio("Go to:", ["Chat", "Instructions"], index=0)
        st.divider()

        response_mode = st.radio("Response Mode", ["Concise", "Detailed"], index=1)
        st.session_state.response_mode = response_mode

        st.divider()
        if page == "Chat":
            if st.button("🗑️ Clear Chat History", use_container_width=True):
                st.session_state.messages = []
