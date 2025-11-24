import streamlit as st
import os
import sys
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from models.llm import get_hf_client, generate_text
from utils.rag import load_documents, create_vector_store, search
from utils.web_search import ddg_search


# ----------------------------------------
# CHAT RESPONSE FUNCTION
# ----------------------------------------
def get_chat_response(chat_client, messages, system_prompt):
    try:
        formatted_messages = [SystemMessage(content=system_prompt)]

        # Add history
        for msg in messages:
            if msg["role"] == "user":
                formatted_messages.append(HumanMessage(content=msg["content"]))
            else:
                formatted_messages.append(AIMessage(content=msg["content"]))

        prompt_text = "\n".join([m.content for m in formatted_messages])

        response_text = generate_text(chat_client, prompt_text)
        return response_text

    except Exception as e:
        return f"Error getting response: {str(e)}"


# ----------------------------------------
# INSTRUCTIONS PAGE
# ----------------------------------------
def instructions_page():
    st.title("Chatbot Blueprint")
    st.markdown("""
    ## Setup
    1. Set your `HF_API_KEY` in Streamlit Secrets or environment variables.
    2. Add `.txt` files in `/docs`.
    3. Run:  
       ```
       streamlit run app.py
       ```
    """)


# ----------------------------------------
# CHAT PAGE
# ----------------------------------------
def chat_page():
    st.title("🤖 AI ChatBot (HuggingFace RAG + Web Search)")

    # Load documents once
    if "vector_store" not in st.session_state:
        st.info("Loading documents...")
        docs = load_documents("docs")
        st.session_state.vector_store = create_vector_store(docs)

    # Load HF model
    try:
        chat_client = get_hf_client()
    except Exception as e:
        st.error(str(e))
        return

    # Chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Type your message here…"):

        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):

                # RAG retrieval
                relevant_docs = search(prompt, st.session_state.vector_store)
                context = "\n\n".join(
                    [f"Source: {d['source']}\n{d['text']}" for d in relevant_docs]
                )

                system_prompt = (
                    "Use this context to answer. If insufficient, also use web search.\n\n"
                    f"=== DOCUMENT CONTEXT ===\n{context}\n"
                    "=========================\n"
                )

                web_results = ddg_search(prompt)
                system_prompt += f"\n=== WEB SEARCH RESULTS ===\n{web_results}\n"

                # ⭐ Correct: using chat_client, not chat_model
                response = get_chat_response(
                    chat_client,
                    st.session_state.messages,
                    system_prompt
                )

                st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})


# ----------------------------------------
# MAIN APP
# ----------------------------------------
def main():
    st.set_page_config(
        page_title="HuggingFace RAG ChatBot",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )

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

    if page == "Instructions":
        instructions_page()
    else:
        chat_page()


if __name__ == "__main__":
    main()
