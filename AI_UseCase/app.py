import streamlit as st
import os
import sys
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# Ensure parent directory (project root) is in Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

# old: from models.llm import get_hf_llm
from models.llm import get_hf_client, generate_text

from utils.rag import load_documents, create_vector_store, search
from utils.web_search import ddg_search


# ------------------------------
# CHAT RESPONSE FUNCTION
# ------------------------------
def get_chat_response(chat_client, messages, system_prompt):
    """Generate assistant response using the HF InferenceClient."""
    try:
        formatted_messages = [SystemMessage(content=system_prompt)]
        # add conversation
        for msg in messages:
            if msg["role"] == "user":
                formatted_messages.append(HumanMessage(content=msg["content"]))
            else:
                formatted_messages.append(AIMessage(content=msg["content"]))

        prompt_text = "\n".join([m.content for m in formatted_messages])

        # Use the generate_text helper to call the HF Inference API
        response_text = generate_text(chat_client, prompt_text)
        return response_text

    except Exception as e:
        return f"Error getting response: {str(e)}"

# ------------------------------
# INSTRUCTIONS PAGE
# ------------------------------
def instructions_page():
    st.title("The Chatbot Blueprint")
    st.markdown("""
    ## Instructions
    1. Set your `HF_API_KEY` using Streamlit Secrets or environment variables.
    2. Add `.txt` documents to the `docs/` folder.
    3. Run locally using:  
       ```
       streamlit run app.py
       ```
    4. Deploy on Streamlit Cloud by connecting GitHub.
    """)


# ------------------------------
# CHAT PAGE
# ------------------------------
def chat_page():
    st.title("🤖 AI ChatBot (HuggingFace RAG + Web Search)")

    # Load RAG vector store once
    if "vector_store" not in st.session_state:
        st.info("Loading documents for RAG...")
        docs = load_documents("docs")
        st.session_state.vector_store = create_vector_store(docs)

    # Load model
    try:
         chat_client = get_hf_client()
    except Exception as e:
        st.error(str(e))
        return

    # Init chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Render chat history
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

                # Build system prompt
                system_prompt = (
                    "You are an intelligent assistant. Use the provided context to answer.\n"
                    "If context is insufficient, also use web search results.\n\n"
                    f"=== DOCUMENT CONTEXT ===\n{context}\n"
                    "=========================\n"
                )

                # Web search results
                web_results = ddg_search(prompt)
                system_prompt += f"\n=== WEB SEARCH RESULTS ===\n{web_results}\n"

                # Generate response
               response = get_chat_response(
                    chat_client,
                    st.session_state.messages,
                    system_prompt
                )


                st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})


# ------------------------------
# MAIN APP
# ------------------------------
def main():
    st.set_page_config(
        page_title="LangChain HuggingFace ChatBot",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Sidebar navigation
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

    # Page routing
    if page == "Instructions":
        instructions_page()
    else:
        chat_page()


# ------------------------------
# RUN APP
# ------------------------------
if __name__ == "__main__":
    main()




