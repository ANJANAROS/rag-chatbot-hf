import os

# HuggingFace token (MUST be set in environment variables or Streamlit secrets)
HF_API_KEY = os.getenv("HF_API_KEY")

# LLM model on HuggingFace Hub
HF_LLM_MODEL = os.getenv("HF_LLM_MODEL", "meta-llama/Llama-3.2-3B-Instruct")

# Embedding model
HF_EMBED_MODEL = os.getenv("HF_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# LLM generation defaults
HF_TEMPERATURE = float(os.getenv("HF_TEMPERATURE", "0.7"))
HF_MAX_NEW_TOKENS = int(os.getenv("HF_MAX_NEW_TOKENS", "512"))
