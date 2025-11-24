from langchain_community.embeddings import HuggingFaceEmbeddings
from config import HF_EMBED_MODEL




def get_embedding_model():
    """Return a HuggingFaceEmbeddings instance."""
    return HuggingFaceEmbeddings(model_name=HF_EMBED_MODEL)