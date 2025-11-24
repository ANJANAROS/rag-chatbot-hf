from langchain_community.llms import HuggingFaceHub
from config import HF_API_KEY, HF_LLM_MODEL, HF_TEMPERATURE, HF_MAX_NEW_TOKENS




def get_hf_llm():
    """Return an LLM instance connected to HuggingFace Hub.


    This requires HF_API_KEY to be set as an environment variable.
    """
    if not HF_API_KEY:
        raise EnvironmentError("HF_API_KEY is not set. Please set it in your environment.")


    return HuggingFaceHub(
        repo_id=HF_LLM_MODEL,
        huggingfacehub_api_token=HF_API_KEY,
        model_kwargs={
        "temperature": HF_TEMPERATURE,
        "max_new_tokens": HF_MAX_NEW_TOKENS,
            }
        )