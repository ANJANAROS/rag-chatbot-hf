# models/llm.py
import os
from huggingface_hub import InferenceClient
from config.config import HF_API_KEY, HF_LLM_MODEL, HF_TEMPERATURE, HF_MAX_NEW_TOKENS

def get_hf_client():
    """
    Return an InferenceClient configured for the selected model.
    The HF_API_KEY must be provided via environment variable or Streamlit secrets.
    """
    if not HF_API_KEY:
        raise EnvironmentError("HF_API_KEY is not set. Add it to your environment or Streamlit Secrets.")
    # Create client with token and default model; newer huggingface_hub lets you pass model here.
    client = InferenceClient(model=HF_LLM_MODEL, token=HF_API_KEY)
    return client

def generate_text(client: InferenceClient, prompt: str) -> str:
    """
    Produce text using the InferenceClient.text_generation API.
    Returns a plain string.
    """
    # The client.text_generation returns a high-level response object (list or dict depending on model).
    # We'll extract the generated text safely.
    response = client.text_generation(
        prompt,
        max_new_tokens=HF_MAX_NEW_TOKENS,
        temperature=HF_TEMPERATURE,
        stream=False  # set True if you want streaming (more complex)
    )

    # response might be a list of dict(s) or a dict depending on the model; handle common shapes:
    if isinstance(response, list) and len(response) > 0:
        # e.g. [{'generated_text': '...'}]
        first = response[0]
        if isinstance(first, dict) and "generated_text" in first:
            return first["generated_text"]
        # handle other possible keys
        return str(first)
    if isinstance(response, dict):
        # e.g. {'generated_text': '...'}
        if "generated_text" in response:
            return response["generated_text"]
        # sometimes nested
        for v in response.values():
            if isinstance(v, str):
                return v
        return str(response)

    # fallback to string conversion
    return str(response)
