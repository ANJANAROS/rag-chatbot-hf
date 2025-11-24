import os
from huggingface_hub import InferenceClient
from config.config import HF_API_KEY, HF_LLM_MODEL, HF_TEMPERATURE, HF_MAX_NEW_TOKENS


def get_hf_llm():
    """Return an HF Inference Client for text generation."""
    if not HF_API_KEY:
        raise EnvironmentError("HF_API_KEY is not set in environment variables.")

    client = InferenceClient(
        model=HF_LLM_MODEL,
        token=HF_API_KEY
    )

    return client


def generate_text(client, prompt):
    """Use the new HF API to generate text properly."""
    response = client.text_generation(
        prompt,
        max_new_tokens=HF_MAX_NEW_TOKENS,
        temperature=HF_TEMPERATURE,
        stream=False
    )
    return response
