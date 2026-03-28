"""LangChain-based LLM inference using free Hugging Face models."""

from __future__ import annotations

import threading
from typing import Generator, Optional

from config import settings
from src.logging_utils import setup_logger

logger = setup_logger("llm_langchain", settings.log_level)

_llm_lock = threading.Lock()
_pipe = None


def load_pipeline():
    global _pipe
    with _llm_lock:
        if _pipe is None:
            try:
                from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

                logger.info("Loading Hugging Face model: %s", settings.langchain_model_name)

                model = AutoModelForSeq2SeqLM.from_pretrained(settings.langchain_model_name)
                tokenizer = AutoTokenizer.from_pretrained(settings.langchain_model_name)

                _pipe = {"model": model, "tokenizer": tokenizer}
            except Exception as e:
                logger.error("Failed to load Hugging Face model: %s", e)
                return None
        return _pipe


def stream_generate(system_prompt: str, user_prompt: str) -> Generator[str, None, None]:
    pipe = load_pipeline()
    if pipe is None:
        raise RuntimeError("Hugging Face pipeline failed to load")

    # For T5 models, format as instruction
    full_prompt = f"Answer the following question: {user_prompt}"

    try:
        model = pipe["model"]
        tokenizer = pipe["tokenizer"]

        inputs = tokenizer(full_prompt, return_tensors="pt", max_length=512, truncation=True)
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # For T5, the generated text is the answer
        response = generated_text

        # Yield the response as a single chunk
        yield response

    except Exception as e:
        logger.error("Error in Hugging Face generation: %s", e)
        raise e