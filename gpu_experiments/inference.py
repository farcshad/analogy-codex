"""Batched chat generation for locally loaded Hugging Face models."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Sequence

from .model_loader import LoadedModel


@dataclass(frozen=True)
class GenerationConfig:
    max_new_tokens: int = 1024
    temperature: float = 0.0
    top_p: float = 0.95
    enable_thinking: bool = False


def _input_device(model: Any) -> Any:
    """Return the device where input embeddings live, including device_map models."""
    embeddings = model.get_input_embeddings()
    return next(embeddings.parameters()).device


def generate_batch(
    loaded: LoadedModel,
    prompts: Sequence[str],
    config: GenerationConfig,
) -> list[dict[str, Any]]:
    """Generate one assistant response per prompt and report token counts."""

    if not prompts:
        return []
    if config.max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    if config.temperature < 0:
        raise ValueError("temperature must be non-negative")

    import torch

    tokenizer = loaded.tokenizer
    model = loaded.model
    conversations = [[{"role": "user", "content": prompt}] for prompt in prompts]
    template_kwargs = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    # Qwen3 consumes this argument. Other chat templates may reject it, so fall
    # back cleanly when a future local model does not expose thinking control.
    try:
        rendered = [
            tokenizer.apply_chat_template(
                messages, enable_thinking=config.enable_thinking, **template_kwargs
            )
            for messages in conversations
        ]
    except TypeError:
        rendered = [
            tokenizer.apply_chat_template(messages, **template_kwargs)
            for messages in conversations
        ]

    inputs = tokenizer(rendered, return_tensors="pt", padding=True)
    inputs = {name: tensor.to(_input_device(model)) for name, tensor in inputs.items()}
    input_width = inputs["input_ids"].shape[1]
    prompt_lengths = inputs["attention_mask"].sum(dim=1).tolist()
    generation_kwargs = {
        "max_new_tokens": config.max_new_tokens,
        "do_sample": config.temperature > 0,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "use_cache": True,
    }
    if config.temperature > 0:
        generation_kwargs.update(temperature=config.temperature, top_p=config.top_p)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        output_ids = model.generate(**inputs, **generation_kwargs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started

    generated = output_ids[:, input_width:]
    texts = tokenizer.batch_decode(generated, skip_special_tokens=True)
    results = []
    for index, (text, token_ids) in enumerate(zip(texts, generated)):
        completion_tokens = int((token_ids != tokenizer.pad_token_id).sum().item())
        results.append(
            {
                "text": text.strip(),
                "prompt_tokens": int(prompt_lengths[index]),
                "completion_tokens": completion_tokens,
                "latency_seconds": round(elapsed, 3),
                "batch_size": len(prompts),
            }
        )
    return results
