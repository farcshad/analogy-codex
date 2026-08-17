"""Reusable Hugging Face causal-language-model loader.

PyTorch and Transformers are imported only inside :func:`load_model`. Importing
this module therefore never initializes CUDA or downloads model files.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ModelConfig:
    """Settings that affect model/tokenizer loading on the GPU server."""

    model_id: str = "Qwen/Qwen3-0.6B"
    revision: str = "main"
    dtype: str = "bfloat16"
    device_map: str | dict[str, Any] = "auto"
    attention_implementation: str | None = None
    cache_dir: str | None = None
    local_files_only: bool = False
    trust_remote_code: bool = False
    require_cuda: bool = True

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class LoadedModel:
    """The loaded model and tokenizer together with their immutable settings."""

    model: Any
    tokenizer: Any
    config: ModelConfig


def _resolve_dtype(torch: Any, name: str) -> Any:
    normalized = name.lower().replace("torch.", "")
    supported = {
        "auto": "auto",
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    if normalized not in supported:
        choices = ", ".join(supported)
        raise ValueError(f"Unsupported dtype {name!r}; choose one of: {choices}")
    return supported[normalized]


def load_model(config: ModelConfig | None = None) -> LoadedModel:
    """Load a tokenizer and causal LM without making any test generation.

    Authentication is picked up by Hugging Face automatically. Setting either
    ``HF_TOKEN`` or logging in with ``huggingface-cli login`` is sufficient for
    gated models; Qwen3-0.6B itself does not require a token.
    """

    config = config or ModelConfig()
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "GPU dependencies are missing. Install gpu_experiments/requirements.txt "
            "inside the server environment."
        ) from exc

    if config.require_cuda and not torch.cuda.is_available():
        raise RuntimeError(
            "No CUDA GPU is visible to PyTorch. Check the server's CUDA/PyTorch "
            "installation, or set require_cuda=False for an intentional CPU run."
        )

    common = {
        "revision": config.revision,
        "cache_dir": config.cache_dir,
        "local_files_only": config.local_files_only,
        "trust_remote_code": config.trust_remote_code,
    }
    token = os.environ.get("HF_TOKEN")
    if token:
        common["token"] = token

    tokenizer = AutoTokenizer.from_pretrained(config.model_id, **common)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {
        **common,
        # torch_dtype works across the full supported Transformers range. Newer
        # releases also accept the shorter `dtype` spelling.
        "torch_dtype": _resolve_dtype(torch, config.dtype),
        "device_map": config.device_map,
    }
    if config.attention_implementation:
        model_kwargs["attn_implementation"] = config.attention_implementation

    model = AutoModelForCausalLM.from_pretrained(config.model_id, **model_kwargs)
    model.eval()
    return LoadedModel(model=model, tokenizer=tokenizer, config=config)
