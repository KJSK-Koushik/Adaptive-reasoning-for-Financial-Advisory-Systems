"""Reasoning-model loading and batched generation.

Used by Phase 2 (difficulty sampling) and Phase 3 (trace generation). Kept device
agnostic so the identical code runs on this CPU laptop for smoke tests and on a Kaggle
GPU for the real runs - no forked logic between the two, which is what keeps the
remote results reproducible locally.
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import Config
from .device import detect, torch_dtype
from .logging_utils import get_logger

log = get_logger("llm")


@dataclass
class Generation:
    """One completion plus the token accounting the RL features are built from."""

    text: str
    n_prompt_tokens: int
    n_generated_tokens: int
    finished: bool          # False when generation stopped at the token cap


class ReasoningLLM:
    """Thin wrapper over a HuggingFace causal LM.

    Deliberately not a subclass and not clever: Phase 3 needs direct access to the
    model, tokenizer and KV cache for early-exit probing, so this only owns loading
    and plain batched generation.
    """

    def __init__(self, cfg: Config, model_id: str | None = None):
        self.cfg = cfg
        self.model_id = model_id or cfg.llm.model_id
        self.hw = detect(cfg.project.device)
        self._model = None
        self._tokenizer = None

    # -- loading ----------------------------------------------------------- #
    @property
    def tokenizer(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            log.info("loading tokenizer %s", self.model_id)
            tok = AutoTokenizer.from_pretrained(self.model_id)
            if tok.pad_token is None:
                # Decoder-only models often ship without a pad token; reusing EOS is
                # standard and harmless as long as attention masks are honoured.
                tok.pad_token = tok.eos_token
            # Left padding is required for batched generation on decoder-only models,
            # otherwise the generated continuation starts after the padding.
            tok.padding_side = "left"
            self._tokenizer = tok
        return self._tokenizer

    @property
    def model(self):
        if self._model is None:
            import torch
            from transformers import AutoModelForCausalLM

            dtype = torch_dtype(self.cfg.llm.dtype, self.hw.backend)
            log.info(
                "loading %s on %s (%s)", self.model_id, self.hw.backend, str(dtype).split(".")[-1]
            )

            kwargs: dict = {"dtype": dtype}

            # Streaming weights to the device avoids ever holding the full model in
            # host RAM. Required for anything above ~3B on a Kaggle session.
            device_map = getattr(self.cfg.llm, "device_map", None)
            if device_map and self.hw.backend == "cuda":
                kwargs["device_map"] = device_map
                kwargs["low_cpu_mem_usage"] = True
            if self.cfg.llm.load_in_4bit and self.hw.backend == "cuda":
                from transformers import BitsAndBytesConfig

                kwargs = {
                    "quantization_config": BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=dtype,
                        bnb_4bit_quant_type="nf4",
                    )
                }

            try:
                model = AutoModelForCausalLM.from_pretrained(self.model_id, **kwargs)
            except TypeError:
                # transformers < 5 spells the argument torch_dtype.
                kwargs["torch_dtype"] = kwargs.pop("dtype", dtype)
                model = AutoModelForCausalLM.from_pretrained(self.model_id, **kwargs)

            # A model dispatched by accelerate is already placed, and moving it
            # afterwards undoes the dispatch (and re-materialises it on one device).
            dispatched = "device_map" in kwargs or "quantization_config" in kwargs
            if not dispatched:
                model = model.to(self.hw.device)
            model.eval()
            torch.set_grad_enabled(False)
            self._model = model
        return self._model

    # -- generation -------------------------------------------------------- #
    def render(self, messages: list[dict[str, str]]) -> str:
        """Apply the model's chat template, leaving it ready to continue."""
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def generate(
        self,
        prompts: list[str],
        max_new_tokens: int,
        temperature: float | None = None,
        top_p: float | None = None,
        do_sample: bool = True,
    ) -> list[Generation]:
        """Generate a completion for each prompt.

        Prompts are already chat-templated strings, not message lists, so callers can
        append a probe suffix before generating.
        """
        import torch

        tok = self.tokenizer
        encoded = tok(prompts, return_tensors="pt", padding=True, truncation=False)
        encoded = {k: v.to(self.model.device) for k, v in encoded.items()}
        prompt_len = encoded["input_ids"].shape[1]

        kwargs: dict = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": tok.pad_token_id,
            "do_sample": do_sample,
        }
        if do_sample:
            kwargs["temperature"] = temperature if temperature is not None else self.cfg.llm.temperature
            kwargs["top_p"] = top_p if top_p is not None else self.cfg.llm.top_p

        with torch.no_grad():
            out = self.model.generate(**encoded, **kwargs)

        results: list[Generation] = []
        for row in out:
            new_tokens = row[prompt_len:]
            # Trim right-hand padding so the token count is the real generation length.
            keep = new_tokens != tok.pad_token_id
            n_generated = int(keep.sum()) if keep.any() else 0
            text = tok.decode(new_tokens, skip_special_tokens=True)
            results.append(
                Generation(
                    text=text,
                    n_prompt_tokens=prompt_len,
                    n_generated_tokens=n_generated,
                    finished=n_generated < max_new_tokens,
                )
            )
        return results

    def batched(self, prompts: list[str], batch_size: int | None = None, **kwargs):
        """Generate over ``prompts`` in batches, yielding ``(index, Generation)``."""
        size = batch_size or self.cfg.llm.batch_size
        for start in range(0, len(prompts), size):
            chunk = prompts[start : start + size]
            for offset, generation in enumerate(self.generate(chunk, **kwargs)):
                yield start + offset, generation
