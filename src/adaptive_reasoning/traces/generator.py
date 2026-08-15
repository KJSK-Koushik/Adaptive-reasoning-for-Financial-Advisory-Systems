"""Reasoning trace generation with early-exit probing.

This is the core of the project. For each question we let the model reason to full
length and, at regular boundaries, ask "what would you answer if you stopped here?" -
recording the answer, whether it is correct, and how confident the model was.

The result is a table of *what would have happened had we stopped at step t*, for every
t. Because the whole future is known, Phase 5 can compute the reward for both CONTINUE
and STOP in closed form and train the DQN with **zero further LLM calls**.

Three engineering decisions carry this module:

**1. Probing reuses the KV cache.** The naive implementation re-runs the prompt plus all
reasoning so far for every probe, which is quadratic and makes 32 probes per question
unaffordable. Instead we append the probe suffix to the live cache, decode ~24 tokens,
then crop the cache back to its pre-probe length and carry on reasoning from exactly
where we left off. ``test_probing_does_not_perturb_reasoning`` asserts that the main
generation is bit-identical with and without probing - if that ever fails, every trace
is silently corrupted.

**2. Probes fire at fixed token intervals, not at text delimiters.** Delimiters
("\\n\\n", "Wait", "Therefore") would place boundaries at more semantically meaningful
points, but every sequence in a batch would then hit its boundaries at different times,
forcing batch size 1. Measured single-sequence decoding would put Phase 3 at roughly 85
GPU-hours instead of ~12. Fixed intervals keep the batch synchronised, which is the
difference between this being runnable on a free Kaggle quota and not. Textual cues are
not lost - they are captured as ``progress_cue``/``doubt_cue`` state features instead.

**3. Decoding is written out by hand rather than using ``model.generate``.** We need
per-token logits for entropy, per-token probabilities for confidence, and direct
control of the cache. ``generate`` gives none of those together.
"""

from __future__ import annotations

import torch

from ..config import Config
from ..grading import is_correct
from ..llm import ReasoningLLM
from ..logging_utils import get_logger
from ..prompting import ANSWER_MARKER, build_messages, extract_answer
from ..schema import QARecord, Trace, TraceStep

log = get_logger("traces.generator")


def crop_cache(cache, target_length: int) -> None:
    """Shorten a KV cache to ``target_length`` tokens, in place.

    ``DynamicCache.crop`` deprecated positive arguments in transformers 5.x in favour of
    a negative "remove this many" form, so the negative form is used and a manual slice
    is kept as a fallback for other cache implementations.
    """
    current = cache.get_seq_length()
    if current <= target_length:
        return
    remove = current - target_length

    try:
        cache.crop(-remove)
        if cache.get_seq_length() == target_length:
            return
    except (AttributeError, TypeError, NotImplementedError):
        pass

    # Fallback: slice each layer's tensors directly.
    layers = getattr(cache, "layers", None)
    if layers is not None:
        for layer in layers:
            layer.keys = layer.keys[:, :, :target_length, :]
            layer.values = layer.values[:, :, :target_length, :]
        return

    raise RuntimeError(
        f"cannot crop cache of type {type(cache).__name__}; Phase 3 needs a croppable "
        f"KV cache to probe without re-running the prompt"
    )


def _logits_to_keep_arg(model) -> str | None:
    """Name of the argument that limits how many positions get a logits row.

    ``transformers`` renamed ``num_logits_to_keep`` to ``logits_to_keep`` in 4.49, and
    older versions have neither, so the supported spelling is detected once.
    """
    import inspect

    try:
        parameters = inspect.signature(model.forward).parameters
    except (TypeError, ValueError):
        return None
    for name in ("logits_to_keep", "num_logits_to_keep"):
        if name in parameters:
            return name
    return None


def _entropy(logits: torch.Tensor) -> torch.Tensor:
    """Shannon entropy of the model's next-token distribution, per sequence.

    Computed on the raw logits, not the temperature-scaled ones: this measures the
    model's actual uncertainty, which is what the stopping policy should react to,
    rather than an artefact of the sampling configuration.
    """
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    return -(log_probs.exp() * log_probs).sum(dim=-1)


def _sample(logits: torch.Tensor, temperature: float, top_p: float) -> torch.Tensor:
    """Nucleus sampling. ``temperature <= 0`` means greedy."""
    if temperature <= 0:
        return logits.argmax(dim=-1)

    scaled = logits.float() / temperature
    probs = torch.softmax(scaled, dim=-1)

    if 0 < top_p < 1:
        ordered, indices = torch.sort(probs, descending=True, dim=-1)
        cumulative = ordered.cumsum(dim=-1)
        # Keep the smallest set whose mass exceeds top_p; always keep the first token.
        remove = cumulative - ordered > top_p
        ordered[remove] = 0.0
        ordered = ordered / ordered.sum(dim=-1, keepdim=True)
        picked = torch.multinomial(ordered, num_samples=1)
        return indices.gather(-1, picked).squeeze(-1)

    return torch.multinomial(probs, num_samples=1).squeeze(-1)


class TraceGenerator:
    """Generates reasoning traces with early-exit probes, in batches."""

    def __init__(self, llm: ReasoningLLM, cfg: Config):
        self.llm = llm
        self.cfg = cfg
        self.tokenizer = llm.tokenizer
        self.model = llm.model
        self.device = self.model.device

        probe_text = cfg.traces.probe_prompt or f"\n\n{ANSWER_MARKER}"
        self.probe_ids = self.tokenizer(
            probe_text, return_tensors="pt", add_special_tokens=False
        ).input_ids.to(self.device)

        self._logits_to_keep_arg = _logits_to_keep_arg(self.model)
        if not self._logits_to_keep_arg:
            log.warning(
                "this model's forward() accepts neither logits_to_keep nor "
                "num_logits_to_keep; prefill will materialise logits for every prompt "
                "position and may run out of memory at large batch sizes"
            )

    # ------------------------------------------------------------------ #
    # low-level stepping
    # ------------------------------------------------------------------ #
    def _forward(self, input_ids, attention_mask, cache, position_ids):
        kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "past_key_values": cache,
            "position_ids": position_ids,
            "use_cache": True,
        }
        # Only the final position's logits are ever used, but by default the model
        # projects *every* position through the vocabulary head. On the prefill pass
        # that is batch x prompt_len x vocab: at batch 32 with 1,100-token prompts and
        # a 152k vocab in fp16 it is a 10 GB tensor, which is what OOM'd the first
        # Kaggle run. Asking for one row cuts it to a few megabytes.
        if self._logits_to_keep_arg:
            kwargs[self._logits_to_keep_arg] = 1

        out = self.model(**kwargs)
        return out.logits[:, -1, :], out.past_key_values

    @torch.no_grad()
    def _probe(self, cache, attention_mask, next_position, batch_size):
        """Ask the model for its answer right now, then restore the cache.

        Returns ``(answers, confidences, min_confidences)``, one entry per sequence.
        """
        checkpoint = cache.get_seq_length()

        probe = self.probe_ids.expand(batch_size, -1)
        mask = torch.cat(
            [attention_mask, torch.ones_like(probe)], dim=1
        )
        positions = next_position + torch.arange(
            probe.shape[1], device=self.device
        ).unsqueeze(0)

        logits, cache = self._forward(probe, mask, cache, positions)

        collected: list[torch.Tensor] = []
        probabilities: list[torch.Tensor] = []
        position = positions[:, -1:] + 1

        for _ in range(self.cfg.traces.probe_max_tokens):
            # Probes are greedy: "what would you answer" must be deterministic, or the
            # recorded outcome would not be the one the policy is credited with.
            token = logits.argmax(dim=-1)
            probs = torch.softmax(logits.float(), dim=-1)
            probabilities.append(probs.gather(-1, token.unsqueeze(-1)).squeeze(-1))
            collected.append(token)

            token = token.unsqueeze(-1)
            mask = torch.cat([mask, torch.ones_like(token)], dim=1)
            logits, cache = self._forward(token, mask, cache, position)
            position = position + 1

        crop_cache(cache, checkpoint)

        tokens = torch.stack(collected, dim=1)
        confidence = torch.stack(probabilities, dim=1)

        answers, means, minimums = [], [], []
        eos = self.tokenizer.eos_token_id
        for row in range(batch_size):
            sequence = tokens[row].tolist()
            # Confidence is measured over the answer only, up to EOS or a newline -
            # trailing chatter would dilute it.
            length = len(sequence)
            for i, tok in enumerate(sequence):
                if tok == eos:
                    length = i
                    break
            length = max(length, 1)

            text = self.tokenizer.decode(sequence[:length], skip_special_tokens=True)
            answers.append(text.strip().split("\n")[0].strip())
            means.append(float(confidence[row, :length].mean()))
            minimums.append(float(confidence[row, :length].min()))

        return answers, means, minimums

    # ------------------------------------------------------------------ #
    # public API
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def generate(self, records: list[QARecord], probe: bool = True) -> list[Trace]:
        """Generate a trace for each record. ``probe=False`` skips probing entirely."""
        cfg = self.cfg
        batch = len(records)

        prompts = [self.llm.render(build_messages(r)) for r in records]
        encoded = self.tokenizer(prompts, return_tensors="pt", padding=True)
        input_ids = encoded.input_ids.to(self.device)
        attention_mask = encoded.attention_mask.to(self.device)

        # Left padding means position ids must come from the mask, not from arange.
        positions = (attention_mask.cumsum(dim=1) - 1).clamp(min=0)

        logits, cache = self._forward(input_ids, attention_mask, None, positions)
        next_position = positions[:, -1:] + 1

        eos = self.tokenizer.eos_token_id
        finished = torch.zeros(batch, dtype=torch.bool, device=self.device)
        generated: list[list[int]] = [[] for _ in range(batch)]
        entropy_sum = torch.zeros(batch, device=self.device)
        entropy_count = 0

        steps: list[list[TraceStep]] = [[] for _ in range(batch)]
        last_answer: list[str | None] = [None] * batch
        step_start = [0] * batch
        recorded: list[bool] = [False] * batch   # sequence has had its terminal step

        def record(rows, answers, confidences, minimums, mean_entropy, terminal):
            for row in rows:
                record_ = records[row]
                answer = extract_answer(answers[row]) or answers[row]
                text = self.tokenizer.decode(
                    generated[row][step_start[row]:], skip_special_tokens=True
                )
                step_start[row] = len(generated[row])

                steps[row].append(
                    TraceStep(
                        question_id=record_.id,
                        step_index=len(steps[row]),
                        tokens_so_far=len(generated[row]),
                        step_text=text,
                        probe_answer=answer,
                        probe_correct=is_correct(
                            answer, record_.gold_answer, str(record_.answer_type),
                            cfg.data.numeric_tolerance,
                        ),
                        confidence=confidences[row],
                        min_token_confidence=minimums[row],
                        entropy=mean_entropy[row],
                        answer_changed=last_answer[row] is not None
                        and answer != last_answer[row],
                        is_terminal=terminal(row),
                    )
                )
                last_answer[row] = answer
                if terminal(row):
                    recorded[row] = True

        for produced in range(1, cfg.llm.max_new_tokens + 1):
            entropy_sum += _entropy(logits)
            entropy_count += 1

            token = _sample(logits, cfg.llm.temperature, cfg.llm.top_p)
            token = torch.where(finished, torch.full_like(token, eos), token)

            for row in range(batch):
                if not finished[row]:
                    generated[row].append(int(token[row]))

            finished = finished | (token == eos)
            if bool(finished.all()):
                break

            token = token.unsqueeze(-1)
            attention_mask = torch.cat([attention_mask, torch.ones_like(token)], dim=1)
            logits, cache = self._forward(token, attention_mask, cache, next_position)
            next_position = next_position + 1

            at_boundary = produced % cfg.traces.step_tokens == 0
            room_for_more = max(len(s) for s in steps) < cfg.traces.max_steps - 1
            if probe and at_boundary and room_for_more:
                answers, confidences, minimums = self._probe(
                    cache, attention_mask, next_position, batch
                )
                mean_entropy = (entropy_sum / max(entropy_count, 1)).tolist()
                entropy_sum = torch.zeros(batch, device=self.device)
                entropy_count = 0

                record(
                    [r for r in range(batch) if not recorded[r]],
                    answers, confidences, minimums, mean_entropy,
                    # Bind the current tensor explicitly: a bare closure over the loop
                    # variable would silently read a later value if this ever became
                    # deferred rather than called immediately.
                    terminal=lambda r, done=finished: bool(done[r]),
                )

        # Every trace must end with a terminal step, even when the model answered
        # before the first boundary. A trace with no steps offers the policy no
        # decision at all, and that happens precisely on the easy questions where
        # learning to stop immediately matters most.
        outstanding = [r for r in range(batch) if not recorded[r]]
        if outstanding:
            if probe:
                answers, confidences, minimums = self._probe(
                    cache, attention_mask, next_position, batch
                )
            else:
                answers = [
                    self.tokenizer.decode(generated[r], skip_special_tokens=True)
                    for r in range(batch)
                ]
                confidences = minimums = [0.0] * batch
            mean_entropy = (entropy_sum / max(entropy_count, 1)).tolist()
            record(outstanding, answers, confidences, minimums, mean_entropy,
                   terminal=lambda r: True)

        return self._finalise(records, generated, steps)

    def _finalise(self, records, generated, steps) -> list[Trace]:
        traces = []
        for row, record in enumerate(records):
            text = self.tokenizer.decode(generated[row], skip_special_tokens=True)
            answer = extract_answer(text)
            if steps[row]:
                steps[row][-1].is_terminal = True

            traces.append(
                Trace(
                    question_id=record.id,
                    difficulty=record.difficulty,
                    total_tokens=len(generated[row]),
                    final_answer=answer,
                    final_correct=is_correct(
                        answer, record.gold_answer, str(record.answer_type),
                        self.cfg.data.numeric_tolerance,
                    ),
                    steps=steps[row],
                )
            )
        return traces
