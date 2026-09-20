"""Causal-LM subclass that patches generate() to beam-search, SID-constrained decoding for RL rollouts."""

from typing import Any

from transformers import AutoModelForCausalLM
from trl import GRPOConfig

from hcgrec.util import print_config_table


def _make_mimigenrec_generate(
    model: Any,
    original_generate: Any,
    *,
    logits_processor=None,
    temperature=1.0,
    top_p=1.0,
    top_k=50,
    num_beams=50,
    max_completion_length=128,
    repetition_penalty=1.0,
    do_sample=False,
):
    _gen_config = model.generation_config
    _gen_config = _gen_config.to_dict() if _gen_config else {}
    _gen_config.pop("max_length", None)

    def _reset_logits_processor_state(proc: Any) -> None:
        if proc is None:
            return
        if hasattr(proc, "reset"):
            proc.reset()
            return
        if hasattr(proc, "count"):
            proc.count = 0

    def _reset_logits_processor_collection_state(proc: Any) -> None:
        if proc is None:
            return
        if isinstance(proc, (list, tuple)):
            for item in proc:
                _reset_logits_processor_state(item)
            return
        try:
            iterator = iter(proc)
        except TypeError:
            _reset_logits_processor_state(proc)
            return
        for item in iterator:
            _reset_logits_processor_state(item)

    def generate(*args, **kwargs):
        kwargs.pop("token_type_ids", None)
        proc = kwargs.get("logits_processor")
        if proc is None:
            proc = getattr(model, "_logits_processor", None)
        if proc is None:
            proc = logits_processor
        if proc is not None:
            _reset_logits_processor_collection_state(proc)
            kwargs["logits_processor"] = proc
        merged = {
            **_gen_config,
            **dict(
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                num_beams=num_beams,
                num_return_sequences=num_beams,
                max_new_tokens=max_completion_length,
                repetition_penalty=repetition_penalty,
                do_sample=do_sample,
            ),
        }
        kwargs.update(merged)
        if "input_ids" in kwargs and kwargs["input_ids"].dim() >= 2:
            bs = kwargs["input_ids"].shape[0] // num_beams
            dim = kwargs["input_ids"].shape[-1]
            kwargs["input_ids"] = kwargs["input_ids"].view(bs, num_beams, dim)[:, 0, :]
            kwargs["attention_mask"] = kwargs["attention_mask"].view(bs, num_beams, dim)[:, 0, :]
        return original_generate(*args, **kwargs)

    return generate


def _print_config_assignments(config: Any) -> None:
    attrs = {k: v for k, v in vars(config).items() if not k.startswith("_")}
    print_config_table("Current configuration", attrs)


def get_grpo_config(**kwargs) -> GRPOConfig:
    _orig_post_init = getattr(GRPOConfig, "__post_init__", None)
    if getattr(GRPOConfig, "_grpo_config_print_patched", False):
        return GRPOConfig(**kwargs)

    def _patched_post_init(self: Any) -> None:
        if _orig_post_init is not None:
            try:
                _orig_post_init(self)
            except AttributeError:
                pass
        _print_config_assignments(self)

    GRPOConfig.__post_init__ = _patched_post_init
    GRPOConfig._grpo_config_print_patched = True
    return GRPOConfig(**kwargs)


class MIMIGenRec(AutoModelForCausalLM):
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        *args,
        logits_processor=None,
        temperature=1.0,
        top_p=1.0,
        top_k=50,
        num_beams=50,
        max_completion_length=128,
        repetition_penalty=1.0,
        do_sample=False,
        **kwargs,
    ):
        model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
        _original_generate = model.generate
        model.generate = _make_mimigenrec_generate(
            model,
            _original_generate,
            logits_processor=logits_processor,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            max_completion_length=max_completion_length,
            repetition_penalty=repetition_penalty,
            do_sample=do_sample,
        )
        return model
