"""Shared helpers: main-process logging, rich config tables, and constrained logits-processor builders."""

import logging
import os
import re
import warnings
from typing import Any, Optional

from rich.console import Console
from rich.table import Table
from transformers import AutoTokenizer, LogitsProcessorList

from hcgrec.evaluate import build_trie_from_index, create_prefix_allowed_tokens_fn
from hcgrec.fixed_hint_logit_processor import FixedHintConstrainedLogitsProcessor
from hcgrec.logit_processor import ConstrainedLogitsProcessor


def _format_config_value(val: Any, max_str_len: int = 120, max_seq_len: int = 5) -> str:
    if isinstance(val, (list, dict)) and len(val) > max_seq_len:
        return f"<{type(val).__name__} len={len(val)}>"
    if isinstance(val, str) and len(val) > max_str_len:
        return repr(val[:max_str_len] + "...")
    return repr(val)


def is_main_process() -> bool:
    for key in ("RANK", "LOCAL_RANK", "ACCELERATE_PROCESS_INDEX", "PROCESS_INDEX"):
        value = os.environ.get(key)
        if value is None:
            continue
        try:
            return int(value) == 0
        except ValueError:
            continue
    return True


def print_main_process(*args: Any, **kwargs: Any) -> None:
    if is_main_process():
        print(*args, **kwargs)


def quiet_non_main_process_logging() -> None:
    if is_main_process():
        return

    try:
        from transformers.utils import logging as transformers_logging

        transformers_logging.set_verbosity_error()
    except Exception:
        logging.getLogger("transformers").setLevel(logging.ERROR)

    warnings.filterwarnings(
        "ignore",
        message=r"`generation_config` default values have been modified to match model-specific defaults: .*",
    )


def print_config_table(title: str, attrs: dict[str, Any]) -> None:
    if not attrs or not is_main_process():
        return
    console = Console()
    table = Table(title=title, show_lines=False)
    table.add_column("config", style="cyan", no_wrap=True)
    table.add_column("value", style="green")
    for key in sorted(attrs.keys()):
        table.add_row(key, _format_config_value(attrs[key]))
    console.print(table)


def build_constrained_logits_processor(
    index_path: str,
    tokenizer: AutoTokenizer,
    prefix: Optional[str] = None,
    num_beams: int = 50,
    sid_levels: int = -1,
) -> LogitsProcessorList:
    """Build Trie from index and return a LogitsProcessorList with ConstrainedLogitsProcessor."""
    print_main_process(f"Building Trie from {index_path}...")
    trie, prompt_suffix_ids, prefix_index = build_trie_from_index(
        index_path,
        tokenizer,
        prefix=prefix,
        sid_levels=sid_levels,
    )
    print_main_process(f"Trie built: prefix_index={prefix_index}, num_items={len(trie)}")

    prefix_allowed_tokens_fn = create_prefix_allowed_tokens_fn(trie, prompt_suffix_ids)
    logits_processor = LogitsProcessorList(
        [
            ConstrainedLogitsProcessor(
                prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
                num_beams=num_beams,
                prefix_index=prefix_index,
                prefix_ids=prompt_suffix_ids,
                eos_token_id=tokenizer.eos_token_id,
            )
        ]
    )
    return logits_processor


def build_fixed_hint_constrained_logits_processor(
    index_path: str,
    tokenizer: AutoTokenizer,
    prefix: Optional[str] = None,
    num_beams: int = 50,
    sid_levels: int = -1,
) -> LogitsProcessorList:
    print_main_process(f"Building FixedHint Trie from {index_path}...")
    trie, prompt_suffix_ids, prefix_index = build_trie_from_index(
        index_path,
        tokenizer,
        prefix=prefix,
        sid_levels=sid_levels,
    )
    print_main_process(f"FixedHint Trie built: prefix_index={prefix_index}, num_items={len(trie)}")

    prefix_allowed_tokens_fn = create_prefix_allowed_tokens_fn(trie, prompt_suffix_ids)
    logits_processor = LogitsProcessorList(
        [
            FixedHintConstrainedLogitsProcessor(
                prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
                num_beams=num_beams,
                prefix_index=prefix_index,
                prefix_ids=prompt_suffix_ids,
                eos_token_id=tokenizer.eos_token_id,
            )
        ]
    )
    return logits_processor


def _extract_number(text: str) -> list[int]:
    return [int(m) for m in re.findall(r"_(\d+)", text)]
