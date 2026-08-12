from collections.abc import Callable
from typing import Any

import torch
from transformers.generation import LogitsProcessor


def find_last_prefix_match_start(sent: list[int], prefix_ids: list[int]) -> int:
    if not prefix_ids:
        return 0
    prefix_len = len(prefix_ids)
    for start in range(len(sent) - prefix_len, -1, -1):
        if sent[start : start + prefix_len] == prefix_ids:
            return start
    raise ValueError("Failed to find prompt suffix prefix_ids inside current sequence.")


def find_valid_prefix_match_start(
    sent: list[int],
    prefix_ids: list[int],
    prefix_allowed_tokens_fn: Callable[[list[int]], list[int]],
) -> int:
    if not prefix_ids:
        return 0

    prefix_len = len(prefix_ids)
    candidate_starts: list[int] = []
    for start in range(len(sent) - prefix_len + 1):
        if sent[start : start + prefix_len] == prefix_ids:
            candidate_starts.append(start)

    for start in candidate_starts:
        candidate_prefix_ids = sent[start:]
        if prefix_allowed_tokens_fn(candidate_prefix_ids):
            return start

    raise ValueError(
        "Failed to find a valid trie prefix match inside current sequence. "
        f"prefix_ids={prefix_ids}, candidate_starts={candidate_starts}, sent={sent}"
    )


def resolve_allowed_tokens_for_sequence(
    sent: list[int],
    prefix_ids: list[int],
    prefix_allowed_tokens_fn: Callable[[list[int]], list[int]],
    eos_token_id: int | None,
) -> list[int]:
    if eos_token_id is not None and sent and sent[-1] == eos_token_id:
        return [eos_token_id]

    start = find_valid_prefix_match_start(
        sent,
        prefix_ids,
        prefix_allowed_tokens_fn,
    )
    constraint_prefix_ids = sent[start:]
    prefix_allowed_tokens = prefix_allowed_tokens_fn(constraint_prefix_ids)
    if len(prefix_allowed_tokens) == 0:
        raise AssertionError("No valid tokens for prefix_ids")
    return prefix_allowed_tokens


class FixedHintConstrainedLogitsProcessor(LogitsProcessor):
    def __init__(
        self,
        prefix_allowed_tokens_fn: Callable[[int, list[int]], list[int]],
        num_beams: int,
        prefix_index: int = 3,
        prefix_ids: list[int] = None,
        eos_token_id: int = None,
    ):
        self._prefix_allowed_tokens_fn = prefix_allowed_tokens_fn
        self._num_beams = num_beams
        self.prefix_index = prefix_index
        self.eos_token_id = eos_token_id
        self.prefix_ids = prefix_ids

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        scores = torch.nn.functional.log_softmax(scores, dim=-1)
        mask = torch.full_like(scores, float("-inf"))
        assert input_ids.dim() == 2, "input_ids must be a 2D tensor"
        seq_len = input_ids.shape[1]
        beam_sents = input_ids.view(-1, self._num_beams, seq_len)
        for batch_id, beam_sent in enumerate(beam_sents):
            for beam_id, sent in enumerate[Any](beam_sent):
                sent_list = sent.tolist()
                prefix_allowed_tokens = resolve_allowed_tokens_for_sequence(
                    sent_list,
                    self.prefix_ids,
                    self._prefix_allowed_tokens_fn,
                    self.eos_token_id,
                )
                idx = batch_id * self._num_beams + beam_id
                mask[idx, prefix_allowed_tokens] = 0
        return scores + mask
