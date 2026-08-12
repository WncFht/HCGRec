from __future__ import annotations

import re
from collections import Counter


ORIGINAL_TOKEN_PREFIX = "[ORIG]"
ADDED_TOKEN_PREFIX = "ADDED_TOKEN:"


def bytes_to_unicode() -> dict[int, str]:
    """Build a reversible mapping between bytes and unicode strings."""
    bs = (
        list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for byte in range(2**8):
        if byte not in bs:
            bs.append(byte)
            cs.append(2**8 + n)
            n += 1
    return dict(zip(bs, [chr(codepoint) for codepoint in cs], strict=False))


def unicode_to_bytes() -> dict[str, bytes]:
    return {value: bytes([key]) for key, value in bytes_to_unicode().items()}


def get_vocab_dict(tokenizer) -> dict[str, int]:
    if hasattr(tokenizer, "get_vocab"):
        return tokenizer.get_vocab()
    if hasattr(tokenizer, "vocab"):
        return tokenizer.vocab
    raise TypeError("Tokenizer does not expose a vocabulary mapping")


def _decode_token_string(token: str, unicode_to_bytes_map: dict[str, bytes]) -> str:
    try:
        token_bytes = b"".join(unicode_to_bytes_map[ch] for ch in token)
    except KeyError:
        return token

    try:
        return token_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return f"INVALID UTF-8: {token_bytes!r}"


def convert_to_readable_vocab(tokenizer, verbose: bool = False) -> dict[int, str]:
    """Convert tokenizer vocabulary into readable token strings."""
    from tqdm import tqdm

    vocab = get_vocab_dict(tokenizer)
    reversed_vocab = {token_id: token for token, token_id in vocab.items()}
    added_tokens = set(getattr(tokenizer, "added_tokens_encoder", {}))
    unicode_to_bytes_map = unicode_to_bytes()

    readable_vocab = {}
    for token_id in tqdm(
        sorted(reversed_vocab),
        desc="Converting to readable vocab",
        disable=not verbose,
    ):
        token = reversed_vocab[token_id]
        if token in added_tokens:
            readable_vocab[token_id] = f"{ADDED_TOKEN_PREFIX} {token}"
            continue
        readable_vocab[token_id] = _decode_token_string(token, unicode_to_bytes_map)

    return readable_vocab


def normalize_token_name(token_name: str) -> tuple[str, bool]:
    token_name = token_name.strip()
    is_original = False

    if token_name.startswith(ORIGINAL_TOKEN_PREFIX):
        is_original = True
        token_name = token_name.removeprefix(ORIGINAL_TOKEN_PREFIX).strip()

    if token_name.startswith(ADDED_TOKEN_PREFIX):
        token_name = token_name.removeprefix(ADDED_TOKEN_PREFIX).strip()

    return token_name, is_original


LANGUAGE_PATTERNS = {
    "Chinese": re.compile(r"^[\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF\U00020000-\U0002EBEF\U00030000-\U0003134F]+$"),
    "Japanese": re.compile(r"^[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF\u3005\u303B\u309D\u30FD]+$"),
    "Korean": re.compile(r"^[\uAC00-\uD7A3]+$"),
    "Thai": re.compile(r"^[\u0E00-\u0E7F]+$"),
    "Vietnamese": re.compile(r"^[A-Za-zàáảãạăắằẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ]+$"),
    "Hindi": re.compile(r"^[\u0900-\u097F]+$"),
    "Bengali": re.compile(r"^[\u0980-\u09FF]+$"),
    "Tamil": re.compile(r"^[\u0B80-\u0BFF]+$"),
    "Indian": re.compile(r"^[\u0900-\u0DFF]+$"),
    "Arabic": re.compile(r"^[\u0600-\u06FF]+$"),
    "Hebrew": re.compile(r"^[\u0590-\u05FF]+$"),
    "Russian": re.compile(r"^[\u0400-\u04FF]+$"),
    "Greek": re.compile(r"^[\u0370-\u03FF\u1F00-\u1FFF]+$"),
    "Armenian": re.compile(r"^[\u0530-\u058F]+$"),
    "Code": re.compile(r"^[=.#@*&^_/\(\)\[\]\{\}\|\?\%\$<>+\-:;,][A-Za-z0-9_]*$"),
    "LaTeX": re.compile(r"^\\[A-Za-z]+$"),
    "Numeric": re.compile(r"^[0-9]+$"),
    "Mathematical": re.compile(r"^[\u2200-\u22FF\U0001D400-\U0001D7FF]+$"),
    "Punctuation": re.compile(
        r"^[\s\u0021-\u002F\u003A-\u0040\u005B-\u0060\u007B-\u007E\u00A1-\u00BF\u2000-\u206F\u2E00-\u2E7F\u3000-\u303F\uFF00-\uFFEF\u2190-\u21FF]+$"
    ),
    "Emoji": re.compile(r"^[\U0001F300-\U0001FAFF\u2600-\u26FF]+$"),
    "English": re.compile(r"^[—–“”‘’'\"-]?[A-Za-z]+$"),
    "French": re.compile(r"^[A-Za-zàâäéèêëïîôùûüÿç]+$"),
    "German": re.compile(r"^[A-Za-zäöüßÄÖÜ]+$"),
    "Spanish": re.compile(r"^[A-Za-záéíóúñÑ]+$"),
    "Portuguese": re.compile(r"^[A-Za-zãõçéíóúâêîôû]+$"),
    "Italian": re.compile(r"^[A-Za-zàèéìíîòóùú]+$"),
}


def detect_token_language(token_name: str) -> str:
    normalized_name, _ = normalize_token_name(token_name)
    if normalized_name.startswith("INVALID UTF-8"):
        return "Invalid UTF-8"
    if not normalized_name:
        return "Whitespace"

    for language, pattern in LANGUAGE_PATTERNS.items():
        if pattern.match(normalized_name):
            return language

    if re.search(r"[A-Za-z]", normalized_name) and re.search(r"[0-9]", normalized_name):
        return "Alphanumeric"

    return "Other"


def parse_token_category(token_name: str) -> str:
    normalized_name, is_original = normalize_token_name(token_name)

    if not normalized_name:
        category = "Whitespace"
    elif normalized_name.startswith("INVALID UTF-8"):
        category = "Invalid UTF-8"
    elif normalized_name.startswith("<") and normalized_name.endswith(">"):
        inner = normalized_name[1:-1].strip()
        if inner.startswith("A"):
            category = "Category A"
        elif inner.startswith("a"):
            category = "Category a"
        elif inner.startswith("B"):
            category = "Category B"
        elif inner.startswith("b"):
            category = "Category b"
        elif inner.startswith("C"):
            category = "Category C"
        elif inner.startswith("c"):
            category = "Category c"
        elif inner.startswith("D"):
            category = "Category D"
        elif inner.startswith("d"):
            category = "Category d"
        else:
            category = "Special Token"
    else:
        category = detect_token_language(normalized_name)

    if is_original:
        return f"Original-{category}"
    return category


def build_label_distribution(labels: list[str]) -> dict[str, int]:
    return dict(Counter(labels))


def get_token_display_name(tokenizer, token_id: int, readable_vocab: dict[int, str] | None = None) -> str:
    if readable_vocab and token_id in readable_vocab:
        return readable_vocab[token_id]

    token = tokenizer.convert_ids_to_tokens(token_id)
    if token is None:
        return f"<token_{token_id}>"
    return str(token)
