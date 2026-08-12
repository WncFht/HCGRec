import datetime
import html
import json
import os
import re


def ensure_dir(dir_path):
    os.makedirs(dir_path, exist_ok=True)


def set_color(log, color, highlight=True):
    color_set = [
        "black",
        "red",
        "green",
        "yellow",
        "blue",
        "pink",
        "cyan",
        "white",
    ]
    try:
        index = color_set.index(color)
    except ValueError:
        index = len(color_set) - 1
    prev_log = "\033["
    if highlight:
        prev_log += "1;3"
    else:
        prev_log += "0;3"
    prev_log += str(index) + "m"
    return prev_log + log + "\033[0m"


def get_local_time():
    r"""
    Get current time

    Returns:
        str: current time

    """
    cur = datetime.datetime.now()
    cur = cur.strftime("%b-%d-%Y_%H-%M-%S")

    return cur


def delete_file(filename):
    if os.path.exists(filename):
        os.remove(filename)


def load_json(path: str):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


_HTML_TAG_RE = re.compile(r"</?\w+[^>]*>")
_QUOTE_NL_RE = re.compile(r'["\n\r]*')


def clean_text(raw_text, max_len: int = 2000) -> str:
    """
    Lightweight text cleaning for Amazon metadata fields.
    - Unescape HTML entities
    - Strip simple HTML tags
    - Remove quotes/newlines
    - Truncate overly-long strings (return empty)
    """
    if raw_text is None:
        return ""

    if isinstance(raw_text, list):
        parts = []
        for part in raw_text:
            if part is None:
                continue
            s = str(part).strip()
            s = html.unescape(s)
            s = _HTML_TAG_RE.sub("", s)
            s = _QUOTE_NL_RE.sub("", s)
            s = s.strip()
            if s:
                parts.append(s)
        text = " ".join(parts)
    elif isinstance(raw_text, dict):
        text = str(raw_text).strip()
    else:
        text = str(raw_text).strip()

    text = html.unescape(text)
    text = _HTML_TAG_RE.sub("", text)
    text = _QUOTE_NL_RE.sub("", text).strip()

    if len(text) >= max_len:
        return ""

    return text
