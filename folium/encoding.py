"""Helpers for repairing common text mojibake."""

from __future__ import annotations

from typing import Any


_MOJIBAKE_MARKERS = (
    "\ufffd",
    "锛",
    "銆",
    "涓",
    "鐨",
    "鏄",
    "杩",
    "鏈",
    "绯",
    "荤",
    "粺",
    "闈",
    "㈠",
    "悜",
    "鍐",
    "儴",
    "鍚",
    "︽",
    "敮",
    "鎸",
    "侀",
    "渶",
    "姹",
    "傜",
    "殑",
    "绫",
    "诲",
    "瀷",
    "浠",
    "鍙",
    "傚",
    "兘",
    "杈",
    "撳",
    "嚭",
    "绉",
    "爺",
    "鏅",
    "鸿",
    "柊",
    "瀵",
    "硅",
    "瘽",
)

_LOSSY_MOJIBAKE_REPAIRS = {
    "鏈€": "最",
    "鏈�": "本",
    "鏄�": "是",
    "闇�": "需",
    "涓�": "一",
    "朄1�7": "本",
    "朢�": "最",
    "昄1�7": "是",
    "霄1�7": "需",
    "丄1�7": "一",
    "仄1�7": "代",
    "��系统": "本系统",
    "��否": "是否",
    "类型��": "类型。",
    "銆�": "。",
    "〄1�7": "。",
    "锛�": "，",
    "鏃�": "日",
    "榛樿�": "默认",
    "鑷�": "自",
    "涔�": "久",
}


def repair_mojibake_text(text: str) -> str:
    """Repair likely UTF-8 text that was decoded as GBK/GB18030.

    The repair is intentionally conservative: it only returns a candidate when
    the original text contains known mojibake markers and the candidate scores
    clearly better. Replacement characters mean some bytes were already lost,
    so those segments may only become partially readable.
    """
    if not text or not _looks_like_mojibake(text):
        return text

    text = _repair_lossy_mojibake(text)
    original_score = _mojibake_score(text)
    best = text
    best_score = original_score
    lossy = _repair_lossy_mojibake(text)
    lossy_score = _mojibake_score(lossy)
    if lossy_score < best_score:
        best = lossy
        best_score = lossy_score

    for encoding in ("gb18030", "gbk", "cp936"):
        try:
            candidate = text.encode(encoding, errors="replace").decode("utf-8", errors="replace")
        except (LookupError, UnicodeError):
            continue
        candidate = _repair_lossy_mojibake(candidate)
        candidate_score = _mojibake_score(candidate)
        if candidate_score < best_score:
            best = candidate
            best_score = candidate_score

    if best is not text and best_score <= max(0, original_score - 2):
        return best
    return text


def repair_mojibake_payload(value: Any) -> Any:
    """Recursively repair text values inside JSON-like payloads."""
    if isinstance(value, str):
        return repair_mojibake_text(value)
    if isinstance(value, list):
        return [repair_mojibake_payload(item) for item in value]
    if isinstance(value, dict):
        return {key: repair_mojibake_payload(item) for key, item in value.items()}
    return value


def _looks_like_mojibake(text: str) -> bool:
    return any(marker in text for marker in _MOJIBAKE_MARKERS)


def _repair_lossy_mojibake(text: str) -> str:
    for bad, good in _LOSSY_MOJIBAKE_REPAIRS.items():
        text = text.replace(bad, good)
    text = text.replace("鏄�鍚�", "是否")
    text = text.replace("闇�姹�", "需求")
    text = text.replace("浠ｇ増本", "代版本")
    text = text.replace("杩�代", "迭代")
    return text


def _mojibake_score(text: str) -> int:
    score = text.count("\ufffd") * 5
    for marker in _MOJIBAKE_MARKERS:
        if marker != "\ufffd":
            score += text.count(marker) * 2
    return score
