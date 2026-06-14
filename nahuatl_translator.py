#!/usr/bin/env python3
"""Drag-and-drop Nahuatl → English/Spanish translation tool using Claude."""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
import tkinter as tk
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable
from tkinter import filedialog, messagebox, scrolledtext, ttk

try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
except ImportError as exc:
    raise SystemExit("Install tkinterdnd2: pip install tkinterdnd2") from exc

import anthropic

SCRIPT_DIR = Path(__file__).resolve().parent
PROMPT_FILE = SCRIPT_DIR / "wikowi_codex_prompt_FINAL.md"
ALIGNMENT_MODEL = "claude-haiku-4-5"
TRANSLATION_MODEL = "claude-opus-4-8"
MAX_TOKENS = 4000
ALIGNMENT_MAX_TOKENS = 2000
ALIGNMENT_WINDOW_WORDS = 2500
UNCERTAIN_CURSOR_SKIP_WORDS = 150
HAIKU_INPUT_COST_PER_M = 1.0
HAIKU_OUTPUT_COST_PER_M = 5.0
INPUT_COST_PER_M = 5.0
OUTPUT_COST_PER_M = 25.0
BATCH_DISCOUNT = 0.5
PREVIEW_COUNT = 3
PREVIEW_HEAD_WORDS = 10
TEST_MODE_WORD_LIMIT = 300
LARGE_PASSAGE_WORDS = 3000
DEFAULT_WORDS_PER_PASSAGE = 400
API_MAX_RETRIES = 4
BATCH_POLL_SEC = 15
BATCH_MAX_REQUESTS = 5000
TEXT_SUFFIXES = {".txt", ".text", ".md"}
ENCODINGS = ("utf-8-sig", "utf-8", "cp1252", "latin-1")

SPLIT_CHAPTER = "chapter"
SPLIT_PARAGRAPH = "paragraph"
SPLIT_WORDS = "words"

DEFAULT_CHAPTER_REGEX = (
    r"^(?:"
    r"(?:Chapter|CHAPTER|Cap[ií]tulo|CAP[IÍ]TULO|Book|BOOK|Libro|LIBRO)\s+[\w\dIVXLCivxlc\-]+.*"
    r"|#{1,3}\s+\S.+"
    r"|\*{2,}.+\*{2,}"
    r"|[IVXLC]+\.\s+\S"
    r"|\d+\.\s+[A-Z].+"
    r")\s*$"
)

# Language-agnostic sentence endings (Latin, CJK full-width, ellipsis)
SENTENCE_END_RE = re.compile(
    r"(?<=[.!?…\u3002\uff01\uff1f])"
    r'["\'\u00bb\u201d\u2019\)\]\u300d\ufeff]*'
    r"(?:\s+|$)"
)

OPEN_QUOTES = frozenset('"\'«「“‘„')
CLOSE_QUOTES = frozenset('"\'»」”’“')

TAG_RE = {
    "english": re.compile(r"<english>(.*?)</english>", re.DOTALL | re.IGNORECASE),
    "spanish": re.compile(r"<spanish>(.*?)</spanish>", re.DOTALL | re.IGNORECASE),
    "flags": re.compile(r"<flags>(.*?)</flags>", re.DOTALL | re.IGNORECASE),
}


@dataclass
class PassageResult:
    index: int
    english: str = ""
    spanish: str = ""
    flags: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    error: str | None = None
    truncated: bool = False
    skipped: bool = False
    uncertain_match: bool = False
    pair_fingerprint: str = ""


@dataclass
class AlignmentEntry:
    index: int
    nahuatl: str
    english: str
    uncertain: bool = False


@dataclass
class RunState:
    pairs: list[tuple[str, str]] = field(default_factory=list)
    pair_uncertain: dict[int, bool] = field(default_factory=dict)
    nahuatl_paths: list[Path] = field(default_factory=list)
    english_paths: list[Path] = field(default_factory=list)
    nahuatl_text: str = ""
    english_text: str = ""
    nahuatl_passages: list[str] = field(default_factory=list)
    nahuatl_passage_count: int = 0
    english_passage_count: int = 0
    aligned: bool = False
    ai_alignment: bool = False
    alignment_uncertain: list[int] = field(default_factory=list)
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    alignment_input_tokens: int = 0
    alignment_output_tokens: int = 0
    alignment_cost: float = 0.0
    total_cost: float = 0.0
    results: list[PassageResult] = field(default_factory=list)
    failed: list[int] = field(default_factory=list)
    truncated: list[int] = field(default_factory=list)
    batch_ids: list[str] = field(default_factory=list)
    test_mode: bool = False
    retry_indices: list[int] = field(default_factory=list)


def _read_env_key() -> str:
    for name in ("ANTHROPIC_API_KEY", "CLAUDE_API_KEY"):
        val = os.environ.get(name, "").strip()
        if val:
            return val
    dotenv = SCRIPT_DIR / ".env"
    if dotenv.is_file():
        for line in dotenv.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            for name in ("ANTHROPIC_API_KEY", "CLAUDE_API_KEY"):
                if line.startswith(f"{name}="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
    return ""


def _read_transcriber_key() -> str:
    base = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "PDF Transcribe"
    path = base / "settings.json"
    if not path.is_file():
        return ""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return ""
    return (
        data.get("api_key") or data.get("anthropic_api_key") or ""
    ).strip()


def resolve_api_key() -> str:
    key = _read_env_key()
    if key:
        return key
    key = _read_transcriber_key()
    if key:
        return key
    raise ValueError(
        "No API key found. Set ANTHROPIC_API_KEY or save a key in PDF Transcribe "
        "(same as the transcriber app)."
    )


def resolve_translation_model() -> str:
    return (os.environ.get("CLAUDE_MODEL") or TRANSLATION_MODEL).strip() or TRANSLATION_MODEL


def resolve_model() -> str:
    """Alias for translation model (Opus)."""
    return resolve_translation_model()


def load_system_prompt() -> str:
    if not PROMPT_FILE.is_file():
        raise FileNotFoundError(f"System prompt not found: {PROMPT_FILE}")
    return PROMPT_FILE.read_text(encoding="utf-8")


def source_content_hash(nahuatl_text: str, english_text: str) -> str:
    digest = hashlib.sha256()
    digest.update(nahuatl_text.encode("utf-8"))
    digest.update(b"\0")
    digest.update(english_text.encode("utf-8"))
    return digest.hexdigest()


def pair_fingerprint(nahuatl: str, english: str) -> str:
    digest = hashlib.sha256()
    digest.update(nahuatl.encode("utf-8"))
    digest.update(b"\0")
    digest.update(english.encode("utf-8"))
    return digest.hexdigest()


def save_system_prompt(text: str) -> None:
    PROMPT_FILE.write_text(text, encoding="utf-8")


def parse_drop_paths(tk_root: tk.Misc, data: str) -> list[Path]:
    raw = (data or "").strip()
    if not raw:
        return []
    try:
        items = tk_root.tk.splitlist(raw)
    except tk.TclError:
        items = [raw]
    paths: list[Path] = []
    seen: set[str] = set()
    for item in items:
        p = Path(str(item).strip().strip('"').strip("'"))
        key = str(p.resolve()) if p.exists() else str(p)
        if key in seen:
            continue
        seen.add(key)
        paths.append(p)
    return paths


def read_text_file(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")
    last_err: Exception | None = None
    for enc in ENCODINGS:
        try:
            return path.read_text(encoding=enc)
        except (UnicodeDecodeError, OSError) as exc:
            last_err = exc
    raise OSError(f"Could not read {path.name}: {last_err}")


def read_text_files(paths: list[Path]) -> str:
    ordered = sorted(paths, key=lambda p: p.name.lower())
    parts: list[str] = []
    for path in ordered:
        text = read_text_file(path).strip()
        if text:
            parts.append(text)
    return "\n\n".join(parts)


def output_directory(paths: list[Path]) -> Path:
    if not paths:
        return SCRIPT_DIR
    parents = {p.resolve().parent for p in paths if p.exists()}
    if len(parents) == 1:
        return next(iter(parents))
    return sorted(paths, key=lambda p: p.name.lower())[0].resolve().parent


def compile_chapter_regex(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern, re.MULTILINE)


def split_by_chapters(text: str, chapter_re: re.Pattern[str]) -> list[str]:
    text = text.strip()
    if not text:
        return []
    headings = list(chapter_re.finditer(text))
    if not headings:
        return []
    starts = [m.start() for m in headings]
    if starts[0] != 0:
        starts.insert(0, 0)
    chunks: list[str] = []
    for i, start in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else len(text)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
    return chunks


def split_by_paragraphs(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    return [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]


def quote_depth_at(text: str, pos: int) -> int:
    depth = 0
    i = 0
    while i < min(pos, len(text)):
        ch = text[i]
        if ch in OPEN_QUOTES:
            depth += 1
        elif ch in CLOSE_QUOTES:
            depth = max(0, depth - 1)
        i += 1
    return depth


def extend_to_quote_close(text: str, pos: int) -> int:
    """If pos falls inside quotes, extend through the closing quote."""
    if quote_depth_at(text, pos) <= 0:
        return min(pos, len(text))
    depth = quote_depth_at(text, pos)
    i = pos
    while i < len(text) and depth > 0:
        ch = text[i]
        if ch in OPEN_QUOTES:
            depth += 1
        elif ch in CLOSE_QUOTES:
            depth -= 1
        i += 1
    return i


def find_natural_split_before(text: str, max_words: int) -> tuple[str, str]:
    """Split text into head/tail at a natural boundary; head has at most max_words."""
    text = text.strip()
    if count_words(text) <= max_words:
        return text, ""
    word_matches = list(re.finditer(r"\S+", text))
    if len(word_matches) <= max_words:
        return text, ""
    target_end = word_matches[max_words - 1].end()
    search_region = text[:target_end]
    best_split: int | None = None
    for match in SENTENCE_END_RE.finditer(search_region):
        candidate = extend_to_quote_close(text, match.end())
        if candidate > len(text) * 0.15:
            best_split = candidate
    if best_split is None:
        best_split = extend_to_quote_close(text, target_end)
    head = text[:best_split].strip()
    tail = text[best_split:].strip()
    if not head:
        best_split = extend_to_quote_close(text, target_end)
        head = text[:best_split].strip()
        tail = text[best_split:].strip()
    return head, tail


def split_by_natural_word_chunks(text: str, words_per_passage: int) -> list[str]:
    """Split on natural boundaries — never mid-sentence or mid-quote when avoidable."""
    chunks: list[str] = []
    remaining = text.strip()
    n = max(1, words_per_passage)
    while remaining:
        if count_words(remaining) <= n:
            chunks.append(remaining)
            break
        head, tail = find_natural_split_before(remaining, n)
        if not head:
            head, tail = find_natural_split_before(remaining, max(1, n // 2))
        if head:
            chunks.append(head)
        remaining = tail
        if not remaining:
            break
    return [c for c in chunks if c.strip()]


def subdivide_large_chunks(chunks: list[str], max_words: int) -> list[str]:
    out: list[str] = []
    for chunk in chunks:
        if count_words(chunk) <= max_words:
            out.append(chunk)
        else:
            out.extend(split_by_natural_word_chunks(chunk, max_words))
    return out


def split_by_word_count(text: str, words_per_passage: int) -> list[str]:
    return split_by_natural_word_chunks(text, words_per_passage)


def split_passages(
    text: str,
    method: str,
    *,
    chapter_pattern: str = DEFAULT_CHAPTER_REGEX,
    words_per_passage: int = DEFAULT_WORDS_PER_PASSAGE,
) -> list[str]:
    if method == SPLIT_CHAPTER:
        try:
            chapter_re = compile_chapter_regex(chapter_pattern)
        except re.error as exc:
            raise ValueError(f"Invalid chapter regex: {exc}") from exc
        chunks = split_by_chapters(text, chapter_re)
        if not chunks:
            chunks = split_by_paragraphs(text)
        return subdivide_large_chunks(chunks, words_per_passage)
    if method == SPLIT_WORDS:
        return split_by_natural_word_chunks(text, words_per_passage)
    chunks = split_by_paragraphs(text)
    return subdivide_large_chunks(chunks, words_per_passage)


def count_words(text: str) -> int:
    return len(re.findall(r"\S+", text))


def truncate_words(text: str, max_words: int) -> tuple[str, int, bool]:
    words = re.findall(r"\S+", text.strip())
    if len(words) <= max_words:
        return text.strip(), len(words), False
    return " ".join(words[:max_words]), max_words, True


def first_words(text: str, n: int = PREVIEW_HEAD_WORDS) -> str:
    words = re.findall(r"\S+", text.strip())
    if not words:
        return "(empty)"
    preview = " ".join(words[:n])
    if len(words) > n:
        preview += "…"
    return preview


def passage_marker(index: int) -> str:
    return f"=== Passage {index:03d} ==="


def output_filenames(*, test_mode: bool) -> dict[str, str]:
    suffix = "_test" if test_mode else ""
    return {
        "english": f"english_all{suffix}.txt",
        "spanish": f"spanish_all{suffix}.txt",
        "flags": f"flags_all{suffix}.txt",
        "summary_json": f"run_summary{suffix}.json",
        "summary_txt": f"run_summary{suffix}.txt",
        "failed_log": f"failed_passages{suffix}.log",
        "truncated_log": f"truncated{suffix}.log",
        "batch_state": f"batch_state{suffix}.json",
        "passages_dir": f"passages{suffix}",
        "alignment_map": f"alignment_map{suffix}.json",
    }


def passages_dir(out_dir: Path, test_mode: bool) -> Path:
    d = out_dir / output_filenames(test_mode=test_mode)["passages_dir"]
    d.mkdir(parents=True, exist_ok=True)
    return d


def passage_record_path(out_dir: Path, index: int, test_mode: bool) -> Path:
    return passages_dir(out_dir, test_mode) / f"passage_{index:05d}.json"


def load_passage_record(path: Path) -> PassageResult | None:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return PassageResult(
            index=int(data["index"]),
            english=data.get("english", ""),
            spanish=data.get("spanish", ""),
            flags=data.get("flags"),
            input_tokens=int(data.get("input_tokens", 0)),
            output_tokens=int(data.get("output_tokens", 0)),
            error=data.get("error"),
            truncated=bool(data.get("truncated", False)),
            skipped=bool(data.get("skipped", False)),
            uncertain_match=bool(data.get("uncertain_match", False)),
            pair_fingerprint=data.get("pair_fingerprint", ""),
        )
    except (json.JSONDecodeError, OSError, KeyError, TypeError, ValueError):
        return None


def passage_is_resumable(result: PassageResult | None, expected_fingerprint: str) -> bool:
    if result is None or result.error or result.truncated:
        return False
    if not result.english.strip() or not result.spanish.strip():
        return False
    if not result.pair_fingerprint or result.pair_fingerprint != expected_fingerprint:
        return False
    return True


def save_passage_record(out_dir: Path, result: PassageResult, test_mode: bool) -> None:
    path = passage_record_path(out_dir, result.index, test_mode)
    path.write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")


def rebuild_aggregate_file(
    out_dir: Path,
    *,
    test_mode: bool,
    field_name: str,
    filename_key: str,
) -> None:
    names = output_filenames(test_mode=test_mode)
    pdir = out_dir / names["passages_dir"]
    if not pdir.is_dir():
        return
    blocks: list[str] = []
    for path in sorted(pdir.glob("passage_*.json")):
        rec = load_passage_record(path)
        if rec is None:
            continue
        marker = passage_marker(rec.index)
        if rec.error:
            body = f"[FAILED: {rec.error}]"
        elif rec.truncated:
            body = "[TRUNCATED — output hit max_tokens; passage NOT complete — re-run or split]"
        else:
            body = getattr(rec, field_name, "")
        blocks.append(f"{marker}\n{body}")
    if blocks:
        (out_dir / names[filename_key]).write_text("\n\n".join(blocks) + "\n", encoding="utf-8")


def rebuild_flags_file(out_dir: Path, test_mode: bool) -> None:
    names = output_filenames(test_mode=test_mode)
    pdir = out_dir / names["passages_dir"]
    if not pdir.is_dir():
        return
    blocks: list[str] = []
    for path in sorted(pdir.glob("passage_*.json")):
        rec = load_passage_record(path)
        if rec and rec.flags and not rec.error:
            blocks.append(f"{passage_marker(rec.index)}\n{rec.flags}")
    flags_path = out_dir / names["flags"]
    if blocks:
        flags_path.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")
    elif flags_path.is_file():
        flags_path.unlink()


def rebuild_failed_log(out_dir: Path, test_mode: bool) -> None:
    names = output_filenames(test_mode=test_mode)
    pdir = out_dir / names["passages_dir"]
    lines: list[str] = []
    if pdir.is_dir():
        for path in sorted(pdir.glob("passage_*.json")):
            rec = load_passage_record(path)
            if rec and rec.error:
                lines.append(f"Passage {rec.index}: {rec.error}")
    log_path = out_dir / names["failed_log"]
    if lines:
        log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    elif log_path.is_file():
        log_path.unlink()


def rebuild_truncated_log(out_dir: Path, test_mode: bool) -> None:
    names = output_filenames(test_mode=test_mode)
    pdir = out_dir / names["passages_dir"]
    lines: list[str] = []
    if pdir.is_dir():
        for path in sorted(pdir.glob("passage_*.json")):
            rec = load_passage_record(path)
            if rec and rec.truncated:
                lines.append(f"Passage {rec.index}: stop_reason=max_tokens (output cut off)")
    log_path = out_dir / names["truncated_log"]
    if lines:
        log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    elif log_path.is_file():
        log_path.unlink()


def persist_passage_output(out_dir: Path, result: PassageResult, test_mode: bool) -> None:
    save_passage_record(out_dir, result, test_mode)
    if result.truncated:
        rebuild_truncated_log(out_dir, test_mode)
        return
    rebuild_aggregate_file(out_dir, test_mode=test_mode, field_name="english", filename_key="english")
    rebuild_aggregate_file(out_dir, test_mode=test_mode, field_name="spanish", filename_key="spanish")
    rebuild_flags_file(out_dir, test_mode)


def alignment_map_path(out_dir: Path, test_mode: bool) -> Path:
    return out_dir / output_filenames(test_mode=test_mode)["alignment_map"]


def split_settings_fingerprint(
    method: str,
    chapter_pattern: str,
    words_per_passage: int,
    test_mode: bool,
    ai_alignment: bool,
) -> dict:
    return {
        "method": method,
        "chapter_pattern": chapter_pattern,
        "words_per_passage": words_per_passage,
        "test_mode": test_mode,
        "ai_alignment": ai_alignment,
    }


def load_alignment_map(
    path: Path,
    *,
    content_hash: str,
    settings: dict,
) -> list[AlignmentEntry] | None:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if data.get("settings") != settings:
        return None
    if data.get("content_hash") != content_hash:
        return None
    entries: list[AlignmentEntry] = []
    for item in data.get("pairs", []):
        entries.append(
            AlignmentEntry(
                index=int(item["index"]),
                nahuatl=item.get("nahuatl", ""),
                english=item.get("english", ""),
                uncertain=bool(item.get("uncertain", False)),
            )
        )
    return entries or None


def save_alignment_map(
    path: Path,
    entries: list[AlignmentEntry],
    *,
    content_hash: str,
    settings: dict,
) -> None:
    payload = {
        "settings": settings,
        "content_hash": content_hash,
        "pairs": [asdict(e) for e in entries],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def english_sliding_window(english_text: str, cursor: int, window_words: int = ALIGNMENT_WINDOW_WORDS) -> str:
    remaining = english_text[cursor:].strip()
    if not remaining:
        return ""
    window, _, _ = truncate_words(remaining, window_words)
    return window


def build_alignment_prompt(english_window: str, nahuatl_passage: str, index: int) -> str:
    return (
        "You align Nahuatl source passages to English reference text.\n\n"
        f"ENGLISH REFERENCE (sequential window — copy verbatim from here only):\n"
        f"{english_window}\n\n"
        f"NAHUATL PASSAGE #{index}:\n{nahuatl_passage}\n\n"
        "Return ONLY the exact substring from the English reference that matches this "
        "Nahuatl passage in meaning. Do not paraphrase.\n\n"
        "If there is no clear match, return exactly: NO_MATCH"
    )


def normalize_alignment_response(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```\w*\n?", "", cleaned)
        cleaned = re.sub(r"\n?```$", "", cleaned).strip()
    return cleaned


def normalize_for_match(text: str) -> str:
    quote_map = str.maketrans(
        {
            "\u201c": '"',
            "\u201d": '"',
            "\u2018": "'",
            "\u2019": "'",
            "\u00ab": '"',
            "\u00bb": '"',
            "\u2014": "-",
            "\u2013": "-",
        }
    )
    text = text.translate(quote_map)
    text = text.replace("…", "...")
    return re.sub(r"\s+", " ", text).strip()


def _build_norm_index_map(text: str) -> tuple[str, list[int]]:
    norm_chars: list[str] = []
    orig_at: list[int] = []
    i = 0
    pending_space = False
    while i < len(text):
        chunk = text[i : i + 1]
        if chunk.isspace():
            pending_space = bool(norm_chars)
            i += 1
            continue
        if pending_space and norm_chars:
            norm_chars.append(" ")
            orig_at.append(i)
            pending_space = False
        translated = chunk.translate(
            str.maketrans(
                {
                    "\u201c": '"',
                    "\u201d": '"',
                    "\u2018": "'",
                    "\u2019": "'",
                    "\u00ab": '"',
                    "\u00bb": '"',
                    "\u2014": "-",
                    "\u2013": "-",
                }
            )
        )
        if translated == "…":
            for ch in "...":
                norm_chars.append(ch)
                orig_at.append(i)
        else:
            norm_chars.append(translated)
            orig_at.append(i)
        i += 1
    return "".join(norm_chars), orig_at


def find_english_match_start(match: str, english_text: str, cursor: int) -> int | None:
    if not match or match.upper() == "NO_MATCH":
        return None
    search_from = max(0, cursor - 200)
    region = english_text[search_from:]
    pos = region.find(match)
    if pos >= 0:
        return search_from + pos
    pos = english_text.find(match)
    if pos >= 0:
        return pos
    norm_match = normalize_for_match(match)
    if len(norm_match) < 8:
        return None
    for haystack, base in ((region, search_from), (english_text, 0)):
        norm_hay, orig_map = _build_norm_index_map(haystack)
        npos = norm_hay.find(norm_match)
        if npos >= 0 and orig_map:
            end_norm = npos + len(norm_match) - 1
            if end_norm < len(orig_map):
                return base + orig_map[npos]
    return None


def verify_english_substring(match: str, english_text: str, cursor: int) -> bool:
    return find_english_match_start(match, english_text, cursor) is not None


def advance_cursor_by_words(english_text: str, cursor: int, skip_words: int) -> int:
    remaining = english_text[cursor:]
    word_matches = list(re.finditer(r"\S+", remaining))
    if not word_matches:
        return len(english_text)
    idx = min(max(1, skip_words), len(word_matches)) - 1
    return cursor + word_matches[idx].end()


def call_haiku_align(
    client: anthropic.Anthropic,
    english_window: str,
    nahuatl_passage: str,
    index: int,
) -> tuple[str, bool, int, int]:
    last_err: Exception | None = None
    for attempt in range(API_MAX_RETRIES):
        try:
            message = client.messages.create(
                model=ALIGNMENT_MODEL,
                max_tokens=ALIGNMENT_MAX_TOKENS,
                messages=[
                    {
                        "role": "user",
                        "content": build_alignment_prompt(english_window, nahuatl_passage, index),
                    },
                ],
            )
            raw = normalize_alignment_response(message_text(message))
            in_tok, out_tok = message_usage(message)
            if raw.upper() == "NO_MATCH" or len(raw) < 8:
                return raw, True, in_tok, out_tok
            return raw, False, in_tok, out_tok
        except Exception as exc:
            last_err = exc
            if attempt < API_MAX_RETRIES - 1 and is_retryable_api_error(exc):
                time.sleep(min(2**attempt, 30))
                continue
            raise
    raise last_err or RuntimeError("Alignment API call failed")


def align_nahuatl_passages(
    client: anthropic.Anthropic,
    nahuatl_passages: list[str],
    english_text: str,
    *,
    out_dir: Path,
    test_mode: bool,
    content_hash: str,
    settings: dict,
    on_progress: Callable[[int, int, str], None] | None = None,
) -> tuple[list[AlignmentEntry], int, int]:
    cache_path = alignment_map_path(out_dir, test_mode)
    cached = load_alignment_map(
        cache_path,
        content_hash=content_hash,
        settings=settings,
    )
    if cached and len(cached) == len(nahuatl_passages):
        return cached, 0, 0

    entries: list[AlignmentEntry] = []
    align_in = 0
    align_out = 0
    cursor = 0
    total = len(nahuatl_passages)
    for i, nahuatl in enumerate(nahuatl_passages, start=1):
        if on_progress:
            on_progress(i, total, f"AI aligning passage {i}/{total} (Haiku)…")
        window = english_sliding_window(english_text, cursor)
        english_match, uncertain, in_tok, out_tok = call_haiku_align(client, window, nahuatl, i)
        align_in += in_tok
        align_out += out_tok
        match_start = find_english_match_start(english_match, english_text, cursor)
        if not uncertain and match_start is not None:
            cursor = match_start + len(english_match)
        else:
            uncertain = True
            skip = max(50, min(count_words(nahuatl), UNCERTAIN_CURSOR_SKIP_WORDS))
            cursor = advance_cursor_by_words(english_text, cursor, skip)
            english_match = "(uncertain — no verified English match)"
        entry = AlignmentEntry(index=i, nahuatl=nahuatl, english=english_match, uncertain=uncertain)
        entries.append(entry)
        save_alignment_map(
            cache_path,
            entries,
            content_hash=content_hash,
            settings=settings,
        )
    return entries, align_in, align_out


def cached_system_prompt(system_prompt: str) -> list[dict]:
    return [
        {
            "type": "text",
            "text": system_prompt,
            "cache_control": {"type": "ephemeral"},
        }
    ]


def build_user_message(nahuatl: str, english: str) -> str:
    return (
        "CONTEXT: Florentine Codex, Nahuatl.\n\n"
        f"ORIGINAL:\n{nahuatl}\n\n"
        f"REFERENCE (meaning only, never style):\n{english}"
    )


def message_text(message: anthropic.types.Message | dict) -> str:
    parts: list[str] = []
    content = message.content if hasattr(message, "content") else message.get("content", [])
    for block in content:
        if isinstance(block, dict):
            if block.get("type") == "text" and block.get("text"):
                parts.append(block["text"])
        elif getattr(block, "type", None) == "text" and getattr(block, "text", None):
            parts.append(block.text)
    return "\n".join(parts)


def message_usage(message: anthropic.types.Message | dict) -> tuple[int, int]:
    ext = message_usage_extended(message)
    return ext["input_tokens"], ext["output_tokens"]


def message_usage_extended(message: anthropic.types.Message | dict) -> dict[str, int]:
    usage = message.usage if hasattr(message, "usage") else message.get("usage", {})
    if isinstance(usage, dict):
        return {
            "input_tokens": int(usage.get("input_tokens", 0)),
            "output_tokens": int(usage.get("output_tokens", 0)),
            "cache_read_input_tokens": int(usage.get("cache_read_input_tokens", 0)),
            "cache_creation_input_tokens": int(usage.get("cache_creation_input_tokens", 0)),
        }
    return {
        "input_tokens": int(getattr(usage, "input_tokens", 0)),
        "output_tokens": int(getattr(usage, "output_tokens", 0)),
        "cache_read_input_tokens": int(getattr(usage, "cache_read_input_tokens", 0)),
        "cache_creation_input_tokens": int(getattr(usage, "cache_creation_input_tokens", 0)),
    }


def message_stop_reason(message: anthropic.types.Message | dict) -> str | None:
    if hasattr(message, "stop_reason"):
        return getattr(message, "stop_reason", None)
    if isinstance(message, dict):
        return message.get("stop_reason")
    return None


def parse_response(text: str) -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for tag, pattern in TAG_RE.items():
        match = pattern.search(text)
        out[tag] = match.group(1).strip() if match else None
    return out


def token_cost(input_tokens: int, output_tokens: int, *, batch: bool = False) -> float:
    factor = BATCH_DISCOUNT if batch else 1.0
    return (input_tokens / 1_000_000 * INPUT_COST_PER_M * factor) + (
        output_tokens / 1_000_000 * OUTPUT_COST_PER_M * factor
    )


def haiku_token_cost(input_tokens: int, output_tokens: int) -> float:
    return (input_tokens / 1_000_000 * HAIKU_INPUT_COST_PER_M) + (
        output_tokens / 1_000_000 * HAIKU_OUTPUT_COST_PER_M
    )


def batch_custom_id(index: int) -> str:
    return f"passage-{index:05d}"


def parse_passage_index(custom_id: str) -> int | None:
    match = re.fullmatch(r"passage-(\d+)", custom_id or "")
    return int(match.group(1)) if match else None


def build_message_params(system_prompt: str, nahuatl: str, english: str) -> dict:
    return {
        "model": resolve_translation_model(),
        "max_tokens": MAX_TOKENS,
        "system": cached_system_prompt(system_prompt),
        "messages": [{"role": "user", "content": build_user_message(nahuatl, english)}],
    }


def apply_parsed_response(result: PassageResult, response_text: str) -> None:
    if not response_text.strip():
        raise ValueError("Empty API response")
    parsed = parse_response(response_text)
    if not parsed.get("english") or not parsed.get("spanish"):
        raise ValueError("Response missing <english> or <spanish> tags")
    result.english = parsed["english"] or ""
    result.spanish = parsed["spanish"] or ""
    result.flags = parsed.get("flags")


def apply_message_to_result(result: PassageResult, message: anthropic.types.Message | dict) -> None:
    if message_stop_reason(message) == "max_tokens":
        result.truncated = True
    try:
        apply_parsed_response(result, message_text(message))
    except ValueError as exc:
        if result.truncated:
            result.error = f"Truncated response: {exc}"
        else:
            raise
    usage = message_usage_extended(message)
    result.input_tokens = usage["input_tokens"]
    result.output_tokens = usage["output_tokens"]


def create_translation_batch(
    client: anthropic.Anthropic,
    system_prompt: str,
    indexed_pairs: list[tuple[int, str, str]],
) -> str:
    requests = [
        {
            "custom_id": batch_custom_id(index),
            "params": build_message_params(system_prompt, nahuatl, english),
        }
        for index, nahuatl, english in indexed_pairs
    ]
    batch = client.messages.batches.create(requests=requests)
    batch_id = getattr(batch, "id", None) or (batch.get("id") if isinstance(batch, dict) else None)
    if not batch_id:
        raise RuntimeError("Batch API did not return a batch id.")
    return batch_id


def wait_for_batch(
    client: anthropic.Anthropic,
    batch_id: str,
    *,
    on_status: Callable[[object], None] | None = None,
) -> object:
    while True:
        batch = client.messages.batches.retrieve(batch_id)
        if on_status:
            on_status(batch)
        status = getattr(batch, "processing_status", None) or (
            batch.get("processing_status") if isinstance(batch, dict) else None
        )
        if status == "ended":
            return batch
        if status in ("canceling", "canceled"):
            raise RuntimeError(f"Batch {batch_id} was canceled.")
        time.sleep(BATCH_POLL_SEC)


def batch_counts_done(batch: object) -> tuple[int, int]:
    counts = getattr(batch, "request_counts", None)
    if counts is None and isinstance(batch, dict):
        counts = batch.get("request_counts")
    if counts is None:
        return 0, 0
    if isinstance(counts, dict):
        done = sum(int(counts.get(k, 0)) for k in ("succeeded", "errored", "canceled", "expired"))
    else:
        done = sum(
            int(getattr(counts, k, 0))
            for k in ("succeeded", "errored", "canceled", "expired")
        )
    return done, done + int(
        counts.get("processing", 0) if isinstance(counts, dict) else getattr(counts, "processing", 0)
    )


def collect_batch_results(client: anthropic.Anthropic, batch_id: str) -> dict[int, PassageResult]:
    by_index: dict[int, PassageResult] = {}
    for entry in client.messages.batches.results(batch_id):
        custom_id = getattr(entry, "custom_id", None) or (
            entry.get("custom_id") if isinstance(entry, dict) else ""
        )
        index = parse_passage_index(custom_id or "")
        if index is None:
            continue
        result = PassageResult(index=index)
        raw_result = getattr(entry, "result", None) or (
            entry.get("result") if isinstance(entry, dict) else None
        )
        if raw_result is None:
            result.error = "Missing batch result payload"
            by_index[index] = result
            continue
        rtype = getattr(raw_result, "type", None) or (
            raw_result.get("type") if isinstance(raw_result, dict) else None
        )
        if rtype == "succeeded":
            message = getattr(raw_result, "message", None) or raw_result.get("message")
            try:
                if message_stop_reason(message) == "max_tokens":
                    result.truncated = True
                apply_parsed_response(result, message_text(message))
                usage = message_usage_extended(message)
                result.input_tokens = usage["input_tokens"]
                result.output_tokens = usage["output_tokens"]
                result._usage = usage  # type: ignore[attr-defined]
            except Exception as exc:
                result.error = str(exc)
                if message_stop_reason(message) == "max_tokens":
                    result.truncated = True
        elif rtype == "errored":
            err = getattr(raw_result, "error", None) or raw_result.get("error", {})
            if isinstance(err, dict):
                result.error = err.get("message") or json.dumps(err)[:500]
            else:
                result.error = str(err)[:500]
        else:
            result.error = f"Unexpected batch result type: {rtype}"
        by_index[index] = result
    return by_index


def is_retryable_api_error(exc: Exception) -> bool:
    if isinstance(exc, anthropic.RateLimitError):
        return True
    if isinstance(exc, anthropic.APIConnectionError):
        return True
    if isinstance(exc, anthropic.InternalServerError):
        return True
    if isinstance(exc, anthropic.APIStatusError) and exc.status_code >= 500:
        return True
    return False


def call_claude(client: anthropic.Anthropic, system_prompt: str, user_message: str) -> anthropic.types.Message:
    last_err: Exception | None = None
    for attempt in range(API_MAX_RETRIES):
        try:
            return client.messages.create(
                model=resolve_translation_model(),
                max_tokens=MAX_TOKENS,
                system=cached_system_prompt(system_prompt),
                messages=[{"role": "user", "content": user_message}],
            )
        except Exception as exc:
            last_err = exc
            if attempt < API_MAX_RETRIES - 1 and is_retryable_api_error(exc):
                time.sleep(min(2**attempt, 30))
                continue
            raise
    raise last_err or RuntimeError("API call failed")


def format_file_list(paths: list[Path], max_names: int = 2) -> str:
    if not paths:
        return ""
    names = [p.name for p in sorted(paths, key=lambda p: p.name.lower())]
    if len(names) <= max_names:
        return ", ".join(names)
    shown = ", ".join(names[:max_names])
    return f"{shown} (+{len(names) - max_names} more)"


def format_pair_preview(
    index: int,
    nahuatl: str,
    english: str,
    *,
    uncertain: bool = False,
) -> list[str]:
    flag = "  ⚠ UNCERTAIN MATCH — review manually" if uncertain else ""
    return [
        f"=== Pair {index} ==={flag}",
        f"  Nahuatl : {first_words(nahuatl)}",
        f"  English : {first_words(english)}",
        "",
    ]


def write_run_summary(
    out_dir: Path,
    *,
    mode: str,
    results: list[PassageResult],
    failed: list[int],
    truncated: list[int],
    nahuatl_paths: list[Path],
    english_paths: list[Path],
    input_tokens: int,
    output_tokens: int,
    cost: float,
    batch_ids: list[str] | None = None,
    test_mode: bool = False,
    skipped_count: int = 0,
    cache_read_tokens: int = 0,
    cache_creation_tokens: int = 0,
    ai_alignment: bool = False,
    alignment_input_tokens: int = 0,
    alignment_output_tokens: int = 0,
    alignment_cost: float = 0.0,
) -> None:
    model = resolve_translation_model()
    names = output_filenames(test_mode=test_mode)
    stats = {
        "translation_model": model,
        "alignment_model": ALIGNMENT_MODEL if ai_alignment else None,
        "model": model,
        "mode": mode,
        "ai_alignment": ai_alignment,
        "test_mode": test_mode,
        "passages_total": len(results),
        "passages_succeeded": len([r for r in results if not r.error and not r.truncated]),
        "passages_failed": len(failed),
        "passages_truncated": len(truncated),
        "passages_skipped_resume": skipped_count,
        "failed_passage_numbers": failed,
        "truncated_passage_numbers": truncated,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cache_read_input_tokens": cache_read_tokens,
        "cache_creation_input_tokens": cache_creation_tokens,
        "alignment_input_tokens": alignment_input_tokens,
        "alignment_output_tokens": alignment_output_tokens,
        "alignment_cost_usd": round(alignment_cost, 4),
        "translation_cost_usd": round(cost - alignment_cost, 4),
        "estimated_cost_usd": round(cost, 4),
        "batch_discount_applied": mode == "batch",
        "batch_ids": batch_ids or [],
        "nahuatl_files": [str(p) for p in nahuatl_paths],
        "english_files": [str(p) for p in english_paths],
    }
    (out_dir / names["summary_json"]).write_text(json.dumps(stats, indent=2), encoding="utf-8")
    lines = [
        "Nahuatl Codex Translator — run summary",
        f"Translation model: {model}",
        f"Alignment model: {ALIGNMENT_MODEL if ai_alignment else 'n/a (mechanical pairing)'}",
        f"Mode: {'Batch (~50% off)' if mode == 'batch' else 'Live (full price)'}",
        f"Passages: {stats['passages_succeeded']}/{stats['passages_total']} OK",
        f"Skipped (resume): {skipped_count}",
        f"Tokens: {input_tokens} in / {output_tokens} out",
        f"Prompt cache: {cache_read_tokens} read / {cache_creation_tokens} written",
    ]
    if ai_alignment:
        lines.append(f"Alignment tokens: {alignment_input_tokens} in / {alignment_output_tokens} out")
    lines.append(f"Alignment cost: ${alignment_cost:.4f}")
    lines.append(f"Translation cost: ${cost - alignment_cost:.4f}")
    fresh_in = max(0, input_tokens - cache_read_tokens)
    lines.append(f"Fresh input tokens (non-cache): {fresh_in}")
    lines.append(f"Total estimated cost: ${cost:.4f}")
    if truncated:
        lines.append(f"Truncated passages: {truncated}")
    if failed:
        lines.append(f"Failed passages: {failed}")
    (out_dir / names["summary_txt"]).write_text("\n".join(lines) + "\n", encoding="utf-8")


class DropZone(tk.Frame):
    def __init__(self, master, label: str, on_files, *, allow_multiple: bool = True, **kwargs):
        super().__init__(master, relief=tk.GROOVE, borderwidth=2, **kwargs)
        self.on_files = on_files
        self.allow_multiple = allow_multiple
        self.file_paths: list[Path] = []
        self._default_bg = self.cget("bg")

        self.label = tk.Label(self, text=label, font=("Segoe UI", 11), wraplength=240)
        self.label.pack(expand=True, fill=tk.BOTH, padx=12, pady=(20, 8))

        self.path_label = tk.Label(self, text="No files loaded", font=("Segoe UI", 9), fg="#555", wraplength=240)
        self.path_label.pack(padx=8, pady=(0, 8))

        btn_row = tk.Frame(self)
        btn_row.pack(pady=(0, 10))
        ttk.Button(btn_row, text="Browse…", command=self._browse).pack(side=tk.LEFT, padx=3)
        ttk.Button(btn_row, text="Clear", command=self._clear).pack(side=tk.LEFT, padx=3)

        self.drop_target_register(DND_FILES)
        self.dnd_bind("<<Drop>>", self._handle_drop)
        self.dnd_bind("<<DragEnter>>", self._drag_enter)
        self.dnd_bind("<<DragLeave>>", self._drag_leave)

    def _drag_enter(self, _event):
        self.configure(bg="#dbeafe")
        self.label.configure(bg="#dbeafe")
        self.path_label.configure(bg="#dbeafe")

    def _drag_leave(self, _event):
        self._reset_bg()

    def _reset_bg(self):
        bg = self._default_bg
        self.configure(bg=bg)
        self.label.configure(bg=bg)
        self.path_label.configure(bg=bg)

    def _valid_paths(self, paths: list[Path]) -> list[Path]:
        good: list[Path] = []
        bad: list[str] = []
        for path in paths:
            if path.suffix.lower() not in TEXT_SUFFIXES:
                bad.append(path.name)
                continue
            if not path.is_file():
                bad.append(path.name)
                continue
            good.append(path)
        if bad:
            messagebox.showwarning(
                "Skipped files",
                "These were skipped (not readable .txt/.md files):\n" + "\n".join(bad),
            )
        return good

    def _handle_drop(self, event):
        self._reset_bg()
        paths = parse_drop_paths(self.winfo_toplevel(), event.data)
        paths = self._valid_paths(paths)
        if not paths:
            return
        if not self.allow_multiple and len(paths) > 1:
            paths = [paths[0]]
        self.add_files(paths, replace=not self.allow_multiple)

    def _browse(self):
        selected = filedialog.askopenfilenames(
            parent=self.winfo_toplevel(),
            title="Select text file(s)",
            filetypes=[("Text files", "*.txt *.text *.md"), ("All files", "*.*")],
        )
        if not selected:
            return
        paths = self._valid_paths([Path(p) for p in selected])
        if paths:
            self.add_files(paths, replace=False)

    def _clear(self):
        self.file_paths = []
        self.path_label.configure(text="No files loaded")
        self.on_files([])

    def add_files(self, paths: list[Path], *, replace: bool = False):
        if replace:
            merged = list(paths)
        else:
            merged = list(self.file_paths)
            for path in paths:
                resolved = path.resolve()
                if resolved not in {p.resolve() for p in merged}:
                    merged.append(path)
        merged = sorted(merged, key=lambda p: p.name.lower())
        self.file_paths = merged
        count = len(merged)
        if count == 0:
            self.path_label.configure(text="No files loaded")
        elif count == 1:
            self.path_label.configure(text=merged[0].name)
        else:
            self.path_label.configure(text=f"{count} files: {format_file_list(merged)}")
        self.on_files(merged)


class TranslatorApp:
    def __init__(self):
        self.root = TkinterDnD.Tk()
        self.root.title(f"Nahuatl Codex Translator — {TRANSLATION_MODEL}")
        self.root.geometry("860x860")
        self.root.minsize(760, 720)

        self.state = RunState()
        self._running = False
        self._progress_done = 0
        self._progress_total = 0
        self._mode_radios: list[ttk.Radiobutton] = []
        self._split_widgets: list[tk.Widget] = []
        self._ai_align_cb: ttk.Checkbutton | None = None
        self._saved_prompt_text = ""
        self._prompt_update_in_progress = False
        self._summary_alignment_cost = 0.0
        self._summary_alignment_input = 0
        self._summary_alignment_output = 0

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self):
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        translate_tab = ttk.Frame(self.notebook)
        prompt_tab = ttk.Frame(self.notebook)
        self.notebook.add(translate_tab, text="Translate")
        self.notebook.add(prompt_tab, text="Prompt")

        self._build_translate_tab(translate_tab)
        self._build_progress_bar()
        self._build_prompt_tab(prompt_tab)

    def _build_translate_tab(self, parent: ttk.Frame):
        pad = {"padx": 10, "pady": 4}

        top = tk.Frame(parent)
        top.pack(fill=tk.X, **pad)

        zones = tk.Frame(top)
        zones.pack(fill=tk.X)

        self.nahuatl_zone = DropZone(
            zones,
            "Drop Nahuatl file(s) here\n(or Browse — multiple OK)",
            self._on_nahuatl,
        )
        self.nahuatl_zone.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=(0, 5))

        self.english_zone = DropZone(
            zones,
            "Drop English reference file(s) here\n(or Browse — multiple OK)",
            self._on_english,
        )
        self.english_zone.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=(5, 0))

        split_frame = ttk.LabelFrame(parent, text="Split method")
        split_frame.pack(fill=tk.X, padx=10, pady=4)

        self.split_method_var = tk.StringVar(value="Chapter headings")
        method_row = tk.Frame(split_frame)
        method_row.pack(fill=tk.X, padx=6, pady=4)
        ttk.Label(method_row, text="Method:").pack(side=tk.LEFT)
        method_cb = ttk.Combobox(
            method_row,
            textvariable=self.split_method_var,
            values=["Chapter headings", "Paragraph breaks (\\n\\n)", "Every N words"],
            state="readonly",
            width=28,
        )
        method_cb.pack(side=tk.LEFT, padx=6)
        method_cb.bind("<<ComboboxSelected>>", lambda _e: self._on_split_setting_changed())
        self._split_widgets.append(method_cb)

        regex_row = tk.Frame(split_frame)
        regex_row.pack(fill=tk.X, padx=6, pady=2)
        ttk.Label(regex_row, text="Chapter regex:").pack(side=tk.LEFT)
        self.chapter_regex_var = tk.StringVar(value=DEFAULT_CHAPTER_REGEX)
        regex_entry = ttk.Entry(regex_row, textvariable=self.chapter_regex_var)
        regex_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=6)
        regex_entry.bind("<KeyRelease>", lambda _e: self._on_split_setting_changed())
        self._split_widgets.append(regex_entry)

        words_row = tk.Frame(split_frame)
        words_row.pack(fill=tk.X, padx=6, pady=2)
        ttk.Label(words_row, text="Words per passage:").pack(side=tk.LEFT)
        self.words_per_passage_var = tk.StringVar(value=str(DEFAULT_WORDS_PER_PASSAGE))
        words_entry = ttk.Entry(words_row, textvariable=self.words_per_passage_var, width=8)
        words_entry.pack(side=tk.LEFT, padx=6)
        words_entry.bind("<KeyRelease>", lambda _e: self._on_split_setting_changed())
        self._split_widgets.append(words_entry)

        self.split_count_var = tk.StringVar(value="Passage counts: —")
        ttk.Label(split_frame, textvariable=self.split_count_var).pack(anchor=tk.W, padx=6, pady=2)
        self.large_passage_var = tk.StringVar(value="")
        ttk.Label(split_frame, textvariable=self.large_passage_var, foreground="#b45309").pack(
            anchor=tk.W, padx=6, pady=(0, 4)
        )

        self.ai_alignment_var = tk.BooleanVar(value=True)
        self._ai_align_cb = ttk.Checkbutton(
            split_frame,
            text=f"AI-assisted alignment ({ALIGNMENT_MODEL}) — recommended",
            variable=self.ai_alignment_var,
            command=self._invalidate_split,
        )
        self._ai_align_cb.pack(anchor=tk.W, padx=8, pady=(0, 4))

        btn_row = tk.Frame(parent)
        btn_row.pack(fill=tk.X, **pad)

        self.preview_btn = ttk.Button(btn_row, text="Split && Preview", command=self.split_and_preview)
        self.preview_btn.pack(side=tk.LEFT)

        self.run_btn = ttk.Button(btn_row, text="Run Translation", command=self.run_translation, state=tk.DISABLED)
        self.run_btn.pack(side=tk.LEFT, padx=6)

        self.run_anyway_btn = ttk.Button(
            btn_row,
            text="Run anyway",
            command=lambda: self.run_translation(allow_uncertain=True),
            state=tk.DISABLED,
        )
        self.run_anyway_btn.pack(side=tk.LEFT, padx=6)

        self.retry_btn = ttk.Button(
            btn_row, text="Retry failed passages", command=self.retry_failed, state=tk.DISABLED
        )
        self.retry_btn.pack(side=tk.LEFT, padx=6)

        self.mode_var = tk.StringVar(value="batch")
        mode_frame = tk.Frame(btn_row)
        mode_frame.pack(side=tk.RIGHT)
        for text, value in (("Batch (50% off)", "batch"), ("Live (immediate)", "live")):
            rb = ttk.Radiobutton(mode_frame, text=text, variable=self.mode_var, value=value)
            rb.pack(side=tk.LEFT, padx=4)
            self._mode_radios.append(rb)

        self.test_mode_var = tk.BooleanVar(value=False)
        self._test_mode_cb = ttk.Checkbutton(
            btn_row,
            text=f"Test mode (first {TEST_MODE_WORD_LIMIT} words)",
            variable=self.test_mode_var,
            command=self._invalidate_split,
        )
        self._test_mode_cb.pack(side=tk.LEFT, padx=6)

        self.alignment_label = tk.Label(
            parent,
            text="Alignment: run Split & Preview",
            font=("Segoe UI", 11, "bold"),
            fg="#666",
        )
        self.alignment_label.pack(anchor=tk.W, padx=12, pady=2)

        tk.Label(
            parent,
            text=(
                f"Alignment: {ALIGNMENT_MODEL}  ·  "
                f"Translation: {resolve_translation_model()}  ·  max {MAX_TOKENS} tokens/passage"
            ),
            font=("Segoe UI", 9),
            fg="#666",
        ).pack(anchor=tk.W, padx=12)

        preview_frame = ttk.LabelFrame(parent, text="Alignment preview")
        preview_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=4)

        jump_row = tk.Frame(preview_frame)
        jump_row.pack(fill=tk.X, padx=6, pady=4)
        ttk.Label(jump_row, text="Preview pair #").pack(side=tk.LEFT)
        self.preview_pair_var = tk.StringVar(value="1")
        ttk.Entry(jump_row, textvariable=self.preview_pair_var, width=8).pack(side=tk.LEFT, padx=4)
        ttk.Button(jump_row, text="Show", command=self._show_preview_pair).pack(side=tk.LEFT)

        self.preview_text = scrolledtext.ScrolledText(
            preview_frame, height=14, wrap=tk.WORD, font=("Consolas", 10)
        )
        self.preview_text.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)
        self.preview_text.tag_configure("uncertain", background="#fef9c3")
        self.preview_text.configure(state=tk.DISABLED)

    def _build_prompt_tab(self, parent: ttk.Frame):
        header = tk.Frame(parent)
        header.pack(fill=tk.X, padx=10, pady=(8, 4))

        ttk.Label(
            header,
            text=f"System prompt — saved to {PROMPT_FILE.name}",
            font=("Segoe UI", 10, "bold"),
        ).pack(side=tk.LEFT)

        self.prompt_dirty_var = tk.StringVar(value="")
        tk.Label(header, textvariable=self.prompt_dirty_var, fg="#b45309", font=("Segoe UI", 9)).pack(
            side=tk.LEFT, padx=10
        )

        btn_row = tk.Frame(parent)
        btn_row.pack(fill=tk.X, padx=10, pady=(0, 4))
        ttk.Button(btn_row, text="Save", command=self._save_prompt).pack(side=tk.LEFT)
        ttk.Button(btn_row, text="Reset to saved", command=self._reset_prompt).pack(side=tk.LEFT, padx=6)

        ttk.Label(
            parent,
            text="Translation runs use the saved file on disk. Save before running if you edit here.",
            font=("Segoe UI", 9),
            foreground="#666",
        ).pack(anchor=tk.W, padx=10, pady=(0, 4))

        self.prompt_text = scrolledtext.ScrolledText(parent, wrap=tk.WORD, font=("Consolas", 10))
        self.prompt_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        self.prompt_text.bind("<KeyRelease>", self._on_prompt_edit)
        self.prompt_text.bind("<<Paste>>", lambda _e: self.root.after(10, self._on_prompt_edit))
        self._load_prompt_into_editor()

    def _build_progress_bar(self):
        pad = {"padx": 10, "pady": 4}
        progress_frame = tk.Frame(self.root)
        progress_frame.pack(fill=tk.X, **pad)

        self.progress = ttk.Progressbar(progress_frame, mode="determinate")
        self.progress.pack(fill=tk.X, pady=4)

        self.status_var = tk.StringVar(value="Drop file(s) on both sides, then Split & Preview.")
        tk.Label(progress_frame, textvariable=self.status_var, anchor=tk.W, wraplength=820).pack(fill=tk.X)

    def _prompt_editor_text(self) -> str:
        return self.prompt_text.get("1.0", "end-1c")

    def _prompt_is_dirty(self) -> bool:
        return self._prompt_editor_text() != self._saved_prompt_text

    def _update_prompt_dirty_indicator(self) -> None:
        if self._prompt_is_dirty():
            self.prompt_dirty_var.set("● Unsaved changes")
            self.notebook.tab(1, text="Prompt *")
        else:
            self.prompt_dirty_var.set("")
            self.notebook.tab(1, text="Prompt")

    def _load_prompt_into_editor(self) -> None:
        self._prompt_update_in_progress = True
        try:
            if PROMPT_FILE.is_file():
                text = PROMPT_FILE.read_text(encoding="utf-8")
            else:
                text = ""
            self._saved_prompt_text = text
            self.prompt_text.delete("1.0", tk.END)
            self.prompt_text.insert("1.0", text)
        finally:
            self._prompt_update_in_progress = False
        self._update_prompt_dirty_indicator()

    def _on_prompt_edit(self, _event=None) -> None:
        if self._prompt_update_in_progress:
            return
        self._update_prompt_dirty_indicator()

    def _save_prompt(self) -> bool:
        text = self._prompt_editor_text()
        if not text.strip():
            messagebox.showerror("Empty prompt", "System prompt cannot be empty.")
            return False
        try:
            save_system_prompt(text)
        except OSError as exc:
            messagebox.showerror("Save error", str(exc))
            return False
        lower = text.lower()
        if "<english>" not in lower or "<spanish>" not in lower:
            messagebox.showwarning(
                "Missing output tags",
                "This prompt does not mention <english> and <spanish> tags.\n\n"
                "Translation will fail to parse without them. Consider adding the "
                "XML output format from the default prompt.",
            )
        self._saved_prompt_text = text
        self._update_prompt_dirty_indicator()
        self.status_var.set(f"Saved system prompt to {PROMPT_FILE.name}")
        return True

    def _reset_prompt(self) -> None:
        if self._prompt_is_dirty():
            if not messagebox.askyesno(
                "Discard edits?",
                "Reload the saved prompt from disk and discard unsaved edits?",
            ):
                return
        self._load_prompt_into_editor()
        self.status_var.set(f"Reloaded system prompt from {PROMPT_FILE.name}")

    def _ensure_prompt_saved_for_run(self) -> bool:
        if not self._prompt_is_dirty():
            return True
        answer = messagebox.askyesnocancel(
            "Unsaved prompt",
            "The system prompt has unsaved changes.\n\nSave to disk before running translation?",
        )
        if answer is None:
            return False
        if answer:
            return self._save_prompt()
        messagebox.showinfo(
            "Save required",
            "Translation uses the saved prompt file.\n"
            "Save your edits on the Prompt tab, or Reset to saved.",
        )
        return False

    def _split_method_key(self) -> str:
        label = self.split_method_var.get()
        if label.startswith("Paragraph"):
            return SPLIT_PARAGRAPH
        if label.startswith("Every"):
            return SPLIT_WORDS
        return SPLIT_CHAPTER

    def _words_per_passage(self) -> int:
        try:
            return max(1, int(self.words_per_passage_var.get().strip()))
        except ValueError:
            return DEFAULT_WORDS_PER_PASSAGE

    def _invalidate_split(self):
        self.state.pairs = []
        self.state.aligned = False
        self.state.alignment_uncertain = []
        self.state.pair_uncertain = {}
        self._refresh_run_buttons()
        self._update_alignment_label(0, 0, aligned=False)
        if self.state.nahuatl_paths and self.state.english_paths:
            self._refresh_split_counts()

    def _refresh_run_buttons(self) -> None:
        uncertain = bool(self.state.alignment_uncertain)
        can_run = bool(self.state.pairs and self.state.aligned and not self._running)
        if uncertain:
            self.run_btn.configure(state=tk.DISABLED)
            self.run_anyway_btn.configure(state=tk.NORMAL if can_run else tk.DISABLED)
        else:
            self.run_btn.configure(state=tk.NORMAL if can_run else tk.DISABLED)
            self.run_anyway_btn.configure(state=tk.DISABLED)
        can_retry = bool(self.state.failed and not self._running)
        self.retry_btn.configure(state=tk.NORMAL if can_retry else tk.DISABLED)

    def _on_split_setting_changed(self):
        if self.state.nahuatl_text and self.state.english_text:
            self._refresh_split_counts()
        self._invalidate_split()

    def _on_nahuatl(self, paths: list[Path]):
        self.state.nahuatl_paths = paths
        self._invalidate_split()
        self._update_ready_status()

    def _on_english(self, paths: list[Path]):
        self.state.english_paths = paths
        self._invalidate_split()
        self._update_ready_status()

    def _update_ready_status(self):
        n = len(self.state.nahuatl_paths)
        e = len(self.state.english_paths)
        if n and e:
            self.status_var.set(f"Ready — {n} Nahuatl + {e} English file(s). Click Split & Preview.")
        elif n or e:
            missing = "English" if n else "Nahuatl"
            self.status_var.set(f"Loaded {max(n, e)} file(s). Still need {missing} file(s).")
        else:
            self.status_var.set("Drop file(s) on both sides, then Split & Preview.")

    def _load_merged_texts(self) -> tuple[str, str] | None:
        if not self.state.nahuatl_paths or not self.state.english_paths:
            return None
        try:
            nahuatl_text = read_text_files(self.state.nahuatl_paths)
            english_text = read_text_files(self.state.english_paths)
        except OSError as exc:
            messagebox.showerror("Read error", str(exc))
            return None
        if not nahuatl_text.strip() or not english_text.strip():
            messagebox.showerror("Empty input", "One or both sides are empty after reading files.")
            return None
        if self.test_mode_var.get():
            nahuatl_text, _, _ = truncate_words(nahuatl_text, TEST_MODE_WORD_LIMIT)
            english_text, _, _ = truncate_words(english_text, TEST_MODE_WORD_LIMIT)
        return nahuatl_text, english_text

    def _split_kwargs(self) -> dict:
        return {
            "method": self._split_method_key(),
            "chapter_pattern": self.chapter_regex_var.get().strip() or DEFAULT_CHAPTER_REGEX,
            "words_per_passage": self._words_per_passage(),
        }

    def _alignment_settings(self) -> dict:
        return split_settings_fingerprint(
            self._split_kwargs()["method"],
            self._split_kwargs()["chapter_pattern"],
            self._split_kwargs()["words_per_passage"],
            self.test_mode_var.get(),
            self.ai_alignment_var.get(),
        )

    def _split_nahuatl_only(self, nahuatl_text: str) -> list[str]:
        return split_passages(nahuatl_text, **self._split_kwargs())

    def _check_large_passages(self, passages: list[str]) -> list[int]:
        return [i for i, p in enumerate(passages, start=1) if count_words(p) > LARGE_PASSAGE_WORDS]

    def _refresh_split_counts(self):
        texts = self._load_merged_texts()
        if not texts:
            return
        nahuatl_text, english_text = texts
        self.state.nahuatl_text = nahuatl_text
        self.state.english_text = english_text
        try:
            nah = self._split_nahuatl_only(nahuatl_text)
            eng = self._split_nahuatl_only(english_text) if not self.ai_alignment_var.get() else []
        except ValueError as exc:
            self.split_count_var.set(f"Split error: {exc}")
            return
        if self.ai_alignment_var.get():
            self.split_count_var.set(f"Passage counts: Nahuatl {len(nah)} (English split skipped — AI matching)")
        else:
            self.split_count_var.set(f"Passage counts: Nahuatl {len(nah)} | English {len(eng)}")
        large = sorted(set(self._check_large_passages(nah) + self._check_large_passages(eng)))
        if large:
            shown = large[:8]
            extra = f" (+{len(large) - 8} more)" if len(large) > 8 else ""
            self.large_passage_var.set(
                f"Warning: passage(s) {shown}{extra} exceed {LARGE_PASSAGE_WORDS} words — may truncate."
            )
        else:
            self.large_passage_var.set("")

    def _update_alignment_label(
        self,
        nah_count: int,
        eng_count: int,
        *,
        aligned: bool,
        ai_mode: bool = False,
        uncertain_count: int = 0,
    ):
        if nah_count == 0 and eng_count == 0:
            self.alignment_label.configure(text="Alignment: run Split & Preview", fg="#666")
            return
        if ai_mode:
            if uncertain_count:
                self.alignment_label.configure(
                    text=(
                        f"Nahuatl: {nah_count} passages | AI-matched: {nah_count} — "
                        f"{uncertain_count} UNCERTAIN (review before run)"
                    ),
                    fg="#b45309",
                )
            else:
                self.alignment_label.configure(
                    text=f"Nahuatl: {nah_count} passages | AI-matched: {nah_count} — ALIGNED",
                    fg="#15803d",
                )
            return
        if aligned:
            self.alignment_label.configure(
                text=f"Nahuatl: {nah_count} passages | English: {eng_count} passages — ALIGNED",
                fg="#15803d",
            )
        else:
            self.alignment_label.configure(
                text=f"Nahuatl: {nah_count} | English: {eng_count} — MISMATCH, do not run",
                fg="#b91c1c",
            )

    def _render_preview(self, pair_indices: list[int], pairs: list[tuple[str, str]] | None = None):
        source_pairs = pairs if pairs is not None else self.state.pairs
        lines: list[str] = []
        if self.test_mode_var.get():
            lines.append(f"TEST MODE — first {TEST_MODE_WORD_LIMIT} words per side\n")
        lines.append(
            f"Sources: Nahuatl [{format_file_list(self.state.nahuatl_paths)}] | "
            f"English [{format_file_list(self.state.english_paths)}]\n"
        )
        lines.append(f"Split: {self.split_method_var.get()}\n")
        if self.state.ai_alignment:
            lines.append(f"Alignment: AI-assisted ({ALIGNMENT_MODEL})\n")
        pair_blocks: list[tuple[list[str], bool]] = []
        for i in pair_indices:
            if i < 1 or i > len(source_pairs):
                pair_blocks.append(([f"Pair {i}: out of range (1–{len(source_pairs)})", ""], False))
                continue
            nah, eng = source_pairs[i - 1]
            uncertain = self.state.pair_uncertain.get(i, False)
            pair_blocks.append((format_pair_preview(i, nah, eng, uncertain=uncertain), uncertain))
        self.preview_text.configure(state=tk.NORMAL)
        self.preview_text.delete("1.0", tk.END)
        if lines:
            self.preview_text.insert(tk.END, "\n".join(lines))
        for block_lines, uncertain in pair_blocks:
            block_text = "\n".join(block_lines)
            if uncertain:
                self.preview_text.insert(tk.END, block_text, "uncertain")
            else:
                self.preview_text.insert(tk.END, block_text)
        self.preview_text.configure(state=tk.DISABLED)

    def _show_preview_pair(self):
        if not self.state.pairs:
            messagebox.showinfo("No pairs", "Run Split & Preview first.")
            return
        try:
            num = int(self.preview_pair_var.get().strip())
        except ValueError:
            messagebox.showwarning("Invalid number", "Enter a passage number.")
            return
        self._render_preview([num])

    def split_and_preview(self):
        if not self.state.nahuatl_paths or not self.state.english_paths:
            messagebox.showinfo("Missing files", "Load at least one Nahuatl and one English file.")
            return

        texts = self._load_merged_texts()
        if not texts:
            return

        nahuatl_text, english_text = texts
        self.state.nahuatl_text = nahuatl_text
        self.state.english_text = english_text
        self.state.test_mode = self.test_mode_var.get()
        self.state.ai_alignment = self.ai_alignment_var.get()

        try:
            nahuatl_passages = self._split_nahuatl_only(nahuatl_text)
        except ValueError as exc:
            messagebox.showerror("Split error", str(exc))
            return

        if not nahuatl_passages:
            messagebox.showerror("Split error", "Nahuatl text produced no passages.")
            return

        self.state.nahuatl_passages = nahuatl_passages
        self.state.nahuatl_passage_count = len(nahuatl_passages)
        self._show_large_passage_warning(nahuatl_passages, [])

        if self.state.ai_alignment:
            self._set_running_ui(True)
            self.status_var.set("Splitting Nahuatl… starting AI alignment (Haiku)…")
            thread = threading.Thread(target=self._split_and_align_worker, daemon=True)
            thread.start()
            return

        try:
            english_passages = self._split_nahuatl_only(english_text)
        except ValueError as exc:
            messagebox.showerror("Split error", str(exc))
            return
        if not english_passages:
            messagebox.showerror("Split error", "English text produced no passages.")
            return
        self._finish_mechanical_split(nahuatl_passages, english_passages)

    def _show_large_passage_warning(self, nahuatl_passages: list[str], english_passages: list[str]):
        large = sorted(
            set(self._check_large_passages(nahuatl_passages) + self._check_large_passages(english_passages))
        )
        if large:
            shown = large[:8]
            extra = f" (+{len(large) - 8} more)" if len(large) > 8 else ""
            self.large_passage_var.set(
                f"Warning: passage(s) {shown}{extra} exceed {LARGE_PASSAGE_WORDS} words — may truncate."
            )
        else:
            self.large_passage_var.set("")

    def _finish_mechanical_split(self, nahuatl_passages: list[str], english_passages: list[str]):
        self.state.english_passage_count = len(english_passages)
        aligned = len(nahuatl_passages) == len(english_passages)
        self.state.aligned = aligned
        self.state.pair_uncertain = {}
        self.state.alignment_uncertain = []

        self._update_alignment_label(len(nahuatl_passages), len(english_passages), aligned=aligned)
        self.split_count_var.set(
            f"Passage counts: Nahuatl {len(nahuatl_passages)} | English {len(english_passages)}"
        )
        self._show_large_passage_warning(nahuatl_passages, english_passages)

        if not aligned:
            self.state.pairs = []
            self._refresh_run_buttons()
            preview_cap = min(PREVIEW_COUNT, len(nahuatl_passages), len(english_passages))
            mismatch_pairs = list(zip(nahuatl_passages[:preview_cap], english_passages[:preview_cap]))
            self._render_preview(list(range(1, preview_cap + 1)), pairs=mismatch_pairs)
            messagebox.showerror(
                "Alignment mismatch",
                f"Nahuatl: {len(nahuatl_passages)} passages\n"
                f"English: {len(english_passages)} passages\n\n"
                "Counts must match exactly, or enable AI-assisted alignment.\n"
                "Run is blocked until aligned.",
            )
            return

        self.state.pairs = list(zip(nahuatl_passages, english_passages))
        preview_nums = list(range(1, min(PREVIEW_COUNT, len(self.state.pairs)) + 1))
        self._render_preview(preview_nums)
        self._refresh_run_buttons()
        test_label = f" [TEST ~{TEST_MODE_WORD_LIMIT} words]" if self.state.test_mode else ""
        self.status_var.set(
            f"Aligned — {len(self.state.pairs)} pairs{test_label}. Review preview, then Run Translation."
        )

    def _apply_alignment_entries(self, entries: list[AlignmentEntry]):
        self.state.pairs = [(e.nahuatl, e.english) for e in entries]
        self.state.pair_uncertain = {e.index: e.uncertain for e in entries}
        self.state.alignment_uncertain = [e.index for e in entries if e.uncertain]
        self.state.english_passage_count = len(entries)
        self.state.aligned = True
        uncertain = len(self.state.alignment_uncertain)
        self._update_alignment_label(
            len(entries),
            len(entries),
            aligned=True,
            ai_mode=True,
            uncertain_count=uncertain,
        )
        self.split_count_var.set(
            f"Passage counts: Nahuatl {len(entries)} | AI-matched English: {len(entries)}"
        )
        preview_nums = list(range(1, min(PREVIEW_COUNT, len(self.state.pairs)) + 1))
        self._render_preview(preview_nums)
        self._refresh_run_buttons()
        test_label = f" [TEST ~{TEST_MODE_WORD_LIMIT} words]" if self.state.test_mode else ""
        note = f" ({uncertain} uncertain)" if uncertain else ""
        self.status_var.set(
            f"AI-aligned — {len(self.state.pairs)} pairs{note}{test_label}. Review preview, then Run."
        )
        if uncertain:
            messagebox.showwarning(
                "Uncertain matches",
                f"{uncertain} passage(s) had no clear English match (yellow in preview).\n"
                "Run Translation is blocked — use Run anyway after review, or re-split.",
            )

    def _split_and_align_worker(self):
        try:
            api_key = resolve_api_key()
            client = anthropic.Anthropic(api_key=api_key)
        except Exception as exc:
            self.root.after(0, lambda: self._on_align_error(str(exc)))
            return

        out_dir = output_directory(self.state.nahuatl_paths)
        settings = self._alignment_settings()
        content_hash = source_content_hash(self.state.nahuatl_text, self.state.english_text)

        def on_progress(done: int, total: int, msg: str):
            self.root.after(0, lambda: self.status_var.set(msg))
            self.root.after(0, lambda: self.progress.configure(maximum=total, value=done - 1))

        try:
            entries, align_in, align_out = align_nahuatl_passages(
                client,
                self.state.nahuatl_passages,
                self.state.english_text,
                out_dir=out_dir,
                test_mode=self.state.test_mode,
                content_hash=content_hash,
                settings=settings,
                on_progress=on_progress,
            )
            if align_in == 0 and align_out == 0:
                cache_path = alignment_map_path(out_dir, self.state.test_mode)
                if cache_path.is_file():
                    self.root.after(0, lambda: self.status_var.set("Loaded cached alignment_map.json"))
        except Exception as exc:
            self.root.after(0, lambda: self._on_align_error(str(exc)))
            return

        self.state.alignment_input_tokens = align_in
        self.state.alignment_output_tokens = align_out
        self.state.alignment_cost = haiku_token_cost(align_in, align_out)
        self.root.after(0, lambda: self._apply_alignment_entries(entries))
        self.root.after(0, self._set_running_ui(False))

    def _on_align_error(self, msg: str):
        self._set_running_ui(False)
        messagebox.showerror("Alignment error", msg)

    def retry_failed(self):
        if self._running:
            return
        if not self.state.failed:
            messagebox.showinfo("Nothing to retry", "No failed passages from the last run.")
            return
        if not self.state.pairs:
            messagebox.showinfo("No pairs", "Run Split & Preview first.")
            return
        self.state.retry_indices = list(self.state.failed)
        self.run_translation(retry_only=True)

    def run_translation(self, *, retry_only: bool = False, allow_uncertain: bool = False):
        if self._running:
            return
        if not self.state.pairs:
            messagebox.showinfo("Nothing to run", "Split & Preview first.")
            return
        if not retry_only and not self.state.aligned:
            messagebox.showerror("Not aligned", "Passage counts must match before running.")
            return
        if not allow_uncertain and self.state.alignment_uncertain:
            messagebox.showwarning(
                "Uncertain alignment",
                f"{len(self.state.alignment_uncertain)} passage(s) have uncertain English matches.\n"
                "Review them in preview (yellow), then click Run anyway if you accept the risk.",
            )
            return
        if not self._ensure_prompt_saved_for_run():
            return

        try:
            resolve_api_key()
            load_system_prompt()
        except (ValueError, FileNotFoundError) as exc:
            messagebox.showerror("Setup error", str(exc))
            return

        if retry_only:
            self._summary_alignment_cost = 0.0
            self._summary_alignment_input = 0
            self._summary_alignment_output = 0
        else:
            self._summary_alignment_cost = self.state.alignment_cost
            self._summary_alignment_input = self.state.alignment_input_tokens
            self._summary_alignment_output = self.state.alignment_output_tokens
            self.state.alignment_cost = 0.0
            self.state.alignment_input_tokens = 0
            self.state.alignment_output_tokens = 0

        self._set_running_ui(True)
        if not retry_only:
            self.state.retry_indices = []
        self.state.total_input_tokens = 0
        self.state.total_output_tokens = 0
        self.state.total_cost = 0.0
        self.state.cache_read_tokens = 0
        self.state.cache_creation_tokens = 0
        self._progress_done = 0
        self._progress_total = len(self._indices_to_run())
        self.progress.configure(maximum=max(1, self._progress_total), value=0)

        worker = self._translate_worker_batch if self.mode_var.get() == "batch" else self._translate_worker_live
        thread = threading.Thread(target=worker, daemon=True)
        thread.start()

    def _set_running_ui(self, running: bool) -> None:
        self._running = running
        state = tk.DISABLED if running else tk.NORMAL
        self.preview_btn.configure(state=state)
        self._refresh_run_buttons()
        for rb in self._mode_radios:
            rb.configure(state=state)
        if self._test_mode_cb is not None:
            self._test_mode_cb.configure(state=state)
        if self._ai_align_cb is not None:
            self._ai_align_cb.configure(state=state)
        for w in self._split_widgets:
            w.configure(state=state)
        if running:
            self.run_anyway_btn.configure(state=tk.DISABLED)
        cursor = "watch" if running else ""
        self.nahuatl_zone.configure(cursor=cursor)
        self.english_zone.configure(cursor=cursor)

    def _indices_to_run(self) -> list[int]:
        if self.state.retry_indices:
            return sorted(self.state.retry_indices)
        return list(range(1, len(self.state.pairs) + 1))

    def _try_load_existing(self, out_dir: Path, index: int) -> PassageResult | None:
        nahuatl, english = self.state.pairs[index - 1]
        expected_fp = pair_fingerprint(nahuatl, english)
        existing = load_passage_record(passage_record_path(out_dir, index, self.state.test_mode))
        if passage_is_resumable(existing, expected_fp):
            existing.skipped = True
            return existing
        return None

    def _finalize_passage_result(
        self,
        result: PassageResult,
        *,
        batch: bool,
        usage: dict[str, int] | None = None,
    ) -> None:
        if result.error or result.skipped:
            return
        if usage:
            self.state.cache_read_tokens += usage.get("cache_read_input_tokens", 0)
            self.state.cache_creation_tokens += usage.get("cache_creation_input_tokens", 0)
        self.state.total_input_tokens += result.input_tokens
        self.state.total_output_tokens += result.output_tokens
        self.state.total_cost += token_cost(result.input_tokens, result.output_tokens, batch=batch)

    def _handle_passage_done(
        self,
        out_dir: Path,
        result: PassageResult,
        *,
        batch: bool,
        results_by_index: dict[int, PassageResult],
        failed: list[int],
        truncated: list[int],
        skipped: list[int],
    ) -> None:
        results_by_index[result.index] = result
        if result.skipped:
            skipped.append(result.index)
        elif result.error:
            failed.append(result.index)
        elif result.truncated:
            truncated.append(result.index)
        if not result.skipped and not result.error:
            if 1 <= result.index <= len(self.state.pairs):
                nah, eng = self.state.pairs[result.index - 1]
                result.pair_fingerprint = pair_fingerprint(nah, eng)
                result.uncertain_match = self.state.pair_uncertain.get(result.index, False)
        if not result.skipped:
            persist_passage_output(out_dir, result, self.state.test_mode)
        if not result.skipped:
            usage = getattr(result, "_usage", None)
            self._finalize_passage_result(result, batch=batch, usage=usage)
        self._progress_done += 1
        self.root.after(0, lambda r=result: self._update_progress(r, batch=batch))

    def _translate_worker_live(self):
        try:
            api_key = resolve_api_key()
            system_prompt = load_system_prompt()
            client = anthropic.Anthropic(api_key=api_key)
        except Exception as exc:
            self.root.after(0, lambda: self._on_run_error(str(exc)))
            return

        out_dir = output_directory(self.state.nahuatl_paths)
        results_by_index: dict[int, PassageResult] = {}
        failed: list[int] = []
        truncated: list[int] = []
        skipped: list[int] = []
        done_count = 0

        for index in self._indices_to_run():
            nahuatl, english = self.state.pairs[index - 1]
            existing = self._try_load_existing(out_dir, index)
            if existing is not None:
                self._handle_passage_done(
                    out_dir, existing, batch=False,
                    results_by_index=results_by_index, failed=failed,
                    truncated=truncated, skipped=skipped,
                )
                done_count += 1
                continue

            result = PassageResult(index=index)
            try:
                message = call_claude(client, system_prompt, build_user_message(nahuatl, english))
                apply_message_to_result(result, message)
                result._usage = message_usage_extended(message)  # type: ignore[attr-defined]
            except Exception as exc:
                result.error = str(exc)

            self._handle_passage_done(
                out_dir, result, batch=False,
                results_by_index=results_by_index, failed=failed,
                truncated=truncated, skipped=skipped,
            )
            done_count += 1

        results = [results_by_index.get(i) or PassageResult(index=i, error="Not processed") for i in range(1, len(self.state.pairs) + 1)]
        self.root.after(
            0,
            lambda: self._on_run_complete(
                results, sorted(set(failed)), sorted(set(truncated)), sorted(set(skipped)), batch=False
            ),
        )

    def _translate_worker_batch(self):
        try:
            api_key = resolve_api_key()
            system_prompt = load_system_prompt()
            client = anthropic.Anthropic(api_key=api_key)
        except Exception as exc:
            self.root.after(0, lambda: self._on_run_error(str(exc)))
            return

        out_dir = output_directory(self.state.nahuatl_paths)
        results_by_index: dict[int, PassageResult] = {}
        failed: list[int] = []
        truncated: list[int] = []
        skipped: list[int] = []
        batch_ids: list[str] = []
        pending: list[tuple[int, str, str]] = []

        for index in self._indices_to_run():
            existing = self._try_load_existing(out_dir, index)
            if existing is not None:
                self._handle_passage_done(
                    out_dir, existing, batch=True,
                    results_by_index=results_by_index, failed=failed,
                    truncated=truncated, skipped=skipped,
                )
                continue
            nahuatl, english = self.state.pairs[index - 1]
            pending.append((index, nahuatl, english))

        try:
            for offset in range(0, len(pending), BATCH_MAX_REQUESTS):
                chunk = pending[offset : offset + BATCH_MAX_REQUESTS]
                chunk_num = offset // BATCH_MAX_REQUESTS + 1
                total_chunks = (len(pending) + BATCH_MAX_REQUESTS - 1) // BATCH_MAX_REQUESTS
                self.root.after(
                    0,
                    lambda c=chunk_num, t=total_chunks, n=len(chunk): self.status_var.set(
                        f"Submitting batch {c}/{t} ({n} passages)…"
                    ),
                )
                batch_id = create_translation_batch(client, system_prompt, chunk)
                batch_ids.append(batch_id)
                self.state.batch_ids = list(batch_ids)
                names = output_filenames(test_mode=self.state.test_mode)
                (out_dir / names["batch_state"]).write_text(
                    json.dumps({"batch_ids": batch_ids, "model": resolve_translation_model()}, indent=2),
                    encoding="utf-8",
                )

                def on_batch_status(batch_info: object, c=chunk_num, tc=total_chunks):
                    done, batch_total = batch_counts_done(batch_info)
                    if batch_total:
                        self.root.after(
                            0,
                            lambda d=done, t=batch_total, cn=c, tcn=tc: self.status_var.set(
                                f"Batch {cn}/{tcn} processing… {d}/{t} (~50% off)"
                            ),
                        )

                wait_for_batch(client, batch_id, on_status=on_batch_status)
                chunk_results = collect_batch_results(client, batch_id)
                for index, _, _ in chunk:
                    result = chunk_results.get(index) or PassageResult(
                        index=index, error="No result returned for this passage"
                    )
                    self._handle_passage_done(
                        out_dir, result, batch=True,
                        results_by_index=results_by_index, failed=failed,
                        truncated=truncated, skipped=skipped,
                    )
        except Exception as exc:
            self.root.after(0, lambda: self._on_run_error(str(exc)))
            return

        results = [results_by_index.get(i) or load_passage_record(passage_record_path(out_dir, i, self.state.test_mode)) or PassageResult(index=i, error="Not processed") for i in range(1, len(self.state.pairs) + 1)]
        self.root.after(
            0,
            lambda: self._on_run_complete(
                results, sorted(set(failed)), sorted(set(truncated)), sorted(set(skipped)), batch=True
            ),
        )

    def _update_progress(self, result: PassageResult, *, batch: bool):
        self.progress.configure(value=self._progress_done)
        if result.skipped:
            detail = f"Passage {result.index} skipped (already done)"
        elif result.error:
            detail = f"Passage {result.index} failed"
        elif result.truncated:
            detail = f"Passage {result.index} TRUNCATED"
        else:
            detail = f"Passage {result.index}: {result.input_tokens}+{result.output_tokens} tok"
        mode = "batch ~50% off" if batch else "live"
        self.status_var.set(
            f"{self._progress_done}/{self._progress_total} — ${self.state.total_cost:.4f} ({mode}) — {detail}"
        )

    def _on_run_error(self, msg: str):
        self._set_running_ui(False)
        messagebox.showerror("Translation error", msg)

    def _on_run_complete(
        self,
        results: list[PassageResult],
        failed: list[int],
        truncated: list[int],
        skipped: list[int],
        *,
        batch: bool,
    ):
        self._set_running_ui(False)
        self.state.results = results
        self.state.failed = failed
        self.state.truncated = truncated
        out_dir = output_directory(self.state.nahuatl_paths)

        try:
            rebuild_failed_log(out_dir, self.state.test_mode)
            rebuild_truncated_log(out_dir, self.state.test_mode)
            write_run_summary(
                out_dir,
                mode="batch" if batch else "live",
                results=results,
                failed=failed,
                truncated=truncated,
                nahuatl_paths=self.state.nahuatl_paths,
                english_paths=self.state.english_paths,
                input_tokens=self.state.total_input_tokens,
                output_tokens=self.state.total_output_tokens,
                cost=self.state.total_cost + self._summary_alignment_cost,
                batch_ids=self.state.batch_ids if batch else None,
                test_mode=self.state.test_mode,
                skipped_count=len(skipped),
                cache_read_tokens=self.state.cache_read_tokens,
                cache_creation_tokens=self.state.cache_creation_tokens,
                ai_alignment=self.state.ai_alignment,
                alignment_input_tokens=self._summary_alignment_input,
                alignment_output_tokens=self._summary_alignment_output,
                alignment_cost=self._summary_alignment_cost,
            )
            names = output_filenames(test_mode=self.state.test_mode)
            batch_state = out_dir / names["batch_state"]
            if batch_state.is_file() and not failed:
                batch_state.unlink()
        except OSError as exc:
            messagebox.showerror("Save error", str(exc))
            return

        self.retry_btn.configure(state=tk.NORMAL if failed else tk.DISABLED)
        ok = len(results) - len(failed) - len(truncated)
        translation_cost = self.state.total_cost
        summary = (
            f"Done — {ok}/{len(results)} complete.\n"
            f"Alignment model: {ALIGNMENT_MODEL if self.state.ai_alignment else 'n/a (mechanical)'}\n"
            f"Translation model: {resolve_translation_model()}\n"
            f"Mode: {'Batch (~50% off)' if batch else 'Live'}\n"
            f"Skipped (resume): {len(skipped)}\n"
            f"Prompt cache: {self.state.cache_read_tokens} read / "
            f"{self.state.cache_creation_tokens} written\n"
            f"Alignment cost: ${self._summary_alignment_cost:.4f}\n"
            f"Translation cost: ${translation_cost:.4f}\n"
            f"Total estimated cost: ${translation_cost + self._summary_alignment_cost:.4f}\n"
            f"Saved to {out_dir}"
        )
        if truncated:
            summary += f"\n\nTruncated (max_tokens): {truncated}\nSee truncated.log"
        if failed:
            summary += f"\n\nFailed: {failed}\nUse Retry failed passages or see failed_passages.log"

        self.status_var.set(f"Complete — ${self.state.total_cost:.4f} — {out_dir.name}")
        messagebox.showinfo("Translation complete", summary)

    def _on_close(self):
        if self._running:
            if not messagebox.askokcancel(
                "Translation running",
                "A translation is still in progress. Quit anyway?",
            ):
                return
        if self._prompt_is_dirty():
            answer = messagebox.askyesnocancel(
                "Unsaved prompt",
                "The system prompt has unsaved changes.\n\nSave before quitting?",
            )
            if answer is None:
                return
            if answer and not self._save_prompt():
                return
        self.root.destroy()

    def run(self):
        self.root.mainloop()


def main():
    app = TranslatorApp()
    app.run()


if __name__ == "__main__":
    main()
