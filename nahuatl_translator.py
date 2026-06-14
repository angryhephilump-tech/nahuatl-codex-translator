#!/usr/bin/env python3
"""Drag-and-drop Nahuatl → English/Spanish translation tool using Claude."""

from __future__ import annotations

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
DEFAULT_MODEL = "claude-opus-4-8"
MAX_TOKENS = 4000
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


@dataclass
class RunState:
    pairs: list[tuple[str, str]] = field(default_factory=list)
    nahuatl_paths: list[Path] = field(default_factory=list)
    english_paths: list[Path] = field(default_factory=list)
    nahuatl_text: str = ""
    english_text: str = ""
    nahuatl_passage_count: int = 0
    english_passage_count: int = 0
    aligned: bool = False
    total_input_tokens: int = 0
    total_output_tokens: int = 0
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
        data.get("api_key") or data.get("anthropic_api_key") or data.get("deepseek_api_key") or ""
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


def resolve_model() -> str:
    return (os.environ.get("CLAUDE_MODEL") or DEFAULT_MODEL).strip() or DEFAULT_MODEL


def load_system_prompt() -> str:
    if not PROMPT_FILE.is_file():
        raise FileNotFoundError(f"System prompt not found: {PROMPT_FILE}")
    return PROMPT_FILE.read_text(encoding="utf-8").strip()


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


def split_by_word_count(text: str, words_per_passage: int) -> list[str]:
    text = text.strip()
    if not text:
        return []
    words = re.findall(r"\S+", text)
    if not words:
        return []
    n = max(1, words_per_passage)
    return [" ".join(words[i : i + n]) for i in range(0, len(words), n)]


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
        if chunks:
            return chunks
        return split_by_paragraphs(text)
    if method == SPLIT_WORDS:
        return split_by_word_count(text, words_per_passage)
    return split_by_paragraphs(text)


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
        )
    except (json.JSONDecodeError, OSError, KeyError, TypeError, ValueError):
        return None


def passage_is_resumable(result: PassageResult | None) -> bool:
    if result is None or result.error or result.truncated:
        return False
    return bool(result.english.strip() and result.spanish.strip())


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
            body = getattr(rec, field_name, "") + "\n[TRUNCATED — output hit max_tokens]"
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
    rebuild_aggregate_file(out_dir, test_mode=test_mode, field_name="english", filename_key="english")
    rebuild_aggregate_file(out_dir, test_mode=test_mode, field_name="spanish", filename_key="spanish")
    rebuild_flags_file(out_dir, test_mode)


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
    usage = message.usage if hasattr(message, "usage") else message.get("usage", {})
    if isinstance(usage, dict):
        return int(usage.get("input_tokens", 0)), int(usage.get("output_tokens", 0))
    return int(getattr(usage, "input_tokens", 0)), int(getattr(usage, "output_tokens", 0))


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


def batch_custom_id(index: int) -> str:
    return f"passage-{index:05d}"


def parse_passage_index(custom_id: str) -> int | None:
    match = re.fullmatch(r"passage-(\d+)", custom_id or "")
    return int(match.group(1)) if match else None


def build_message_params(system_prompt: str, nahuatl: str, english: str) -> dict:
    return {
        "model": resolve_model(),
        "max_tokens": MAX_TOKENS,
        "system": system_prompt,
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
    apply_parsed_response(result, message_text(message))
    in_tok, out_tok = message_usage(message)
    result.input_tokens = in_tok
    result.output_tokens = out_tok
    if message_stop_reason(message) == "max_tokens":
        result.truncated = True


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
                apply_message_to_result(result, message)
            except Exception as exc:
                result.error = str(exc)
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
                model=resolve_model(),
                max_tokens=MAX_TOKENS,
                system=system_prompt,
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


def format_pair_preview(index: int, nahuatl: str, english: str) -> list[str]:
    return [
        f"=== Pair {index} ===",
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
) -> None:
    model = resolve_model()
    names = output_filenames(test_mode=test_mode)
    stats = {
        "model": model,
        "mode": mode,
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
        "estimated_cost_usd": round(cost, 4),
        "batch_discount_applied": mode == "batch",
        "batch_ids": batch_ids or [],
        "nahuatl_files": [str(p) for p in nahuatl_paths],
        "english_files": [str(p) for p in english_paths],
    }
    (out_dir / names["summary_json"]).write_text(json.dumps(stats, indent=2), encoding="utf-8")
    lines = [
        "Nahuatl Codex Translator — run summary",
        f"Model: {model}",
        f"Mode: {'Batch (~50% off)' if mode == 'batch' else 'Live (full price)'}",
        f"Passages: {stats['passages_succeeded']}/{stats['passages_total']} OK",
        f"Skipped (resume): {skipped_count}",
        f"Tokens: {input_tokens} in / {output_tokens} out",
        f"Estimated cost: ${cost:.4f}",
    ]
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
        self.root.title(f"Nahuatl Codex Translator — {resolve_model()}")
        self.root.geometry("860x820")
        self.root.minsize(760, 680)

        self.state = RunState()
        self._running = False
        self._progress_done = 0
        self._progress_total = 0
        self._mode_radios: list[ttk.Radiobutton] = []
        self._split_widgets: list[tk.Widget] = []

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self):
        pad = {"padx": 10, "pady": 4}

        top = tk.Frame(self.root)
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

        split_frame = ttk.LabelFrame(self.root, text="Split method")
        split_frame.pack(fill=tk.X, padx=10, pady=4)

        self.split_method_var = tk.StringVar(value=SPLIT_CHAPTER)
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

        btn_row = tk.Frame(self.root)
        btn_row.pack(fill=tk.X, **pad)

        self.preview_btn = ttk.Button(btn_row, text="Split && Preview", command=self.split_and_preview)
        self.preview_btn.pack(side=tk.LEFT)

        self.run_btn = ttk.Button(btn_row, text="Run Translation", command=self.run_translation, state=tk.DISABLED)
        self.run_btn.pack(side=tk.LEFT, padx=6)

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
            self.root,
            text="Alignment: run Split & Preview",
            font=("Segoe UI", 11, "bold"),
            fg="#666",
        )
        self.alignment_label.pack(anchor=tk.W, padx=12, pady=2)

        model_label = tk.Label(
            self.root,
            text=f"Model: {resolve_model()}  ·  max {MAX_TOKENS} tokens/passage",
            font=("Segoe UI", 9),
            fg="#666",
        )
        model_label.pack(anchor=tk.W, padx=12)

        preview_frame = ttk.LabelFrame(self.root, text="Alignment preview")
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
        self.preview_text.configure(state=tk.DISABLED)

        progress_frame = tk.Frame(self.root)
        progress_frame.pack(fill=tk.X, **pad)

        self.progress = ttk.Progressbar(progress_frame, mode="determinate")
        self.progress.pack(fill=tk.X, pady=4)

        self.status_var = tk.StringVar(value="Drop file(s) on both sides, then Split & Preview.")
        tk.Label(progress_frame, textvariable=self.status_var, anchor=tk.W, wraplength=820).pack(fill=tk.X)

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
        self.run_btn.configure(state=tk.DISABLED)
        self.retry_btn.configure(state=tk.DISABLED)
        self._update_alignment_label(0, 0, aligned=False)
        if self.state.nahuatl_paths and self.state.english_paths:
            self._refresh_split_counts()

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

    def _split_both(self, nahuatl_text: str, english_text: str) -> tuple[list[str], list[str]]:
        method = self._split_method_key()
        kwargs = {
            "method": method,
            "chapter_pattern": self.chapter_regex_var.get().strip() or DEFAULT_CHAPTER_REGEX,
            "words_per_passage": self._words_per_passage(),
        }
        nahuatl_passages = split_passages(nahuatl_text, **kwargs)
        english_passages = split_passages(english_text, **kwargs)
        return nahuatl_passages, english_passages

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
            nah, eng = self._split_both(nahuatl_text, english_text)
        except ValueError as exc:
            self.split_count_var.set(f"Split error: {exc}")
            return
        large = sorted(set(self._check_large_passages(nah) + self._check_large_passages(eng)))
        self.split_count_var.set(f"Passage counts: Nahuatl {len(nah)} | English {len(eng)}")
        if large:
            shown = large[:8]
            extra = f" (+{len(large) - 8} more)" if len(large) > 8 else ""
            self.large_passage_var.set(
                f"Warning: passage(s) {shown}{extra} exceed {LARGE_PASSAGE_WORDS} words — may truncate."
            )
        else:
            self.large_passage_var.set("")

    def _update_alignment_label(self, nah_count: int, eng_count: int, *, aligned: bool):
        if nah_count == 0 and eng_count == 0:
            self.alignment_label.configure(
                text="Alignment: run Split & Preview", fg="#666"
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
        for i in pair_indices:
            if i < 1 or i > len(source_pairs):
                lines.append(f"Pair {i}: out of range (1–{len(source_pairs)})\n")
                continue
            nah, eng = source_pairs[i - 1]
            lines.extend(format_pair_preview(i, nah, eng))
        self.preview_text.configure(state=tk.NORMAL)
        self.preview_text.delete("1.0", tk.END)
        self.preview_text.insert(tk.END, "\n".join(lines))
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

        try:
            nahuatl_passages, english_passages = self._split_both(nahuatl_text, english_text)
        except ValueError as exc:
            messagebox.showerror("Split error", str(exc))
            return

        if not nahuatl_passages or not english_passages:
            messagebox.showerror("Split error", "One or both sides produced no passages.")
            return

        self.state.nahuatl_passage_count = len(nahuatl_passages)
        self.state.english_passage_count = len(english_passages)
        aligned = len(nahuatl_passages) == len(english_passages)
        self.state.aligned = aligned

        self._update_alignment_label(len(nahuatl_passages), len(english_passages), aligned=aligned)
        self.split_count_var.set(
            f"Passage counts: Nahuatl {len(nahuatl_passages)} | English {len(english_passages)}"
        )

        large = sorted(
            set(self._check_large_passages(nahuatl_passages) + self._check_large_passages(english_passages))
        )
        if large:
            self.large_passage_var.set(
                f"Warning: passage(s) {large[:8]} exceed {LARGE_PASSAGE_WORDS} words — may truncate."
            )
        else:
            self.large_passage_var.set("")

        if not aligned:
            self.state.pairs = []
            self.run_btn.configure(state=tk.DISABLED)
            self.retry_btn.configure(state=tk.DISABLED)
            preview_cap = min(PREVIEW_COUNT, len(nahuatl_passages), len(english_passages))
            mismatch_pairs = list(
                zip(nahuatl_passages[:preview_cap], english_passages[:preview_cap])
            )
            self._render_preview(list(range(1, preview_cap + 1)), pairs=mismatch_pairs)
            messagebox.showerror(
                "Alignment mismatch",
                f"Nahuatl: {len(nahuatl_passages)} passages\n"
                f"English: {len(english_passages)} passages\n\n"
                "Counts must match exactly. Fix split settings or source files.\n"
                "Run is blocked until aligned.",
            )
            return

        self.state.pairs = list(zip(nahuatl_passages, english_passages))
        preview_nums = list(range(1, min(PREVIEW_COUNT, len(self.state.pairs)) + 1))
        self._render_preview(preview_nums)

        self.run_btn.configure(state=tk.NORMAL)
        self.retry_btn.configure(state=tk.DISABLED)
        test_label = f" [TEST ~{TEST_MODE_WORD_LIMIT} words]" if self.state.test_mode else ""
        self.status_var.set(
            f"Aligned — {len(self.state.pairs)} pairs{test_label}. Review preview, then Run Translation."
        )

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

    def run_translation(self, *, retry_only: bool = False):
        if self._running:
            return
        if not self.state.pairs:
            messagebox.showinfo("Nothing to run", "Split & Preview first.")
            return
        if not retry_only and not self.state.aligned:
            messagebox.showerror("Not aligned", "Passage counts must match before running.")
            return

        try:
            resolve_api_key()
            load_system_prompt()
        except (ValueError, FileNotFoundError) as exc:
            messagebox.showerror("Setup error", str(exc))
            return

        self._set_running_ui(True)
        if not retry_only:
            self.state.retry_indices = []
        self.state.total_input_tokens = 0
        self.state.total_output_tokens = 0
        self.state.total_cost = 0.0
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
        can_run = bool(self.state.pairs and self.state.aligned and not running)
        self.run_btn.configure(state=tk.NORMAL if can_run else tk.DISABLED)
        can_retry = bool(self.state.failed and not running)
        self.retry_btn.configure(state=tk.NORMAL if can_retry else tk.DISABLED)
        for rb in self._mode_radios:
            rb.configure(state=state)
        if self._test_mode_cb is not None:
            self._test_mode_cb.configure(state=state)
        for w in self._split_widgets:
            w.configure(state=state)
        cursor = "watch" if running else ""
        self.nahuatl_zone.configure(cursor=cursor)
        self.english_zone.configure(cursor=cursor)

    def _indices_to_run(self) -> list[int]:
        if self.state.retry_indices:
            return sorted(self.state.retry_indices)
        return list(range(1, len(self.state.pairs) + 1))

    def _try_load_existing(self, out_dir: Path, index: int) -> PassageResult | None:
        existing = load_passage_record(passage_record_path(out_dir, index, self.state.test_mode))
        if passage_is_resumable(existing):
            existing.skipped = True
            return existing
        return None

    def _finalize_passage_result(self, result: PassageResult, *, batch: bool) -> None:
        if result.error or result.skipped:
            return
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
        if not result.skipped:
            persist_passage_output(out_dir, result, self.state.test_mode)
        if not result.skipped:
            self._finalize_passage_result(result, batch=batch)
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
                    json.dumps({"batch_ids": batch_ids, "model": resolve_model()}, indent=2),
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
                cost=self.state.total_cost,
                batch_ids=self.state.batch_ids if batch else None,
                test_mode=self.state.test_mode,
                skipped_count=len(skipped),
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
        summary = (
            f"Done — {ok}/{len(results)} complete.\n"
            f"Model: {resolve_model()}\n"
            f"Mode: {'Batch (~50% off)' if batch else 'Live'}\n"
            f"Skipped (resume): {len(skipped)}\n"
            f"Estimated cost: ${self.state.total_cost:.4f}\n"
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
        self.root.destroy()

    def run(self):
        self.root.mainloop()


def main():
    app = TranslatorApp()
    app.run()


if __name__ == "__main__":
    main()
