#!/usr/bin/env python3
"""Drag-and-drop Nahuatl → English/Spanish translation tool using Claude."""

from __future__ import annotations

import json
import os
import re
import threading
import time
import tkinter as tk
from dataclasses import dataclass, field
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
API_MAX_RETRIES = 4
BATCH_POLL_SEC = 15
BATCH_MAX_REQUESTS = 5000
TEXT_SUFFIXES = {".txt", ".text", ".md"}
ENCODINGS = ("utf-8-sig", "utf-8", "cp1252", "latin-1")

CHAPTER_HEADING_RE = re.compile(
    r"^(?:"
    r"(?:Chapter|CHAPTER|Cap[ií]tulo|CAP[IÍ]TULO|Book|BOOK|Libro|LIBRO)\s+[\w\dIVXLCivxlc\-]+.*"
    r"|#{1,3}\s+\S.+"
    r"|\*{2,}.+\*{2,}"
    r"|[IVXLC]+\.\s+\S"
    r"|\d+\.\s+[A-Z].+"
    r")\s*$",
    re.MULTILINE,
)

TAG_RE = {
    "english": re.compile(r"<english>(.*?)</english>", re.DOTALL | re.IGNORECASE),
    "spanish": re.compile(r"<spanish>(.*?)</spanish>", re.DOTALL | re.IGNORECASE),
    "flags": re.compile(r"<flags>(.*?)</flags>", re.DOTALL | re.IGNORECASE),
}


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
    """Parse one or many file paths from a drag-and-drop event (Windows-safe)."""
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
    """Read multiple files in sorted name order, joined with a blank line."""
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


def split_passages(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []

    headings = list(CHAPTER_HEADING_RE.finditer(text))
    if headings:
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

    return [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]


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


def create_translation_batch(
    client: anthropic.Anthropic,
    system_prompt: str,
    pairs: list[tuple[str, str]],
    start_index: int = 1,
) -> str:
    requests = [
        {
            "custom_id": batch_custom_id(i),
            "params": build_message_params(system_prompt, nahuatl, english),
        }
        for i, (nahuatl, english) in enumerate(pairs, start=start_index)
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
        processing = int(counts.get("processing", 0))
        done = sum(int(counts.get(k, 0)) for k in ("succeeded", "errored", "canceled", "expired"))
    else:
        processing = int(getattr(counts, "processing", 0))
        done = sum(
            int(getattr(counts, k, 0))
            for k in ("succeeded", "errored", "canceled", "expired")
        )
    return done, done + processing


def collect_batch_results(
    client: anthropic.Anthropic,
    batch_id: str,
) -> dict[int, PassageResult]:
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
                apply_parsed_response(result, message_text(message))
                in_tok, out_tok = message_usage(message)
                result.input_tokens = in_tok
                result.output_tokens = out_tok
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


def write_run_summary(
    out_dir: Path,
    *,
    mode: str,
    results: list[PassageResult],
    failed: list[int],
    nahuatl_paths: list[Path],
    english_paths: list[Path],
    input_tokens: int,
    output_tokens: int,
    cost: float,
    batch_ids: list[str] | None = None,
) -> None:
    model = resolve_model()
    stats = {
        "model": model,
        "mode": mode,
        "passages_total": len(results),
        "passages_succeeded": len(results) - len(failed),
        "passages_failed": len(failed),
        "failed_passage_numbers": failed,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "estimated_cost_usd": round(cost, 4),
        "batch_discount_applied": mode == "batch",
        "batch_ids": batch_ids or [],
        "nahuatl_files": [str(p) for p in nahuatl_paths],
        "english_files": [str(p) for p in english_paths],
        "outputs": {
            "english": str(out_dir / "english_all.txt"),
            "spanish": str(out_dir / "spanish_all.txt"),
            "flags": str(out_dir / "flags_all.txt") if any(r.flags for r in results if not r.error) else None,
        },
    }
    (out_dir / "run_summary.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    lines = [
        "Nahuatl Codex Translator — run summary",
        f"Model: {model}",
        f"Mode: {'Batch (~50% off)' if mode == 'batch' else 'Live (full price)'}",
        f"Passages: {stats['passages_succeeded']}/{stats['passages_total']} succeeded",
        f"Tokens: {input_tokens} in / {output_tokens} out",
        f"Estimated cost: ${cost:.4f}",
    ]
    if batch_ids:
        lines.append(f"Batch IDs: {', '.join(batch_ids)}")
    if failed:
        lines.append(f"Failed passages: {failed}")
    (out_dir / "run_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


@dataclass
class PassageResult:
    index: int
    english: str = ""
    spanish: str = ""
    flags: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    error: str | None = None


@dataclass
class RunState:
    pairs: list[tuple[str, str]] = field(default_factory=list)
    nahuatl_paths: list[Path] = field(default_factory=list)
    english_paths: list[Path] = field(default_factory=list)
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost: float = 0.0
    results: list[PassageResult] = field(default_factory=list)
    failed: list[int] = field(default_factory=list)
    batch_ids: list[str] = field(default_factory=list)


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
        self.root.geometry("820x700")
        self.root.minsize(720, 580)

        self.state = RunState()
        self._running = False
        self._mode_radios: list[ttk.Radiobutton] = []

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self):
        pad = {"padx": 10, "pady": 6}

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

        btn_row = tk.Frame(self.root)
        btn_row.pack(fill=tk.X, **pad)

        self.preview_btn = ttk.Button(btn_row, text="Split && Preview", command=self.split_and_preview)
        self.preview_btn.pack(side=tk.LEFT)

        self.run_btn = ttk.Button(btn_row, text="Run Translation", command=self.run_translation, state=tk.DISABLED)
        self.run_btn.pack(side=tk.LEFT, padx=8)

        self.mode_var = tk.StringVar(value="batch")
        mode_frame = tk.Frame(btn_row)
        mode_frame.pack(side=tk.RIGHT)
        for text, value in (("Batch (50% off)", "batch"), ("Live (immediate)", "live")):
            rb = ttk.Radiobutton(mode_frame, text=text, variable=self.mode_var, value=value)
            rb.pack(side=tk.LEFT, padx=4)
            self._mode_radios.append(rb)

        model_label = tk.Label(
            self.root,
            text=f"Model: {resolve_model()}  ·  max {MAX_TOKENS} tokens/passage",
            font=("Segoe UI", 9),
            fg="#666",
        )
        model_label.pack(anchor=tk.W, padx=12, pady=(0, 2))

        preview_frame = ttk.LabelFrame(self.root, text="Preview (first 3 pairs)")
        preview_frame.pack(fill=tk.BOTH, expand=True, **pad)

        self.preview_text = scrolledtext.ScrolledText(
            preview_frame, height=12, wrap=tk.WORD, font=("Consolas", 10)
        )
        self.preview_text.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)
        self.preview_text.configure(state=tk.DISABLED)

        progress_frame = tk.Frame(self.root)
        progress_frame.pack(fill=tk.X, **pad)

        self.progress = ttk.Progressbar(progress_frame, mode="determinate")
        self.progress.pack(fill=tk.X, pady=4)

        self.status_var = tk.StringVar(value="Drop file(s) on both sides, then Split & Preview.")
        tk.Label(progress_frame, textvariable=self.status_var, anchor=tk.W, wraplength=780).pack(fill=tk.X)

    def _on_nahuatl(self, paths: list[Path]):
        self.state.nahuatl_paths = paths
        self.state.pairs = []
        self.run_btn.configure(state=tk.DISABLED)
        self._update_ready_status()

    def _on_english(self, paths: list[Path]):
        self.state.english_paths = paths
        self.state.pairs = []
        self.run_btn.configure(state=tk.DISABLED)
        self._update_ready_status()

    def _update_ready_status(self):
        n = len(self.state.nahuatl_paths)
        e = len(self.state.english_paths)
        if n and e:
            self.status_var.set(
                f"Ready — {n} Nahuatl + {e} English file(s). Click Split & Preview."
            )
        elif n or e:
            missing = "English" if n else "Nahuatl"
            self.status_var.set(f"Loaded {max(n, e)} file(s). Still need {missing} file(s).")
        else:
            self.status_var.set("Drop file(s) on both sides, then Split & Preview.")

    def split_and_preview(self):
        if not self.state.nahuatl_paths or not self.state.english_paths:
            messagebox.showinfo("Missing files", "Load at least one Nahuatl and one English file.")
            return

        try:
            nahuatl_text = read_text_files(self.state.nahuatl_paths)
            english_text = read_text_files(self.state.english_paths)
        except OSError as exc:
            messagebox.showerror("Read error", str(exc))
            return

        if not nahuatl_text.strip() or not english_text.strip():
            messagebox.showerror("Empty input", "One or both sides are empty after reading files.")
            return

        nahuatl_passages = split_passages(nahuatl_text)
        english_passages = split_passages(english_text)

        if not nahuatl_passages or not english_passages:
            messagebox.showerror("Split error", "One or both sides produced no passages.")
            return

        if len(nahuatl_passages) != len(english_passages):
            messagebox.showwarning(
                "Count mismatch",
                f"Nahuatl: {len(nahuatl_passages)} passages "
                f"(from {format_file_list(self.state.nahuatl_paths)})\n"
                f"English: {len(english_passages)} passages "
                f"(from {format_file_list(self.state.english_paths)})\n\n"
                "Files are merged in alphabetical order before splitting.\n"
                "Check alignment before running. Preview shows the first pairs anyway.",
            )

        count = min(len(nahuatl_passages), len(english_passages))
        self.state.pairs = list(zip(nahuatl_passages[:count], english_passages[:count]))

        source_note = (
            f"Sources: Nahuatl [{format_file_list(self.state.nahuatl_paths)}] | "
            f"English [{format_file_list(self.state.english_paths)}]\n\n"
        )
        lines: list[str] = [source_note]
        for i, (nah, eng) in enumerate(self.state.pairs[:PREVIEW_COUNT], start=1):
            lines.append(f"=== Pair {i} ===")
            lines.append("[NAHUATL]")
            lines.append(nah[:800] + ("…" if len(nah) > 800 else ""))
            lines.append("")
            lines.append("[ENGLISH REF]")
            lines.append(eng[:800] + ("…" if len(eng) > 800 else ""))
            lines.append("")

        self.preview_text.configure(state=tk.NORMAL)
        self.preview_text.delete("1.0", tk.END)
        self.preview_text.insert(tk.END, "\n".join(lines))
        self.preview_text.configure(state=tk.DISABLED)

        self.run_btn.configure(state=tk.NORMAL if self.state.pairs else tk.DISABLED)
        self.status_var.set(
            f"Split into {len(self.state.pairs)} pairs from "
            f"{len(self.state.nahuatl_paths)} + {len(self.state.english_paths)} file(s). "
            "Review preview, then Run Translation."
        )

    def run_translation(self):
        if self._running:
            return
        if not self.state.pairs:
            messagebox.showinfo("Nothing to run", "Split & Preview first.")
            return

        try:
            resolve_api_key()
            load_system_prompt()
        except (ValueError, FileNotFoundError) as exc:
            messagebox.showerror("Setup error", str(exc))
            return

        self._set_running_ui(True)
        self.state.total_input_tokens = 0
        self.state.total_output_tokens = 0
        self.state.total_cost = 0.0
        self.state.results = []
        self.state.failed = []
        self.state.batch_ids = []
        self.progress.configure(maximum=len(self.state.pairs), value=0)

        thread = threading.Thread(
            target=self._translate_worker_batch if self.mode_var.get() == "batch" else self._translate_worker_live,
            daemon=True,
        )
        thread.start()

    def _set_running_ui(self, running: bool) -> None:
        self._running = running
        state = tk.DISABLED if running else tk.NORMAL
        self.preview_btn.configure(state=state)
        self.run_btn.configure(state=tk.DISABLED if running else (tk.NORMAL if self.state.pairs else tk.DISABLED))
        for rb in self._mode_radios:
            rb.configure(state=state)
        cursor = "watch" if running else ""
        self.nahuatl_zone.configure(cursor=cursor)
        self.english_zone.configure(cursor=cursor)

    def _finalize_passage_result(
        self,
        result: PassageResult,
        *,
        batch: bool,
    ) -> None:
        if result.error:
            return
        self.state.total_input_tokens += result.input_tokens
        self.state.total_output_tokens += result.output_tokens
        passage_cost = token_cost(result.input_tokens, result.output_tokens, batch=batch)
        self.state.total_cost += passage_cost
        print(
            f"Passage {result.index}: in={result.input_tokens} out={result.output_tokens} "
            f"cost=${passage_cost:.4f} running_total=${self.state.total_cost:.4f}"
            + (" (batch rate)" if batch else "")
        )

    def _translate_worker_live(self):
        try:
            api_key = resolve_api_key()
            system_prompt = load_system_prompt()
            client = anthropic.Anthropic(api_key=api_key)
        except Exception as exc:
            self.root.after(0, lambda: self._on_run_error(str(exc)))
            return

        results: list[PassageResult] = []
        failed: list[int] = []

        for i, (nahuatl, english) in enumerate(self.state.pairs, start=1):
            result = PassageResult(index=i)
            try:
                message = call_claude(
                    client,
                    system_prompt,
                    build_user_message(nahuatl, english),
                )
                response_text = message_text(message)
                apply_parsed_response(result, response_text)
                result.input_tokens = message.usage.input_tokens
                result.output_tokens = message.usage.output_tokens
                self._finalize_passage_result(result, batch=False)
            except Exception as exc:
                result.error = str(exc)
                failed.append(i)
                print(f"Passage {i} FAILED: {exc}")

            results.append(result)
            done = i
            self.root.after(0, lambda d=done, r=result: self._update_progress(d, r, batch=False))

        self.root.after(0, lambda: self._on_run_complete(results, failed, batch=False))

    def _translate_worker_batch(self):
        try:
            api_key = resolve_api_key()
            system_prompt = load_system_prompt()
            client = anthropic.Anthropic(api_key=api_key)
        except Exception as exc:
            self.root.after(0, lambda: self._on_run_error(str(exc)))
            return

        pairs = self.state.pairs
        total = len(pairs)
        results_by_index: dict[int, PassageResult] = {
            i: PassageResult(index=i) for i in range(1, total + 1)
        }
        failed: list[int] = []
        batch_ids: list[str] = []

        try:
            chunks: list[tuple[int, list[tuple[str, str]]]] = []
            for offset in range(0, total, BATCH_MAX_REQUESTS):
                start = offset + 1
                chunk_pairs = pairs[offset : offset + BATCH_MAX_REQUESTS]
                chunks.append((start, chunk_pairs))

            for chunk_num, (start_index, chunk_pairs) in enumerate(chunks, start=1):
                label = (
                    f"Submitting batch {chunk_num}/{len(chunks)} "
                    f"({len(chunk_pairs)} passages)…"
                )
                self.root.after(0, lambda msg=label: self.status_var.set(msg))

                batch_id = create_translation_batch(
                    client, system_prompt, chunk_pairs, start_index=start_index
                )
                batch_ids.append(batch_id)
                self.state.batch_ids = list(batch_ids)
                out_dir = output_directory(self.state.nahuatl_paths)
                (out_dir / "batch_state.json").write_text(
                    json.dumps({"batch_ids": batch_ids, "model": resolve_model()}, indent=2),
                    encoding="utf-8",
                )
                print(f"Batch submitted: {batch_id} ({len(chunk_pairs)} passages)")

                def on_batch_status(batch_info: object, c=chunk_num, tc=len(chunks)):
                    done, batch_total = batch_counts_done(batch_info)
                    if batch_total:
                        self.root.after(
                            0,
                            lambda d=done, t=batch_total, cn=c, tcn=tc: (
                                self.progress.configure(value=min(d, total), maximum=total),
                                self.status_var.set(
                                    f"Batch {cn}/{tcn} processing… {d}/{t} requests "
                                    "(~50% off, usually under 1 hour)"
                                ),
                            ),
                        )

                wait_for_batch(client, batch_id, on_status=on_batch_status)
                chunk_results = collect_batch_results(client, batch_id)

                for idx in range(start_index, start_index + len(chunk_pairs)):
                    result = chunk_results.get(idx) or PassageResult(
                        index=idx, error="No result returned for this passage"
                    )
                    results_by_index[idx] = result
                    if result.error:
                        failed.append(idx)
                        print(f"Passage {idx} FAILED: {result.error}")
                    else:
                        self._finalize_passage_result(result, batch=True)
                    done = idx
                    self.root.after(
                        0,
                        lambda d=done, r=result: self._update_progress(d, r, batch=True),
                    )
        except Exception as exc:
            self.root.after(0, lambda: self._on_run_error(str(exc)))
            return

        results = [results_by_index[i] for i in range(1, total + 1)]
        self.root.after(
            0, lambda: self._on_run_complete(results, sorted(set(failed)), batch=True)
        )

    def _update_progress(self, done: int, result: PassageResult, *, batch: bool):
        self.progress.configure(value=done)
        if result.error:
            detail = f"Passage {result.index} failed"
        else:
            detail = f"Passage {result.index}: {result.input_tokens}+{result.output_tokens} tok"
        mode = "batch ~50% off" if batch else "live"
        self.status_var.set(
            f"{done}/{len(self.state.pairs)} done — ${self.state.total_cost:.4f} est. ({mode}) — {detail}"
        )

    def _on_run_error(self, msg: str):
        self._set_running_ui(False)
        messagebox.showerror("Translation error", msg)

    def _on_run_complete(self, results: list[PassageResult], failed: list[int], *, batch: bool):
        self._set_running_ui(False)
        self.state.results = results
        self.state.failed = failed

        out_dir = output_directory(self.state.nahuatl_paths)

        english_lines: list[str] = []
        spanish_lines: list[str] = []
        flag_blocks: list[str] = []

        for r in results:
            if r.error:
                english_lines.append(f"[PASSAGE {r.index} — FAILED: {r.error}]")
                spanish_lines.append(f"[PASSAGE {r.index} — FAILED: {r.error}]")
            else:
                english_lines.append(r.english)
                spanish_lines.append(r.spanish)
                if r.flags:
                    flag_blocks.append(f"=== Passage {r.index} ===\n{r.flags}")

        try:
            (out_dir / "english_all.txt").write_text(
                "\n\n".join(english_lines) + "\n", encoding="utf-8"
            )
            (out_dir / "spanish_all.txt").write_text(
                "\n\n".join(spanish_lines) + "\n", encoding="utf-8"
            )
            if flag_blocks:
                (out_dir / "flags_all.txt").write_text(
                    "\n\n".join(flag_blocks) + "\n", encoding="utf-8"
                )
            if failed:
                fail_log = out_dir / "failed_passages.log"
                fail_lines = [f"Passage {i}: {results[i - 1].error}" for i in failed]
                fail_log.write_text("\n".join(fail_lines) + "\n", encoding="utf-8")
            write_run_summary(
                out_dir,
                mode="batch" if batch else "live",
                results=results,
                failed=failed,
                nahuatl_paths=self.state.nahuatl_paths,
                english_paths=self.state.english_paths,
                input_tokens=self.state.total_input_tokens,
                output_tokens=self.state.total_output_tokens,
                cost=self.state.total_cost,
                batch_ids=self.state.batch_ids if batch else None,
            )
            batch_state = out_dir / "batch_state.json"
            if batch_state.is_file() and not failed:
                batch_state.unlink()
        except OSError as exc:
            messagebox.showerror("Save error", str(exc))
            self._set_running_ui(False)
            return

        rate_note = " (batch rate, ~50% off)" if batch else ""
        summary = (
            f"Done — {len(results) - len(failed)}/{len(results)} succeeded.\n"
            f"Model: {resolve_model()}\n"
            f"Mode: {'Batch (~50% off)' if batch else 'Live (full price)'}\n"
            f"Tokens: {self.state.total_input_tokens} in / {self.state.total_output_tokens} out\n"
            f"Estimated cost: ${self.state.total_cost:.4f}{rate_note}\n"
            f"Saved to {out_dir}\n"
            f"(run_summary.json + run_summary.txt)"
        )
        if failed:
            summary += f"\n\nFailed passages: {failed}\nSee failed_passages.log in output folder."
            print(f"Failed passages: {failed}")

        self.status_var.set(
            f"Complete — ${self.state.total_cost:.4f} — saved to {out_dir.name}"
        )
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
