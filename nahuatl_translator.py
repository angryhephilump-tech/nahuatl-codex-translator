#!/usr/bin/env python3
"""Drag-and-drop Nahuatl → English/Spanish translation tool using Claude."""

from __future__ import annotations

import json
import os
import re
import threading
import tkinter as tk
from dataclasses import dataclass, field
from pathlib import Path
from tkinter import messagebox, scrolledtext, ttk

try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
except ImportError as exc:
    raise SystemExit("Install tkinterdnd2: pip install tkinterdnd2") from exc

import anthropic

SCRIPT_DIR = Path(__file__).resolve().parent
PROMPT_FILE = SCRIPT_DIR / "wikowi_codex_prompt_FINAL.md"
MODEL = "claude-opus-4-8"
MAX_TOKENS = 4000
INPUT_COST_PER_M = 5.0
OUTPUT_COST_PER_M = 25.0
PREVIEW_COUNT = 3

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


def load_system_prompt() -> str:
    if not PROMPT_FILE.is_file():
        raise FileNotFoundError(f"System prompt not found: {PROMPT_FILE}")
    return PROMPT_FILE.read_text(encoding="utf-8").strip()


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


def parse_response(text: str) -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for tag, pattern in TAG_RE.items():
        match = pattern.search(text)
        out[tag] = match.group(1).strip() if match else None
    return out


def token_cost(input_tokens: int, output_tokens: int) -> float:
    return (input_tokens / 1_000_000 * INPUT_COST_PER_M) + (
        output_tokens / 1_000_000 * OUTPUT_COST_PER_M
    )


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
    nahuatl_path: Path | None = None
    english_path: Path | None = None
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost: float = 0.0
    results: list[PassageResult] = field(default_factory=list)
    failed: list[int] = field(default_factory=list)


class DropZone(tk.Frame):
    def __init__(self, master, label: str, on_file, **kwargs):
        super().__init__(master, relief=tk.GROOVE, borderwidth=2, **kwargs)
        self.on_file = on_file
        self.file_path: Path | None = None

        self.label = tk.Label(self, text=label, font=("Segoe UI", 11), wraplength=220)
        self.label.pack(expand=True, fill=tk.BOTH, padx=12, pady=24)

        self.path_label = tk.Label(self, text="", font=("Segoe UI", 9), fg="#555", wraplength=220)
        self.path_label.pack(padx=8, pady=(0, 12))

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
        self.configure(bg=self.master.cget("bg"))
        self.label.configure(bg=self.master.cget("bg"))
        self.path_label.configure(bg=self.master.cget("bg"))

    def _handle_drop(self, event):
        self._reset_bg()
        raw = event.data.strip()
        if raw.startswith("{") and raw.endswith("}"):
            raw = raw[1:-1]
        path = Path(raw)
        if path.suffix.lower() not in {".txt", ".text"}:
            messagebox.showwarning("Invalid file", "Please drop a .txt file.")
            return
        self.set_file(path)

    def set_file(self, path: Path):
        self.file_path = path
        self.path_label.configure(text=path.name)
        self.on_file(path)


class TranslatorApp:
    def __init__(self):
        self.root = TkinterDnD.Tk()
        self.root.title("Nahuatl Codex Translator")
        self.root.geometry("780x640")
        self.root.minsize(680, 560)

        self.state = RunState()
        self._running = False

        self._build_ui()

    def _build_ui(self):
        pad = {"padx": 10, "pady": 6}

        top = tk.Frame(self.root)
        top.pack(fill=tk.X, **pad)

        zones = tk.Frame(top)
        zones.pack(fill=tk.X)

        self.nahuatl_zone = DropZone(zones, "Drop Nahuatl file here", self._on_nahuatl)
        self.nahuatl_zone.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=(0, 5))

        self.english_zone = DropZone(zones, "Drop English reference file here", self._on_english)
        self.english_zone.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=(5, 0))

        btn_row = tk.Frame(self.root)
        btn_row.pack(fill=tk.X, **pad)

        self.preview_btn = ttk.Button(btn_row, text="Split && Preview", command=self.split_and_preview)
        self.preview_btn.pack(side=tk.LEFT)

        self.run_btn = ttk.Button(btn_row, text="Run Translation", command=self.run_translation, state=tk.DISABLED)
        self.run_btn.pack(side=tk.LEFT, padx=8)

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

        self.status_var = tk.StringVar(value="Drop both files, then Split & Preview.")
        tk.Label(progress_frame, textvariable=self.status_var, anchor=tk.W).pack(fill=tk.X)

    def _on_nahuatl(self, path: Path):
        self.state.nahuatl_path = path
        self._maybe_enable_run()

    def _on_english(self, path: Path):
        self.state.english_path = path
        self._maybe_enable_run()

    def _maybe_enable_run(self):
        if self.state.nahuatl_path and self.state.english_path:
            self.preview_btn.configure(state=tk.NORMAL)
        else:
            self.run_btn.configure(state=tk.DISABLED)

    def split_and_preview(self):
        if not self.state.nahuatl_path or not self.state.english_path:
            messagebox.showinfo("Missing files", "Drop both a Nahuatl and an English file first.")
            return

        try:
            nahuatl_text = self.state.nahuatl_path.read_text(encoding="utf-8")
            english_text = self.state.english_path.read_text(encoding="utf-8")
        except OSError as exc:
            messagebox.showerror("Read error", str(exc))
            return

        nahuatl_passages = split_passages(nahuatl_text)
        english_passages = split_passages(english_text)

        if not nahuatl_passages or not english_passages:
            messagebox.showerror("Split error", "One or both files produced no passages.")
            return

        if len(nahuatl_passages) != len(english_passages):
            messagebox.showwarning(
                "Count mismatch",
                f"Nahuatl: {len(nahuatl_passages)} passages\n"
                f"English: {len(english_passages)} passages\n\n"
                "Check alignment before running. Preview shows the first pairs anyway.",
            )

        count = min(len(nahuatl_passages), len(english_passages))
        self.state.pairs = list(zip(nahuatl_passages[:count], english_passages[:count]))

        lines: list[str] = []
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
        self.status_var.set(f"Split into {len(self.state.pairs)} pairs. Review preview, then Run Translation.")

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

        self._running = True
        self.preview_btn.configure(state=tk.DISABLED)
        self.run_btn.configure(state=tk.DISABLED)
        self.state.total_input_tokens = 0
        self.state.total_output_tokens = 0
        self.state.total_cost = 0.0
        self.state.results = []
        self.state.failed = []
        self.progress.configure(maximum=len(self.state.pairs), value=0)

        thread = threading.Thread(target=self._translate_worker, daemon=True)
        thread.start()

    def _translate_worker(self):
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
                message = client.messages.create(
                    model=MODEL,
                    max_tokens=MAX_TOKENS,
                    system=system_prompt,
                    messages=[{"role": "user", "content": build_user_message(nahuatl, english)}],
                )
                response_text = message.content[0].text if message.content else ""
                parsed = parse_response(response_text)

                if not parsed.get("english") or not parsed.get("spanish"):
                    raise ValueError("Response missing <english> or <spanish> tags")

                result.english = parsed["english"] or ""
                result.spanish = parsed["spanish"] or ""
                result.flags = parsed.get("flags")
                result.input_tokens = message.usage.input_tokens
                result.output_tokens = message.usage.output_tokens

                self.state.total_input_tokens += result.input_tokens
                self.state.total_output_tokens += result.output_tokens
                passage_cost = token_cost(result.input_tokens, result.output_tokens)
                self.state.total_cost += passage_cost

                print(
                    f"Passage {i}: in={result.input_tokens} out={result.output_tokens} "
                    f"cost=${passage_cost:.4f} running_total=${self.state.total_cost:.4f}"
                )
            except Exception as exc:
                result.error = str(exc)
                failed.append(i)
                print(f"Passage {i} FAILED: {exc}")

            results.append(result)
            done = i
            self.root.after(0, lambda d=done, r=result: self._update_progress(d, r))

        self.root.after(0, lambda: self._on_run_complete(results, failed))

    def _update_progress(self, done: int, result: PassageResult):
        self.progress.configure(value=done)
        if result.error:
            detail = f"Passage {result.index} failed"
        else:
            detail = f"Passage {result.index}: {result.input_tokens}+{result.output_tokens} tok"
        self.status_var.set(
            f"{done}/{len(self.state.pairs)} done — ${self.state.total_cost:.4f} est. — {detail}"
        )

    def _on_run_error(self, msg: str):
        self._running = False
        self.preview_btn.configure(state=tk.NORMAL)
        self.run_btn.configure(state=tk.NORMAL)
        messagebox.showerror("Translation error", msg)

    def _on_run_complete(self, results: list[PassageResult], failed: list[int]):
        self._running = False
        self.state.results = results
        self.state.failed = failed

        out_dir = self.state.nahuatl_path.parent if self.state.nahuatl_path else SCRIPT_DIR

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
        except OSError as exc:
            messagebox.showerror("Save error", str(exc))
            self.preview_btn.configure(state=tk.NORMAL)
            self.run_btn.configure(state=tk.NORMAL)
            return

        self.preview_btn.configure(state=tk.NORMAL)
        self.run_btn.configure(state=tk.NORMAL)

        summary = (
            f"Done — {len(results) - len(failed)}/{len(results)} succeeded.\n"
            f"Tokens: {self.state.total_input_tokens} in / {self.state.total_output_tokens} out\n"
            f"Estimated cost: ${self.state.total_cost:.4f}\n"
            f"Saved to {out_dir}"
        )
        if failed:
            summary += f"\n\nFailed passages: {failed}"
            print(f"Failed passages: {failed}")

        self.status_var.set(
            f"Complete — ${self.state.total_cost:.4f} — saved to {out_dir.name}"
        )
        messagebox.showinfo("Translation complete", summary)

    def run(self):
        self.root.mainloop()


def main():
    app = TranslatorApp()
    app.run()


if __name__ == "__main__":
    main()
