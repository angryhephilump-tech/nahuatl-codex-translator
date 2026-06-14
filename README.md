# Nahuatl Codex Translator

Simple desktop tool for translating Nahuatl source passages with English reference alignment, using Claude Opus via the Anthropic API.

## Setup

```bash
pip install -r requirements.txt
```

### API key

The tool reads your Anthropic key from (in order):

1. `ANTHROPIC_API_KEY` or `CLAUDE_API_KEY` environment variable
2. A `.env` file next to the script
3. **PDF Transcribe** saved settings (`%LOCALAPPDATA%\PDF Transcribe\settings.json`) — same key as the transcriber app

Never commit API keys to git.

## Usage

```bash
python nahuatl_translator.py
```

1. Drop Nahuatl and English `.txt` files (multiple OK; merged alphabetically by filename).
2. Choose a **split method** and tune settings — passage counts update live.
3. **Split & Preview** — check the green/red alignment banner and eyeball the first ~10 words of each side.
4. Use **Preview pair #** to inspect any passage number (misalignment often starts mid-file).
5. **Run Translation** is blocked until Nahuatl and English passage counts match exactly.
6. Optionally: **Test mode** (~300 words), **Batch (50% off)**, or **Live**.
7. After failures: **Retry failed passages** (does not re-run successful ones).

## Alignment safety

- Green: `Nahuatl: N | English: N — ALIGNED` → Run enabled  
- Red: count mismatch → Run blocked (no silent `min(count)` pairing)

## Split methods

- **Chapter headings** — editable regex (defaults to Chapter/Book/markdown heading patterns; falls back to paragraphs if no headings found)
- **Paragraph breaks** (`\n\n`)
- **Every N words** — set N (default 400); warns if any passage exceeds 3000 words

## Resume & retry

- Each passage saves to `passages/passage_NNNNN.json` as it completes; aggregate `.txt` files rebuild incrementally.
- Re-running skips passages already on disk (unless failed or truncated).
- **Retry failed passages** re-runs only failures.

## Outputs (next to Nahuatl input)

- `english_all.txt` / `spanish_all.txt` — with `=== Passage NNN ===` markers
- `passages/` — per-passage JSON records
- `flags_all.txt`, `failed_passages.log`, `truncated.log`, `run_summary.json`

Test runs use `_test` suffixes.

## Model

**`claude-opus-4-8`** for live and batch. Override with `CLAUDE_MODEL`.

## System prompt

Edit `wikowi_codex_prompt_FINAL.md` in this folder.
