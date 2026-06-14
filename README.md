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
3. Leave **AI-assisted alignment** checked (recommended) or turn it off for already-aligned mechanical pairing.
4. **Split & Preview** — check the green/red alignment banner and eyeball the first ~10 words of each side.
5. Use **Preview pair #** to inspect any passage number (misalignment often starts mid-file).
6. **Run Translation** is blocked on count mismatch in mechanical mode; AI mode always pairs by meaning.
7. Optionally: **Test mode** (~300 words), **Batch (50% off)**, or **Live**.
8. After failures: **Retry failed passages** (does not re-run successful ones).

## Alignment safety

- **Mechanical mode:** Green `Nahuatl: N | English: N — ALIGNED` → Run enabled. Red mismatch → Run blocked (no silent `min(count)` pairing).
- **AI-assisted mode (default):** Splits Nahuatl only, then uses Claude Haiku to match each passage to the correct English span. Uncertain matches are highlighted yellow in preview. Results cached in `alignment_map.json`.

## Split methods

All splitting respects natural boundaries — chapter headings, `\n\n` paragraph breaks, or sentence/quote-safe word limits (never mid-sentence or mid-quotation when avoidable).

- **Chapter headings** — editable regex (defaults to Chapter/Book/markdown heading patterns; falls back to paragraphs if no headings found)
- **Paragraph breaks** (`\n\n`)
- **Every N words** — set N (default 400); large passages subdivided at sentence boundaries; warns if any passage exceeds 3000 words

## Resume & retry

- Each passage saves to `passages/passage_NNNNN.json` as it completes; aggregate `.txt` files rebuild incrementally.
- Re-running skips passages already on disk (unless failed or truncated).
- **Retry failed passages** re-runs only failures listed in `failed_passages.log`.

## Truncation

If the API returns `stop_reason: max_tokens`, the passage is flagged and logged to `truncated.log`. Truncated output is **not** saved as complete in the aggregate files.

## Prompt caching

The system prompt uses Anthropic prompt caching (`cache_control: ephemeral`) on every Opus call (live and batch). Run summary shows cache read vs creation tokens and fresh (non-cache) input.

## Outputs (next to Nahuatl input)

- `english_all.txt` / `spanish_all.txt` — with `=== Passage NNN ===` markers
- `passages/` — per-passage JSON records
- `alignment_map.json` — cached AI alignment pairs
- `flags_all.txt`, `failed_passages.log`, `truncated.log`, `run_summary.json`

Test runs use `_test` suffixes.

## Models

- **Translation:** `claude-opus-4-8` (override with `CLAUDE_MODEL`)
- **AI alignment:** `claude-haiku-4-5` (override with `ALIGNMENT_MODEL`)

## System prompt

Edit `wikowi_codex_prompt_FINAL.md` in this folder.
