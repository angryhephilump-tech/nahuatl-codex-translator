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

1. Drop one or more Nahuatl `.txt` files on the left zone (or **Browse**).
2. Drop one or more matching English reference `.txt` files on the right zone.
3. Multiple files on a side are merged in **alphabetical order** before splitting.
4. Click **Split & Preview** to verify the first 3 aligned pairs.
5. Optionally enable **Test mode** to translate only the first ~300 words (cheap sanity check).
6. Choose **Batch (50% off)** — default — or **Live (immediate)**.
7. Click **Run Translation** to process all passages.

Outputs are saved next to the Nahuatl input file:

- `english_all.txt`
- `spanish_all.txt`
- `flags_all.txt` (only if any passage returned `<flags>`)
- `run_summary.json` / `run_summary.txt` — model, mode, tokens, cost, failures
- `failed_passages.log` — if any passage failed
- `batch_state.json` — written while a batch is in flight (removed on clean completion)

## Model

Uses **`claude-opus-4-8`** (Claude Opus) for both live and batch modes — same as your PDF transcriber.
Override with the `CLAUDE_MODEL` environment variable if needed.

## Splitting

Both files are split on the same logic:

- Chapter-style headings (e.g. `Chapter 1`, `BOOK II`, markdown `#` headings), or
- Double line breaks if no headings are detected.

Passage *n* in Nahuatl is paired with passage *n* in English.

## API modes

- **Batch (50% off)** — default. Submits all passages via Anthropic Message Batches API (~50% token discount). Results usually arrive within an hour; the app polls until done.
- **Live (immediate)** — one request per passage at full price, with live progress.

## Test mode

Check **Test mode (first 300 words)** before Split & Preview. Both sides are truncated to the first 300 words, then split and translated. Outputs use a `_test` suffix (`english_all_test.txt`, etc.) so they won't overwrite a full run.

## System prompt

Edit `wikowi_codex_prompt_FINAL.md` in this folder to change translation instructions.
