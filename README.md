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

1. Drop the Nahuatl `.txt` file on the left zone.
2. Drop the matching English reference `.txt` on the right zone.
3. Click **Split & Preview** to verify the first 3 aligned pairs.
4. Click **Run Translation** to process all passages.

Outputs are saved next to the Nahuatl input file:

- `english_all.txt`
- `spanish_all.txt`
- `flags_all.txt` (only if any passage returned `<flags>`)

## Splitting

Both files are split on the same logic:

- Chapter-style headings (e.g. `Chapter 1`, `BOOK II`, markdown `#` headings), or
- Double line breaks if no headings are detected.

Passage *n* in Nahuatl is paired with passage *n* in English.

## System prompt

Edit `wikowi_codex_prompt_FINAL.md` in this folder to change translation instructions.
