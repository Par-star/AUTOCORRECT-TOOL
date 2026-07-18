# Autocorrect Tool

A small AI project with two layers of autocorrection:

1. **Spelling-level correction** using **TextBlob** — fixes typos like
   `havv` → `have`.
2. **Context-level correction** using a pre-trained **BERT** model —
   catches real-word errors that are spelled correctly but wrong for the
   sentence, like `I like your short` → `I like your shirt`.

## Why two layers?

TextBlob's `.correct()` looks at **one word at a time** against a
frequency-weighted dictionary (a classic Norvig-style spell checker). It's
fast and needs no downloads, but it:

- Can't tell that "short" is the wrong word in "I like your short" — both
  "short" and "shirt" are valid, correctly-spelled English words, so a
  pure spellchecker has no signal to go on.
- Sometimes "fixes" a correct word into a different real word if that
  word happens to be statistically more common (e.g. `is` → `as`).
- Can mis-correct badly mangled words (e.g. `Ths` → `The` instead of
  `This`), since it only reasons about edit distance/frequency, not
  meaning.

BERT was trained on **masked language modeling**: given a sentence with
one word hidden, predict what word belongs there. That means we can mask
each word in turn and ask BERT "does this word fit here?" — giving us
context awareness TextBlob doesn't have.

## Project structure

```
autocorrect_tool/
├── basic_autocorrect.py           # Layer 1: TextBlob spelling correction
├── context_aware_autocorrect.py   # Layer 2: BERT context correction
├── combined_autocorrect.py        # Full pipeline: TextBlob -> BERT
├── demo.py                        # Side-by-side comparison demo
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

Notes:
- `basic_autocorrect.py` only needs `textblob` + `nltk` — works
  immediately, no downloads.
- `context_aware_autocorrect.py` and `combined_autocorrect.py` need
  `transformers` + `torch`. The **first time** you run them, Hugging Face
  will download the `bert-base-uncased` weights (~440 MB) — this needs
  an internet connection with access to huggingface.co. After the first
  run, the model is cached locally and loads instantly.
- If you want something lighter/faster than full BERT, swap the model
  name to `"distilbert-base-uncased"` when constructing
  `ContextAwareAutocorrect` — it's ~60% smaller and noticeably faster,
  at a small cost to accuracy.

## Usage

**Spelling only:**
```bash
python3 basic_autocorrect.py
```

**Context-aware only:**
```bash
python3 context_aware_autocorrect.py
```

**Full pipeline (recommended):**
```bash
python3 combined_autocorrect.py
```

**Programmatic use:**
```python
from combined_autocorrect import SmartAutocorrect

corrector = SmartAutocorrect()
result = corrector.correct("I like your short")
print(result)  # "I like your shirt"
```

## How the context-check works, step by step

For the sentence `"I like your short"`:

1. Mask each word in turn: `"I like your [MASK]"`.
2. Ask BERT for its top-k predictions for the masked slot — likely
   candidates: `shirt`, `hair`, `smile`, `dress`, `outfit`...
3. Check whether the original word (`short`) is among those top
   predictions.
4. It isn't → BERT's top guess (`shirt`) replaces it.

Words that fit their context fine (most words, most of the time) are left
untouched, since they'll show up among BERT's own top predictions for
that slot.

## Known limitations

- **TextBlob layer**: no context awareness on its own (that's why layer 2
  exists); can occasionally "correct" a correct word.
- **BERT layer**: judges one masked word per pass against the *rest of
  the sentence as typed* — if there are multiple mistakes very close
  together, predictions can get noisy; it can also be overly aggressive
  on unusual-but-correct phrasing (a poetic or informal sentence can get
  "corrected" into something blander). Tune `top_k` in
  `ContextAwareAutocorrect` to make it more or less strict.
- Both layers work at the word/whitespace-token level, so ensure
  reasonable spacing/punctuation in the input for best results.

## Tools and libraries used

- **TextBlob** — lexicon/frequency-based spelling correction
- **NLTK** — underlying corpora TextBlob relies on
- **Hugging Face `transformers`** — access to pre-trained BERT
- **BERT (`bert-base-uncased`)** — masked-language-model used for
  context-aware correction
- **PyTorch** — backend BERT runs on
