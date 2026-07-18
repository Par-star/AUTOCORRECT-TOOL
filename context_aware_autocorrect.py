"""
context_aware_autocorrect.py
-----------------------------
A context-aware autocorrect tool built on a pre-trained BERT
masked-language-model (via Hugging Face `transformers`).

WHY BERT?
TextBlob (see basic_autocorrect.py) checks each word in isolation against a
dictionary. It has no notion of which words "belong together". BERT, on the
other hand, was trained to predict a missing word FROM ITS CONTEXT (the
"masked language modeling" objective). That means we can:

  1. Mask each word in a sentence, one at a time.
  2. Ask BERT: "given everything else in this sentence, what word most
     likely goes here?"
  3. Compare BERT's top guesses to the word that was actually typed.
  4. If the typed word is a low-probability outlier for this context (e.g.
     "short" in "I like your ___", where BERT strongly prefers "shirt",
     "smile", "hair", etc.), flag or replace it.

This lets us catch REAL-WORD errors that are spelled correctly but wrong in
context - exactly the "I like your short" -> "I like your shirt" case that
a pure spellchecker like TextBlob will always miss.

NOTE ON SETUP
The first time you run this, `transformers` will download the pre-trained
"bert-base-uncased" weights (~440MB) from Hugging Face. You'll need an
internet connection with access to huggingface.co for that one-time
download; after that it's cached locally.

Install with:
    pip install transformers torch
"""

from typing import List, Tuple
import re

try:
    from transformers import pipeline
except ImportError as e:
    raise ImportError(
        "This module needs the `transformers` and `torch` packages.\n"
        "Install them with: pip install transformers torch"
    ) from e


class ContextAwareAutocorrect:
    """
    Wraps a BERT fill-mask pipeline to do context-based word correction.
    """

    def __init__(self, model_name: str = "bert-base-uncased", top_k: int = 5):
        """
        Args:
            model_name: Any Hugging Face masked-LM checkpoint
                        (e.g. "bert-base-uncased", "distilbert-base-uncased"
                        for a smaller/faster model).
            top_k: How many BERT candidate predictions to consider per word.
        """
        print(f"Loading model '{model_name}' (first run downloads weights)...")
        self.fill_mask = pipeline("fill-mask", model=model_name, top_k=top_k)
        self.mask_token = self.fill_mask.tokenizer.mask_token
        self.top_k = top_k

    @staticmethod
    def _split_word(token: str) -> Tuple[str, str, str]:
        """Split a raw token into (leading punctuation, core word, trailing punctuation)."""
        match = re.match(r"^(\W*)(\w+)(\W*)$", token)
        if not match:
            return "", token, ""
        return match.group(1), match.group(2), match.group(3)

    def check_sentence(self, text: str, verbose: bool = False) -> List[dict]:
        """
        Run every word in `text` through BERT, one at a time, and report
        whether the original word appears among BERT's top predictions for
        that spot.

        Returns a list of dicts, one per word:
            {
                "word": original word,
                "in_top_predictions": bool,
                "top_prediction": BERT's #1 guess for that slot,
                "predictions": full list of (word, score) candidates
            }
        """
        raw_tokens = text.split()
        report = []

        for i, token in enumerate(raw_tokens):
            lead, core, trail = self._split_word(token)

            if not core.isalpha():
                # Skip pure numbers/symbols - nothing for BERT to judge here.
                report.append({
                    "word": token,
                    "in_top_predictions": True,
                    "top_prediction": token,
                    "predictions": [],
                })
                continue

            masked_tokens = raw_tokens.copy()
            masked_tokens[i] = f"{lead}{self.mask_token}{trail}"
            masked_sentence = " ".join(masked_tokens)

            predictions = self.fill_mask(masked_sentence)
            candidates = [(p["token_str"].strip(), p["score"]) for p in predictions]
            candidate_words = [c[0].lower() for c in candidates]

            in_top = core.lower() in candidate_words
            top_guess = candidates[0][0] if candidates else core

            if verbose:
                print(f"  '{core}' -> candidates: {candidates}")

            report.append({
                "word": token,
                "in_top_predictions": in_top,
                "top_prediction": top_guess,
                "predictions": candidates,
            })

        return report

    def correct(self, text: str, verbose: bool = False) -> str:
        """
        Replace any word that BERT considers a poor fit for its context
        with BERT's top suggestion for that slot. Words already among the
        top predictions are left untouched.
        """
        report = self.check_sentence(text, verbose=verbose)
        raw_tokens = text.split()
        corrected_tokens = raw_tokens.copy()

        for i, entry in enumerate(report):
            if not entry["in_top_predictions"] and entry["predictions"]:
                lead, core, trail = self._split_word(entry["word"])
                corrected_tokens[i] = f"{lead}{entry['top_prediction']}{trail}"

        return " ".join(corrected_tokens)


if __name__ == "__main__":
    corrector = ContextAwareAutocorrect(model_name="bert-base-uncased")

    sample_sentences = [
        "I like your short",          # real-word error: should be "shirt"
        "I am going to the beach to eat some sand",  # context-plausible, should stay
        "Can you pass the sult please",  # not a real word - BERT can still guess "salt"
    ]

    print("\n=== Context-Aware (BERT) Autocorrect Demo ===\n")
    for sentence in sample_sentences:
        corrected = corrector.correct(sentence, verbose=True)
        print(f"Original : {sentence}")
        print(f"Corrected: {corrected}\n")
