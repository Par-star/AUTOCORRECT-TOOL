"""
combined_autocorrect.py
------------------------
The full pipeline: run TextBlob first to fix obvious typos, then run the
BERT context-checker over the result to catch "real word, wrong context"
mistakes that TextBlob can never see.

Pipeline:
    raw text
        -> [TextBlob] fixes non-words / typos (e.g. "havv" -> "have")
        -> [BERT]     fixes real-word-but-wrong-context mistakes
                      (e.g. "your short" -> "your shirt")
        -> final corrected text
"""

from basic_autocorrect import autocorrect_text
from context_aware_autocorrect import ContextAwareAutocorrect


class SmartAutocorrect:
    def __init__(self, bert_model: str = "bert-base-uncased"):
        self.bert_corrector = ContextAwareAutocorrect(model_name=bert_model)

    def correct(self, text: str, verbose: bool = False) -> str:
        # Stage 1: spelling-level correction
        spelling_fixed = autocorrect_text(text)
        if verbose:
            print(f"After TextBlob (spelling): {spelling_fixed}")

        # Stage 2: context-level correction
        context_fixed = self.bert_corrector.correct(spelling_fixed, verbose=verbose)
        if verbose:
            print(f"After BERT (context):       {context_fixed}")

        return context_fixed


if __name__ == "__main__":
    smart_corrector = SmartAutocorrect()

    sample_sentences = [
        "I havv goood speling",
        "I like your short",
        "Ths systm needs testt",
    ]

    print("\n=== Combined (TextBlob + BERT) Autocorrect Demo ===\n")
    for sentence in sample_sentences:
        result = smart_corrector.correct(sentence, verbose=True)
        print(f"Original : {sentence}")
        print(f"Final    : {result}\n")
