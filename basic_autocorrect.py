"""
basic_autocorrect.py
---------------------
A simple spelling-only autocorrect tool built on TextBlob.

TextBlob's `.correct()` method is built on a Naive Bayes / edit-distance
spelling corrector (based on Peter Norvig's classic algorithm). It looks at
each word in isolation, compares it to a dictionary of known English words
weighted by frequency, and swaps in the closest match if the word looks
misspelled.

STRENGTHS
  - Fast, no downloads, no GPU needed.
  - Great at catching "typo" style mistakes: havv -> have, speling -> spelling.

LIMITATIONS (by design of this simple approach)
  - No context awareness: it looks at ONE word at a time, never the
    sentence around it. So "I like your short" is never touched, because
    "short" is a perfectly valid English word - the model has no idea it's
    the wrong word for this sentence (see context_aware_autocorrect.py).
  - Can "correct" a word that was already right, into a different real
    word, if the original word looks statistically less common than a
    close neighbor (e.g. "is" -> "as" in some inputs).
  - Struggles on badly mangled words ("Ths" -> "The" instead of "This")
    because it is only doing local edit-distance/frequency scoring.
"""

from textblob import TextBlob


def autocorrect_text(text: str) -> str:
    """
    Run TextBlob's spelling correction over a piece of text.

    Args:
        text: The raw input text, possibly containing spelling mistakes.

    Returns:
        The text with likely spelling mistakes corrected.
    """
    blob = TextBlob(text)
    return str(blob.correct())


def autocorrect_word(word: str) -> str:
    """
    Correct a single word and show the model's confidence in alternatives.

    Returns the top-corrected word.
    """
    blob = TextBlob(word)
    return str(blob.correct())


def show_suggestions(word: str, n: int = 3):
    """
    Print the top-n candidate corrections for a single word along with
    TextBlob's internal confidence score for each, so you can see how the
    algorithm is "thinking".
    """
    blob = TextBlob(word)
    spelling_suggestions = blob.words[0].spellcheck() if blob.words else []
    print(f"Suggestions for '{word}':")
    for candidate, score in spelling_suggestions[:n]:
        print(f"   {candidate}  (confidence: {score:.3f})")


if __name__ == "__main__":
    sample_sentences = [
        "I havv goood speling",
        "Ths is a testt of the systm",
        "I like your short",  # classic real-word error TextBlob will MISS
    ]

    print("=== Basic (TextBlob) Autocorrect Demo ===\n")
    for sentence in sample_sentences:
        corrected = autocorrect_text(sentence)
        print(f"Original : {sentence}")
        print(f"Corrected: {corrected}\n")
