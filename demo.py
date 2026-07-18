"""
demo.py
--------
Side-by-side demo: shows what TextBlob alone does vs. what the full
TextBlob + BERT pipeline does, on the same set of example sentences -
including the classic "spelled correctly, wrong word" case.

Run:
    python3 demo.py
"""

from basic_autocorrect import autocorrect_text

TEST_SENTENCES = [
    "I havv goood speling",              # pure typos
    "Ths is a testt of the systm",       # pure typos, some tricky
    "I like your short",                 # real-word error (needs BERT)
    "Can you pass the sult please",      # typo BERT can also fix via context
]


def run_basic_only():
    print("=" * 60)
    print("LAYER 1 ONLY: TextBlob (spelling-level)")
    print("=" * 60)
    for s in TEST_SENTENCES:
        print(f"  Original : {s}")
        print(f"  Corrected: {autocorrect_text(s)}\n")


def run_full_pipeline():
    print("=" * 60)
    print("LAYER 1 + 2: TextBlob + BERT (spelling + context)")
    print("=" * 60)
    try:
        from combined_autocorrect import SmartAutocorrect
    except ImportError:
        print("Skipping: install `transformers` and `torch` to run this "
              "layer (see requirements.txt).")
        return

    try:
        corrector = SmartAutocorrect()
    except Exception as e:
        print(f"Skipping: could not load BERT model ({e}).\n"
              f"This usually means no internet access to huggingface.co "
              f"for the first-time model download.")
        return

    for s in TEST_SENTENCES:
        print(f"  Original : {s}")
        print(f"  Corrected: {corrector.correct(s)}\n")


if __name__ == "__main__":
    run_basic_only()
    run_full_pipeline()
