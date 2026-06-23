"""
Module 6 Week A — Core Skills Drill: Text Processing & NLP Basics
"""

import spacy

nlp = spacy.load("en_core_web_sm")


def preprocess_text(text, stop_words):
    doc = nlp(text)
    return [
        token.text.lower()
        for token in doc
        if not token.is_punct
        and not token.is_space
        and token.text.lower() not in stop_words
    ]


def extract_linguistic_annotations(text):
    doc = nlp(text)
    return [(token.text, token.pos_, token.dep_) for token in doc]


def extract_entities(text):
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]


if __name__ == "__main__":
    sample = (
        "The IPCC released its latest report in Geneva on March 20, 2024. "
        "Dr. Ahmad presented findings on Jordan's climate adaptation strategy "
        "at the COP28 conference in Dubai."
    )
    stop_words = {"the", "a", "an", "in", "on", "at", "its", "of", "and", "is"}

    tokens = preprocess_text(sample, stop_words)
    if tokens is not None:
        print(f"Cleaned tokens ({len(tokens)}): {tokens[:10]}...")

    annotations = extract_linguistic_annotations(sample)
    if annotations is not None:
        print(f"\nAnnotations ({len(annotations)} tokens):")
        for tok, pos, dep in annotations[:5]:
            print(f"  {tok:15s} {pos:8s} {dep}")

    entities = extract_entities(sample)
    if entities is not None:
        print(f"\nEntities ({len(entities)}):")
        for text, label in entities:
            print(f"  {text:25s} {label}")