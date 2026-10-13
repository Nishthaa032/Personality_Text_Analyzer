from __future__ import annotations

import re
from collections import Counter

_SENTENCE_PATTERN = re.compile(r"(?<=[.!?])\s+")
_WORD_PATTERN = re.compile(r"[A-Za-z0-9']+")
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "has", "have", "i", "if", "in", "is", "it", "me", "my", "of", "on",
    "or", "our", "that", "the", "this", "to", "was", "we", "with", "you",
    "your",
}


def summarize_text(text: str, max_sentences: int = 3) -> str:
    """Return a short extractive summary without requiring an external model."""
    if max_sentences < 1:
        raise ValueError("max_sentences must be at least 1")

    sentences = [sentence.strip() for sentence in _SENTENCE_PATTERN.split(text.strip()) if sentence.strip()]
    if len(sentences) <= max_sentences:
        return " ".join(sentences)

    words = [
        word.lower()
        for word in _WORD_PATTERN.findall(text)
        if word.lower() not in _STOPWORDS
    ]
    frequencies = Counter(words)
    scored = []
    for index, sentence in enumerate(sentences):
        sentence_words = [
            word.lower()
            for word in _WORD_PATTERN.findall(sentence)
            if word.lower() not in _STOPWORDS
        ]
        score = sum(frequencies[word] for word in sentence_words) / max(len(sentence_words), 1)
        scored.append((score, index, sentence))

    selected = sorted(scored, key=lambda item: item[0], reverse=True)[:max_sentences]
    return " ".join(sentence for _, _, sentence in sorted(selected, key=lambda item: item[1]))
