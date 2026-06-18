from __future__ import annotations

from collections import Counter
from typing import Any

from akan_speech.data.normalize import normalize_akan_text


def tokenizer_fragmentation(tokenizer: Any, texts: list[str], *, top_k: int = 30) -> dict:
    words = []
    total_text_tokens = 0
    total_characters = 0
    for text in texts:
        normalized = normalize_akan_text(text)
        total_text_tokens += len(tokenizer(normalized, add_special_tokens=False).input_ids)
        total_characters += len(normalized.replace(" ", ""))
        words.extend(normalized.split())

    word_counts = Counter(words)
    token_counts = {
        word: len(tokenizer(word, add_special_tokens=False).input_ids) for word in word_counts
    }
    weighted_word_tokens = sum(token_counts[word] * count for word, count in word_counts.items())
    total_words = sum(word_counts.values())
    fragmented = sorted(
        (
            {
                "word": word,
                "token_count": token_counts[word],
                "frequency": frequency,
                "weighted_cost": token_counts[word] * frequency,
            }
            for word, frequency in word_counts.items()
        ),
        key=lambda row: (row["token_count"], row["weighted_cost"], row["word"]),
        reverse=True,
    )
    return {
        "texts": len(texts),
        "words": total_words,
        "unique_words": len(word_counts),
        "text_tokens": total_text_tokens,
        "tokens_per_word": round(weighted_word_tokens / max(total_words, 1), 4),
        "characters_per_token": round(total_characters / max(total_text_tokens, 1), 4),
        "words_split_into_3plus_tokens": sum(
            frequency for word, frequency in word_counts.items() if token_counts[word] >= 3
        ),
        "most_fragmented_words": fragmented[:top_k],
    }
