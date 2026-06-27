"""
Chunker Service — splits document text using multiple strategies.
Strategies: fixed, recursive, semantic (sentence-based), markdown, token-based.
"""
import re
from typing import List, Dict, Any


class Chunker:

    @staticmethod
    def chunk_document(
        text: str,
        strategy: str = "recursive",
        size: int = 512,
        overlap: int = 64,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Main entry point. Returns list of chunk dicts:
        {index, text_content, token_count, char_count, strategy}
        """
        text = text.strip()
        if not text:
            return []

        if strategy == "fixed":
            chunks = Chunker._fixed_chunks(text, size, overlap)
        elif strategy == "semantic":
            chunks = Chunker._semantic_chunks(text, size)
        elif strategy == "markdown":
            chunks = Chunker._markdown_chunks(text, size, overlap)
        elif strategy == "sentence":
            chunks = Chunker._sentence_chunks(text, size, overlap)
        elif strategy == "token":
            chunks = Chunker._token_chunks(text, size, overlap)
        else:
            # Default: recursive
            chunks = Chunker._recursive_chunks(text, size, overlap)

        return [
            {
                "index": i,
                "text_content": c,
                "token_count": Chunker._estimate_tokens(c),
                "char_count": len(c),
                "strategy": strategy,
            }
            for i, c in enumerate(chunks)
            if c.strip()
        ]

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Approximation: 1 token ≈ 4 chars."""
        return max(1, len(text) // 4)

    @staticmethod
    def _fixed_chunks(text: str, size: int, overlap: int) -> List[str]:
        """Split by fixed character count with overlap."""
        chunks = []
        start = 0
        step = max(1, size - overlap)
        while start < len(text):
            end = min(start + size, len(text))
            chunks.append(text[start:end])
            start += step
        return chunks

    @staticmethod
    def _recursive_chunks(text: str, size: int, overlap: int) -> List[str]:
        """
        Recursive character splitting: tries to split at paragraph, then sentence,
        then word boundaries to keep chunks under `size` chars.
        """
        separators = ["\n\n", "\n", ". ", "! ", "? ", " ", ""]

        def split_text(txt: str, seps: List[str]) -> List[str]:
            if len(txt) <= size or not seps:
                return [txt] if txt.strip() else []
            sep = seps[0]
            splits = txt.split(sep)
            chunks = []
            current = ""
            for part in splits:
                candidate = current + (sep if current else "") + part
                if len(candidate) <= size:
                    current = candidate
                else:
                    if current:
                        chunks.append(current)
                    if len(part) > size:
                        sub_chunks = split_text(part, seps[1:])
                        chunks.extend(sub_chunks)
                        current = ""
                    else:
                        current = part
            if current:
                chunks.append(current)
            return chunks

        raw_chunks = split_text(text, separators)

        # Apply overlap by merging context from previous chunk
        if overlap <= 0 or len(raw_chunks) <= 1:
            return raw_chunks

        result = [raw_chunks[0]]
        for i in range(1, len(raw_chunks)):
            prev_tail = result[-1][-overlap:] if len(result[-1]) > overlap else result[-1]
            result.append(prev_tail + " " + raw_chunks[i])
        return result

    @staticmethod
    def _semantic_chunks(text: str, max_size: int) -> List[str]:
        """
        Sentence-boundary chunking — groups sentences until chunk reaches max_size.
        """
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current = ""
        for sentence in sentences:
            if len(current) + len(sentence) + 1 <= max_size:
                current += (" " if current else "") + sentence
            else:
                if current:
                    chunks.append(current)
                # If a single sentence is too long, split it further
                if len(sentence) > max_size:
                    words = sentence.split()
                    part = ""
                    for word in words:
                        if len(part) + len(word) + 1 <= max_size:
                            part += (" " if part else "") + word
                        else:
                            if part:
                                chunks.append(part)
                            part = word
                    current = part
                else:
                    current = sentence
        if current:
            chunks.append(current)
        return chunks

    @staticmethod
    def _sentence_chunks(text: str, size: int, overlap: int) -> List[str]:
        """Similar to semantic but with sentence-level overlap."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_sentences = []
        current_len = 0

        for sent in sentences:
            if current_len + len(sent) > size and current_sentences:
                chunks.append(" ".join(current_sentences))
                # Keep last N sentences for overlap
                overlap_sents = []
                overlap_len = 0
                for s in reversed(current_sentences):
                    if overlap_len + len(s) < overlap:
                        overlap_sents.insert(0, s)
                        overlap_len += len(s)
                    else:
                        break
                current_sentences = overlap_sents
                current_len = overlap_len

            current_sentences.append(sent)
            current_len += len(sent) + 1

        if current_sentences:
            chunks.append(" ".join(current_sentences))

        return chunks

    @staticmethod
    def _markdown_chunks(text: str, size: int, overlap: int) -> List[str]:
        """Split at markdown headers (##, ###) first, then recursively split large sections."""
        sections = re.split(r'(?=^#{1,3} )', text, flags=re.MULTILINE)
        result = []
        for section in sections:
            if section.strip():
                if len(section) <= size:
                    result.append(section.strip())
                else:
                    sub = Chunker._recursive_chunks(section, size, overlap)
                    result.extend(sub)
        return result

    @staticmethod
    def _token_chunks(text: str, token_size: int, overlap_tokens: int) -> List[str]:
        """Approximate token-based splitting (1 token ≈ 4 chars)."""
        char_size = token_size * 4
        char_overlap = overlap_tokens * 4
        return Chunker._fixed_chunks(text, char_size, char_overlap)

    @staticmethod
    def available_strategies() -> List[Dict[str, str]]:
        return [
            {"id": "recursive", "name": "Recursive", "description": "Smart split at paragraph → sentence → word boundaries (recommended)"},
            {"id": "fixed", "name": "Fixed Character", "description": "Split at exact character count with overlap"},
            {"id": "semantic", "name": "Semantic (Sentence)", "description": "Group full sentences to preserve meaning"},
            {"id": "sentence", "name": "Sentence Overlap", "description": "Sentence-based with configurable sentence overlap"},
            {"id": "markdown", "name": "Markdown", "description": "Split at Markdown headers (##, ###)"},
            {"id": "token", "name": "Token-Based", "description": "Split by approximate token count (1 token ≈ 4 chars)"},
        ]
