from typing import List, Dict, Any
import re

class Chunker:
    @staticmethod
    def split_fixed(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        if chunk_size <= 0:
            return [text]
        
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])
            start += chunk_size - chunk_overlap
            
            # Prevent infinite loops in edge cases
            if chunk_overlap >= chunk_size:
                start += 1
                
        return chunks

    @staticmethod
    def split_recursive(text: str, chunk_size: int, chunk_overlap: int, separators: List[str] = None) -> List[str]:
        if separators is None:
            separators = ["\n\n", "\n", " ", ""]
            
        final_chunks = []
        
        # Helper to recursively split text blocks
        def split_text(text_block: str, separator_index: int):
            if len(text_block) <= chunk_size or separator_index >= len(separators):
                final_chunks.append(text_block)
                return
            
            sep = separators[separator_index]
            if sep == "":
                # Hard character fallback
                sub_blocks = Chunker.split_fixed(text_block, chunk_size, chunk_overlap)
                final_chunks.extend(sub_blocks)
                return
            
            splits = text_block.split(sep)
            current_chunk = ""
            
            for part in splits:
                if len(current_chunk) + len(part) + len(sep) <= chunk_size:
                    current_chunk += (sep if current_chunk else "") + part
                else:
                    if current_chunk:
                        final_chunks.append(current_chunk)
                    
                    # If single part is larger than chunk_size, split it further
                    if len(part) > chunk_size:
                        split_text(part, separator_index + 1)
                    else:
                        current_chunk = part
                        
            if current_chunk:
                final_chunks.append(current_chunk)

        split_text(text, 0)
        
        # Implement optional overlap adjustment post-split
        if chunk_overlap > 0:
            overlapped_chunks = []
            for i, chunk in enumerate(final_chunks):
                if i == 0:
                    overlapped_chunks.append(chunk)
                else:
                    prev_chunk = final_chunks[i-1]
                    overlap_text = prev_chunk[-chunk_overlap:] if len(prev_chunk) >= chunk_overlap else prev_chunk
                    overlapped_chunks.append(overlap_text + chunk)
            return overlapped_chunks
            
        return final_chunks

    @staticmethod
    def split_semantic(text: str, chunk_size: int) -> List[str]:
        # Basic sentence splitting
        sentences = re.split(r'(?<=[.!?])\s+', text)
        if not sentences or len(sentences) == 1:
            return [text]
            
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_len = len(sentence)
            if current_length + sentence_len <= chunk_size:
                current_chunk.append(sentence)
                current_length += sentence_len
            else:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                current_chunk = [sentence]
                current_length = sentence_len
                
        if current_chunk:
            chunks.append(" ".join(current_chunk))
            
        return chunks
        
    @classmethod
    def chunk_document(cls, text: str, method: str, chunk_size: int, chunk_overlap: int) -> List[Dict[str, Any]]:
        if method.lower() == "fixed":
            raw_chunks = cls.split_fixed(text, chunk_size, chunk_overlap)
        elif method.lower() == "recursive":
            raw_chunks = cls.split_recursive(text, chunk_size, chunk_overlap)
        elif method.lower() == "semantic":
            raw_chunks = cls.split_semantic(text, chunk_size)
        else:
            # Default fallback
            raw_chunks = cls.split_recursive(text, chunk_size, chunk_overlap)
            
        chunks_metadata = []
        for index, content in enumerate(raw_chunks):
            # Roughly estimate token count (1 token ≈ 4 characters)
            tokens = max(1, len(content) // 4)
            chunks_metadata.append({
                "index": index,
                "text_content": content,
                "token_count": tokens
            })
            
        return chunks_metadata
