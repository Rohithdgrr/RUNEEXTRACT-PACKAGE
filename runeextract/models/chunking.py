from __future__ import annotations

import re
from typing import List, Optional

from runeextract.models.types import Chunk, ChunkingStrategy, _get_token_encoding


def chunk_fixed_size(text: str, size: int, overlap: int) -> List[Chunk]:
    chunks = []
    start = 0
    chunk_id = 0
    while start < len(text):
        end = start + size
        chunk_text = text[start:end]
        chunks.append(Chunk(
            text=chunk_text,
            chunk_id=f"chunk_{chunk_id}",
            start_index=start,
            end_index=end,
            metadata={"strategy": "fixed_size", "size": size, "overlap": overlap}
        ))
        step = max(size - overlap, 1)
        start += step
        chunk_id += 1
    return chunks


def chunk_by_page(text: str, page_breaks: Optional[List[int]] = None) -> List[Chunk]:
    if not page_breaks:
        return [Chunk(
            text=text,
            chunk_id="chunk_0",
            start_index=0,
            end_index=len(text),
            metadata={"strategy": "by_page"}
        )]
    chunks = []
    prev = 0
    for idx, break_pos in enumerate(page_breaks):
        chunk_text = text[prev:break_pos]
        chunks.append(Chunk(
            text=chunk_text,
            chunk_id=f"chunk_{idx}",
            start_index=prev,
            end_index=break_pos,
            metadata={"strategy": "by_page", "page": idx + 1}
        ))
        prev = break_pos
    if prev < len(text):
        chunks.append(Chunk(
            text=text[prev:],
            chunk_id=f"chunk_{len(chunks)}",
            start_index=prev,
            end_index=len(text),
            metadata={"strategy": "by_page", "page": len(chunks) + 1}
        ))
    return chunks


def chunk_by_heading(text: str) -> List[Chunk]:
    heading_pattern = re.compile(
        r'^(#{1,6}\s+.*)$|^(.+)\n[=\-]+\s*$',
        re.MULTILINE
    )
    matches = list(heading_pattern.finditer(text))
    if not matches:
        return chunk_fixed_size(text, 1000, 0)

    chunks = []
    prev_end = 0
    chunk_counter = 0
    for match in matches:
        heading_text = (match.group(1) or match.group(2)).strip()
        if match.start() > prev_end:
            chunk_text = text[prev_end:match.start()]
            if chunk_text.strip():
                chunks.append(Chunk(
                    text=chunk_text.strip(),
                    chunk_id=f"chunk_{chunk_counter}",
                    start_index=prev_end,
                    end_index=match.start(),
                    metadata={"strategy": "by_heading"}
                ))
                chunk_counter += 1
        prev_end = match.end()
    if prev_end < len(text):
        chunks.append(Chunk(
            text=text[prev_end:].strip(),
            chunk_id=f"chunk_{chunk_counter}",
            start_index=prev_end,
            end_index=len(text),
            metadata={"strategy": "by_heading"}
        ))
    return chunks or [Chunk(
        text=text,
        chunk_id="chunk_0",
        start_index=0,
        end_index=len(text),
        metadata={"strategy": "by_heading"}
    )]


def chunk_semantic(text: str, size: int) -> List[Chunk]:
    if text.strip():
        sep = "_PARAGRAPH_BREAK_"
        text_with_markers = text.replace('\n\n', f'\n{sep}\n')
    else:
        text_with_markers = text
    parts = text_with_markers.split(sep)
    chunks = []
    current_chunk = ""
    chunk_id = 0
    start_index = 0
    for part in parts:
        part = part.strip()
        if not part:
            continue
        part_len = len(part)
        if len(current_chunk) + part_len > size and current_chunk:
            chunks.append(Chunk(
                text=current_chunk.strip(),
                chunk_id=f"chunk_{chunk_id}",
                start_index=start_index,
                end_index=start_index + len(current_chunk),
                metadata={"strategy": "semantic"}
            ))
            start_index += len(current_chunk)
            current_chunk = part
            chunk_id += 1
        else:
            if current_chunk:
                current_chunk += "\n\n" + part
            else:
                current_chunk = part
    if current_chunk:
        chunks.append(Chunk(
            text=current_chunk.strip(),
            chunk_id=f"chunk_{chunk_id}",
            start_index=start_index,
            end_index=start_index + len(current_chunk),
            metadata={"strategy": "semantic"}
        ))
    return chunks if chunks else [Chunk(
        text=text,
        chunk_id="chunk_0",
        start_index=0,
        end_index=len(text),
        metadata={"strategy": "semantic"}
    )]


def chunk_by_token(text: str, size: int, overlap: int, encoding_name: str = "cl100k_base") -> List[Chunk]:
    enc = _get_token_encoding(encoding_name)
    chunks = []
    chunk_id = 0
    start = 0

    if enc:
        tokens = enc.encode(text)
        token_count = len(tokens)
        while start < token_count:
            end = min(start + size, token_count)
            chunk_tokens = tokens[start:end]
            chunk_text = enc.decode(chunk_tokens)
            char_start = len(enc.decode(tokens[:start])) if start > 0 else 0
            char_end = char_start + len(chunk_text)
            chunks.append(Chunk(
                text=chunk_text,
                chunk_id=f"chunk_{chunk_id}",
                start_index=char_start,
                end_index=char_end,
                metadata={
                    "strategy": "by_token",
                    "size": size,
                    "overlap": overlap,
                    "encoding": encoding_name,
                    "token_start": start,
                    "token_end": end,
                }
            ))
            step = max(size - overlap, 1)
            start += step
            chunk_id += 1
    else:
        chunks = chunk_fixed_size(text, size * 4, overlap * 4)
    return chunks


def chunk_sentence_window(text: str, size: int = 5, overlap: int = 1) -> List[Chunk]:
    sentence_endings = re.finditer(r'(?<=[.!?])\s+', text)
    boundaries = [0]
    for m in sentence_endings:
        boundaries.append(m.end())
    if boundaries[-1] < len(text):
        boundaries.append(len(text))

    sentences = []
    for i in range(len(boundaries) - 1):
        sent_text = text[boundaries[i]:boundaries[i + 1]].strip()
        if sent_text:
            sentences.append((sent_text, boundaries[i], boundaries[i + 1]))

    if not sentences:
        return [Chunk(text=text, chunk_id="chunk_0", start_index=0, end_index=len(text),
                      metadata={"strategy": "sentence_window", "size": size, "overlap": overlap})]

    chunks = []
    chunk_id = 0
    step = max(size - overlap, 1)
    for i in range(0, len(sentences), step):
        window = sentences[i:i + size]
        if not window:
            break
        chunk_text = " ".join(s[0] for s in window)
        start_idx = window[0][1]
        end_idx = window[-1][2]
        chunks.append(Chunk(
            text=chunk_text,
            chunk_id=f"chunk_{chunk_id}",
            start_index=start_idx,
            end_index=end_idx,
            metadata={
                "strategy": "sentence_window",
                "size": size,
                "overlap": overlap,
                "num_sentences": len(window),
            }
        ))
        chunk_id += 1
    return chunks
