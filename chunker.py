def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """
    Splits text into overlapping chunks by word count.

    Why overlap? So that sentences at chunk boundaries don't lose context.
    Example: chunk 1 ends at word 500, chunk 2 starts at word 450.

    Args:
        text:       The full document text
        chunk_size: How many words per chunk (default 500)
        overlap:    How many words to repeat between chunks (default 50)

    Returns:
        List of text chunks
    """
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)

        # Move forward by (chunk_size - overlap) so chunks overlap
        start += chunk_size - overlap

    return chunks


if __name__ == "__main__":
    sample = " ".join([f"word{i}" for i in range(1200)])  # 1200 fake words
    chunks = chunk_text(sample, chunk_size=500, overlap=50)

    print(f"Total words: 1200")
    print(f"Total chunks created: {len(chunks)}")
    print(f"Chunk 1 word count: {len(chunks[0].split())}")
    print(f"Chunk 2 starts with: {chunks[1].split()[:3]}")  # Should overlap with end of chunk 1
