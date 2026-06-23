"""
RAG Pipeline — PDF Q&A Chatbot
===============================
Flow:
  PDF → load → chunk → embed → FAISS index   (indexing, done once)
  Question → embed → search FAISS → top chunks → Claude → answer  (per query)
"""

from pdf_loader import load_pdf
from chunker import chunk_text
from vector_store import embed_chunks, build_faiss_index, search
from generator import generate_answer


def build_index(pdf_path: str):
    """
    Indexing phase — run once per document.
    Returns (chunks, faiss_index) to be reused for all queries.
    """
    print("\n📄 Step 1: Loading PDF...")
    text = load_pdf(pdf_path)
    print(f"   Extracted {len(text.split())} words from the PDF.")

    print("\n✂️  Step 2: Chunking text...")
    chunks = chunk_text(text, chunk_size=500, overlap=50)
    print(f"   Created {len(chunks)} chunks.")

    print("\n🔢 Step 3: Embedding chunks...")
    embeddings = embed_chunks(chunks)

    print("\n🗂️  Step 4: Building FAISS index...")
    index = build_faiss_index(embeddings)

    return chunks, index


def ask(question: str, chunks: list, index) -> str:
    """
    Retrieval + Generation phase — run per user query.
    """
    print(f"\n🔍 Retrieving relevant chunks for: '{question}'")
    top_chunks = search(question, index, chunks, top_k=3)
    print(f"   Found {len(top_chunks)} relevant chunks.")

    print("\n🤖 Generating answer with Claude...")
    answer = generate_answer(question, top_chunks)
    return answer


def main():
    pdf_path = input("Enter path to your PDF file: ").strip()

    # Build index once
    chunks, index = build_index(pdf_path)

    print("\n✅ Ready! Ask questions about your PDF. Type 'quit' to exit.\n")
    print("=" * 60)

    while True:
        question = input("\nYour question: ").strip()
        if question.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break
        if not question:
            continue

        answer = ask(question, chunks, index)
        print(f"\n💬 Answer:\n{answer}")
        print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
