# PDF RAG Chatbot

A PDF question-answering chatbot built from scratch using Retrieval-Augmented Generation (RAG) — no LangChain, no abstractions, just raw understanding of every step.

## How It Works

```
PDF → chunk → embed → FAISS index       (indexing, once)
Question → embed → search → Claude      (per query)
```

1. **Load** — Extract text from any PDF using PyMuPDF
2. **Chunk** — Split into 500-word overlapping passages
3. **Embed** — Convert chunks to vectors using `sentence-transformers`
4. **Store** — Index vectors in FAISS for fast similarity search
5. **Retrieve** — Embed the user's question and find top-3 relevant chunks
6. **Generate** — Pass chunks as context to Claude and get a grounded answer

## Tech Stack

| Component | Library |
|---|---|
| PDF parsing | PyMuPDF (`fitz`) |
| Embeddings | `sentence-transformers` (all-MiniLM-L6-v2) |
| Vector store | FAISS |
| LLM | Anthropic Claude API |
| Env management | `python-dotenv` |

## Project Structure

```
pdf-rag-chatbot/
├── pdf_loader.py     # Extracts text from PDF
├── chunker.py        # Splits text into overlapping chunks
├── vector_store.py   # Embeds chunks, builds FAISS index, handles search
├── generator.py      # Calls Claude with retrieved context
├── main.py           # Wires everything together, chat loop
├── .env.example      # Environment variable template
└── README.md
```

## Setup

**1. Clone the repo**
```bash
git clone https://github.com/shivamdesale13/pdf-rag-chatbot.git
cd pdf-rag-chatbot
```

**2. Install dependencies**
```bash
pip install pymupdf sentence-transformers faiss-cpu anthropic python-dotenv
```

**3. Set up your API key**
```bash
cp .env.example .env
# Edit .env and add your Anthropic API key
```

**4. Run**
```bash
python main.py
```

Enter the path to any PDF when prompted, then start asking questions.

## Example

```
Enter path to your PDF file: ./research_paper.pdf

📄 Step 1: Loading PDF...
✂️  Step 2: Chunking text... Created 42 chunks.
🔢 Step 3: Embedding chunks...
🗂️  Step 4: Building FAISS index...

✅ Ready! Ask questions about your PDF.

Your question: What is the main contribution of this paper?

💬 Answer:
The paper proposes a novel attention mechanism that reduces...
```
