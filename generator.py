import anthropic
from dotenv import load_dotenv

load_dotenv()  # Reads .env file and loads ANTHROPIC_API_KEY into environment

client = anthropic.Anthropic()  # Automatically picks up ANTHROPIC_API_KEY from environment


def generate_answer(question: str, context_chunks: list[str]) -> str:
    """
    Builds a RAG prompt and calls Claude to generate a grounded answer.

    Args:
        question:       The user's question
        context_chunks: Retrieved text chunks from the vector store

    Returns:
        Claude's answer as a string
    """
    # Join chunks into one context block
    context = "\n\n---\n\n".join(context_chunks)

    prompt = f"""You are a helpful assistant. Answer the question using ONLY the context provided below.
If the answer is not present in the context, say "I don't know based on the provided document."
Do not make up information.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:"""

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return response.content[0].text
