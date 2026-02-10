def build_rag_prompt(context_chunks: list[str], question: str) -> str:
    context_text = "\n\n".join(context_chunks)

    prompt = f"""
You are a helpful assistant. Answer the question using ONLY the context below.
If the answer is not in the context, say: "I don't know based on the provided content."

Context:
{context_text}

Question:
{question}

Answer:
"""
    return prompt.strip()
