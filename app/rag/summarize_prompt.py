def build_summarize_prompt(context_chunks: list[str], focus: str | None = None) -> str:
    context_text = "\n\n".join(context_chunks)

    focus_part = ""
    if focus:
        focus_part = f"\nFocus the summary on: {focus}\n"

    prompt = f"""
You are a helpful assistant. Write a concise, well-structured summary using ONLY the context below.
If the context is insufficient, say: "I don't have enough information to summarize."

{focus_part}
Context:
{context_text}

Summary:
"""
    return prompt.strip()
