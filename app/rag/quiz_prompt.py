def build_quiz_prompt(context_chunks: list[str], focus: str | None, num_questions: int) -> str:
    context_text = "\n\n".join(context_chunks)

    focus_part = ""
    if focus:
        focus_part = f"\nFocus the quiz on: {focus}\n"

    prompt = f"""
You are a helpful tutor. Using ONLY the context below, generate {num_questions} quiz questions.
- Prefer multiple-choice questions (MCQs) with 4 options each.
- Mark the correct answer for each question.
- If the context is insufficient, say: "I don't have enough information to generate a quiz."

{focus_part}
Context:
{context_text}

Output format:
1) Question
A) Option
B) Option
C) Option
D) Option
Correct: <letter>

Quiz:
"""
    return prompt.strip()
