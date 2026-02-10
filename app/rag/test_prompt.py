def build_test_question_prompt(context_chunks: list[str], focus: str | None, difficulty: str) -> str:
    context_text = "\n\n".join(context_chunks)

    focus_part = f"\nFocus on: {focus}\n" if focus else ""

    difficulty_instructions = {
        "easy": "Ask a simple, basic recall or definition question.",
        "medium": "Ask a conceptual understanding or explanation question.",
        "hard": "Ask a challenging question involving application, edge cases, or deeper reasoning."
    }

    diff_text = difficulty_instructions.get(difficulty, difficulty_instructions["medium"])

    prompt = f"""
You are a tutor. Using ONLY the context below, generate ONE question.

Difficulty: {difficulty.upper()}
Instruction: {diff_text}

Rules:
- The question must be answerable from the context.
- Do NOT include the answer.
- If the context is insufficient, say: "I don't have enough information to generate a question."

{focus_part}
Context:
{context_text}

Question:
"""
    return prompt.strip()


def build_test_grader_prompt(context_chunks: list[str], question: str, user_answer: str) -> str:
    context_text = "\n\n".join(context_chunks)

    prompt = f"""
You are a strict but fair tutor. Grade the student's answer using ONLY the context below.

Context:
{context_text}

Question:
{question}

Student Answer:
{user_answer}

Instructions:
- Say whether the answer is Correct, Partially Correct, or Incorrect.
- Briefly explain why.
- Provide the correct answer if needed.
"""
    return prompt.strip()
