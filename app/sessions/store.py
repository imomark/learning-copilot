import uuid
from typing import Dict, List

class TestSession:
    def __init__(self, focus: str | None):
        self.id = str(uuid.uuid4())
        self.focus = focus
        self.total = 0
        self.correct = 0
        self.partial = 0
        self.incorrect = 0
        self.history: List[dict] = []  # each: {question, user_answer, grade}

    def record(self, question: str, user_answer: str, grade: str):
        self.total += 1
        g = grade.lower()
        if "incorrect" in g.lower():
            self.incorrect += 1
        elif "correct" in g.lower() and "partial" not in g.lower():
            self.correct += 1
        elif "partial" in g:
            self.partial += 1
        else:
            self.incorrect += 1

        self.history.append({
            "question": question,
            "user_answer": user_answer,
            "grade": grade
        })

    def summary(self):
        return {
            "session_id": self.id,
            "focus": self.focus,
            "total": self.total,
            "correct": self.correct,
            "partial": self.partial,
            "incorrect": self.incorrect
        }


class SessionStore:
    def __init__(self):
        self.sessions: Dict[str, TestSession] = {}

    def create(self, focus: str | None) -> TestSession:
        s = TestSession(focus)
        self.sessions[s.id] = s
        return s

    def get(self, session_id: str) -> TestSession | None:
        return self.sessions.get(session_id)
