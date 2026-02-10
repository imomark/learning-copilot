# app/sessions/store.py
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
        self.history: List[dict] = []
        # topic -> {correct, partial, incorrect}
        self.topic_stats: Dict[str, Dict[str, int]] = {}

    def _ensure_topic(self, topic: str):
        if topic not in self.topic_stats:
            self.topic_stats[topic] = {"correct": 0, "partial": 0, "incorrect": 0}

    def record(self, question: str, user_answer: str, grade: str, topic: str):
        self.total += 1
        g = grade.lower()

        if "incorrect" in g.lower():
            self.incorrect += 1
            outcome = "incorrect"
        elif "correct" in g.lower() and "partial" not in g.lower():
            self.correct += 1
            outcome = "correct"
        elif "partial" in g:
            self.partial += 1
            outcome = "partial"
        else:
            self.incorrect += 1
            outcome = "incorrect"

        self._ensure_topic(topic)
        self.topic_stats[topic][outcome] += 1

        self.history.append({
            "question": question,
            "user_answer": user_answer,
            "grade": grade,
            "topic": topic
        })

    def summary(self):
        return {
            "session_id": self.id,
            "focus": self.focus,
            "total": self.total,
            "correct": self.correct,
            "partial": self.partial,
            "incorrect": self.incorrect,
        }

    def weak_areas(self):
        # Rank topics by (incorrect + partial) descending
        scored = []
        for topic, stats in self.topic_stats.items():
            weakness = stats["incorrect"] + stats["partial"]
            scored.append({
                "topic": topic,
                "stats": stats,
                "weakness_score": weakness
            })
        scored.sort(key=lambda x: x["weakness_score"], reverse=True)
        return scored


class SessionStore:
    def __init__(self):
        self.sessions: Dict[str, TestSession] = {}

    def create(self, focus: str | None) -> TestSession:
        s = TestSession(focus)
        self.sessions[s.id] = s
        return s

    def get(self, session_id: str) -> TestSession | None:
        return self.sessions.get(session_id)
