import uuid
from sqlalchemy.orm import Session
from app.db import SessionLocal
from app.models import TestSessionModel, AttemptModel
from datetime import datetime, timedelta
from app.models import ReviewScheduleModel


class DBSessionStore:
    def create(self, focus: str | None):
        db: Session = SessionLocal()
        try:
            sid = str(uuid.uuid4())
            s = TestSessionModel(
                id=sid, focus=focus, total=0, correct=0, partial=0, incorrect=0
            )
            db.add(s)
            db.commit()
            db.refresh(s)
            return s
        finally:
            db.close()

    def get(self, session_id: str):
        db: Session = SessionLocal()
        try:
            return db.query(TestSessionModel).filter(TestSessionModel.id == session_id).first()
        finally:
            db.close()

    def record_attempt(self, session_id: str, question: str, user_answer: str, grade: str, topic: str):
        db: Session = SessionLocal()
        try:
            s = db.query(TestSessionModel).filter(TestSessionModel.id == session_id).first()
            if not s:
                return None

            g = grade.lower()
            if "correct" in g and "partial" not in g:
                s.correct += 1
            elif "partial" in g:
                s.partial += 1
            else:
                s.incorrect += 1
            s.total += 1

            a = AttemptModel(
                session_id=session_id,
                question=question,
                user_answer=user_answer,
                grade=grade,
                topic=topic,
            )
            db.add(a)
            db.commit()
            db.refresh(s)
            return s
        finally:
            db.close()

    def summary(self, session_id: str):
        db: Session = SessionLocal()
        try:
            s = db.query(TestSessionModel).filter(TestSessionModel.id == session_id).first()
            if not s:
                return None
            return {
                "session_id": s.id,
                "focus": s.focus,
                "total": s.total,
                "correct": s.correct,
                "partial": s.partial,
                "incorrect": s.incorrect,
            }
        finally:
            db.close()

    def weak_areas(self, session_id: str):
        db: Session = SessionLocal()
        try:
            attempts = db.query(AttemptModel).filter(AttemptModel.session_id == session_id).all()
            stats = {}
            for a in attempts:
                t = a.topic or "general"
                if t not in stats:
                    stats[t] = {"correct": 0, "partial": 0, "incorrect": 0}
                g = a.grade.lower()
                if "correct" in g and "partial" not in g:
                    stats[t]["correct"] += 1
                elif "partial" in g:
                    stats[t]["partial"] += 1
                else:
                    stats[t]["incorrect"] += 1

            ranked = []
            for topic, s in stats.items():
                weakness = s["incorrect"] + s["partial"]
                ranked.append({"topic": topic, "stats": s, "weakness_score": weakness})

            ranked.sort(key=lambda x: x["weakness_score"], reverse=True)
            return ranked
        finally:
            db.close()

    def topic_difficulty(self, session_id: str, topic: str | None):
        # Default difficulty
        if not topic:
            return "medium"

        db: Session = SessionLocal()
        try:
            from app.models import AttemptModel
            attempts = (
                db.query(AttemptModel)
                .filter(AttemptModel.session_id == session_id)
                .filter(AttemptModel.topic == topic)
                .all()
            )

            if not attempts:
                return "medium"

            correct = 0
            partial = 0
            incorrect = 0

            for a in attempts:
                g = a.grade.lower()
                if "correct" in g and "partial" not in g:
                    correct += 1
                elif "partial" in g:
                    partial += 1
                else:
                    incorrect += 1

            strength = correct - incorrect - 0.5 * partial

            if strength <= -1:
                return "easy"
            elif strength >= 2:
                return "hard"
            else:
                return "medium"
        finally:
            db.close()
    def update_review_schedule(self, session_id: str, topic: str, grade: str):
        db: Session = SessionLocal()
        try:
            sched = (
                db.query(ReviewScheduleModel)
                .filter(ReviewScheduleModel.session_id == session_id)
                .filter(ReviewScheduleModel.topic == topic)
                .first()
            )

            if not sched:
                sched = ReviewScheduleModel(
                    session_id=session_id,
                    topic=topic,
                    interval_days=1,
                    ease_factor=2.5,
                    next_review_at=datetime.utcnow(),
                )
                db.add(sched)
                db.commit()
                db.refresh(sched)

            g = grade.lower()

            if "correct" in g and "partial" not in g:
                # successful recall → increase interval
                sched.interval_days = int(sched.interval_days * sched.ease_factor)
                sched.ease_factor = min(sched.ease_factor + 0.1, 3.0)
            elif "partial" in g:
                # small progress
                sched.interval_days = max(1, int(sched.interval_days * 1.2))
            else:
                # failed recall → reset
                sched.interval_days = 1
                sched.ease_factor = max(1.3, sched.ease_factor - 0.2)

            sched.next_review_at = datetime.utcnow() + timedelta(days=sched.interval_days)

            db.commit()
        finally:
            db.close()

    def due_topics(self, session_id: str):
        db: Session = SessionLocal()
        try:
            now = datetime.utcnow()
            rows = (
                db.query(ReviewScheduleModel)
                .filter(ReviewScheduleModel.session_id == session_id)
                .filter(ReviewScheduleModel.next_review_at <= now)
                .all()
            )
            return [r.topic for r in rows]
        finally:
            db.close()

