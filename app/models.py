from sqlalchemy import Column, Integer, String, Text, ForeignKey
from sqlalchemy.orm import relationship
from app.db import Base
from sqlalchemy import DateTime, Float
from datetime import datetime

class ReviewScheduleModel(Base):
    __tablename__ = "review_schedules"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, index=True)
    topic = Column(String, index=True)

    interval_days = Column(Integer, default=1)
    ease_factor = Column(Float, default=2.5)
    next_review_at = Column(DateTime, default=datetime.utcnow)

class TestSessionModel(Base):
    __tablename__ = "test_sessions"

    id = Column(String, primary_key=True, index=True)
    focus = Column(String, nullable=True)
    total = Column(Integer, default=0)
    correct = Column(Integer, default=0)
    partial = Column(Integer, default=0)
    incorrect = Column(Integer, default=0)

    attempts = relationship("AttemptModel", back_populates="session")


class AttemptModel(Base):
    __tablename__ = "attempts"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, ForeignKey("test_sessions.id"))
    question = Column(Text)
    user_answer = Column(Text)
    grade = Column(String)
    topic = Column(String)

    session = relationship("TestSessionModel", back_populates="attempts")
