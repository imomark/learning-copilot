from sqlalchemy import Column, Integer, String, Text, ForeignKey
from sqlalchemy.orm import relationship
from app.db import Base

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
