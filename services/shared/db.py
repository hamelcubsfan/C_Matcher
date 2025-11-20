"""Database session and base model definitions."""
from __future__ import annotations

from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from .config import get_settings


class Base(DeclarativeBase):
    pass


_engine = None
_SessionLocal = None


def _init_engine():
    global _engine, _SessionLocal
    if _engine is None:
        settings = get_settings()
        _engine = create_engine(settings.database_url, echo=False, future=True)
        _SessionLocal = sessionmaker(bind=_engine, class_=Session, expire_on_commit=False)


def get_engine():
    if _engine is None:
        _init_engine()
    return _engine


def get_session_factory():
    if _SessionLocal is None:
        _init_engine()
    return _SessionLocal


def reset_db(session_factory: sessionmaker):
    """Reset the database schema."""
    engine = session_factory.kw["bind"]
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)


@contextmanager
def session_scope():
    factory = get_session_factory()
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
