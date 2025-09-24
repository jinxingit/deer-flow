"""Session management package providing SQLite-backed persistence and APIs."""

from .dependencies import get_session_store
from .store import SQLiteSessionStore

__all__ = ["SQLiteSessionStore", "get_session_store"]
