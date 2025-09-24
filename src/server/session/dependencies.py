from __future__ import annotations

import logging
from typing import Optional

from fastapi import Depends

from src.config.loader import get_str_env

from .store import SQLiteSessionStore

logger = logging.getLogger(__name__)

_SESSION_STORE: Optional[SQLiteSessionStore] = None


def initialise_session_store() -> SQLiteSessionStore:
    """Create session store instance using configuration."""
    global _SESSION_STORE
    if _SESSION_STORE is not None:
        return _SESSION_STORE

    db_path = get_str_env("SESSION_DB_PATH", "deerflow.db")
    store = SQLiteSessionStore(db_path)
    _SESSION_STORE = store
    logger.info("Initialised session store with DB path %s", store.db_path)
    return store


def set_session_store(store: SQLiteSessionStore) -> None:
    global _SESSION_STORE
    _SESSION_STORE = store


def get_session_store(_: SQLiteSessionStore = Depends(initialise_session_store)) -> SQLiteSessionStore:
    if _SESSION_STORE is None:
        raise RuntimeError("Session store has not been initialised")
    return _SESSION_STORE
