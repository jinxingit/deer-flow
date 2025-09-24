from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional
from uuid import uuid4

from .models import MessageRecord, SessionRecord

logger = logging.getLogger(__name__)


_SESSIONS_DDL = """
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    thread_id TEXT NOT NULL UNIQUE,
    title TEXT,
    summary TEXT,
    last_message_preview TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    archived INTEGER NOT NULL DEFAULT 0
);
"""

_MESSAGES_DDL = """
CREATE TABLE IF NOT EXISTS messages (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL,
    agent TEXT,
    content TEXT NOT NULL,
    reasoning_content TEXT,
    tool_calls TEXT,
    metadata TEXT,
    seq INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE
);
"""

_CREATE_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_sessions_updated ON sessions(updated_at DESC);",
    "CREATE INDEX IF NOT EXISTS idx_messages_session_seq ON messages(session_id, seq);",
]


def _ensure_pragmas(connection: sqlite3.Connection) -> None:
    connection.execute("PRAGMA foreign_keys = ON;")
    connection.execute("PRAGMA journal_mode = WAL;")


class SQLiteSessionStore:
    """SQLite-backed repository for chat sessions and messages."""

    def __init__(self, db_path: str) -> None:
        path = Path(db_path)
        if not path.is_absolute() and path.name != ":memory":
            path = Path.cwd() / path
        if path.suffix != ".db" and path.name != ":memory":
            path = path.with_suffix(".db")
        self._db_path = str(path)
        self._write_lock = asyncio.Lock()

    @property
    def db_path(self) -> str:
        return self._db_path

    async def init(self) -> None:
        """Initialise database schema."""
        db_path_obj = Path(self._db_path)
        if db_path_obj.name != ":memory":
            db_path_obj.parent.mkdir(parents=True, exist_ok=True)

        def _init() -> None:
            with sqlite3.connect(self._db_path) as connection:
                _ensure_pragmas(connection)
                connection.execute(_SESSIONS_DDL)
                connection.execute(_MESSAGES_DDL)
                for statement in _CREATE_INDEXES:
                    connection.execute(statement)
                connection.commit()

        await asyncio.to_thread(_init)
        logger.info("Session database initialised at %s", self._db_path)

    async def close(self) -> None:  # pragma: no cover - compatibility placeholder
        return None

    async def create_session(self, *, title: Optional[str] = None) -> SessionRecord:
        session_id = uuid4().hex
        thread_id = uuid4().hex
        now = _utc_now_str()

        async with self._write_lock:
            await asyncio.to_thread(
                self._execute,
                "INSERT INTO sessions (id, thread_id, title, summary, last_message_preview, created_at, updated_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                (session_id, thread_id, title, None, None, now, now),
            )

        return SessionRecord(
            id=session_id,
            thread_id=thread_id,
            title=title,
            summary=None,
            created_at=_parse_ts(now),
            updated_at=_parse_ts(now),
            archived=False,
            last_message_preview=None,
        )

    async def list_sessions(self, *, include_archived: bool = False) -> list[SessionRecord]:
        condition = "" if include_archived else "WHERE archived = 0"
        query = (
            "SELECT id, thread_id, title, summary, last_message_preview, created_at, updated_at, archived "
            f"FROM sessions {condition} ORDER BY updated_at DESC"
        )
        rows = await asyncio.to_thread(self._fetchall, query)
        return [self._row_to_session(row) for row in rows]

    async def get_session_by_thread(self, thread_id: str) -> Optional[SessionRecord]:
        row = await asyncio.to_thread(
            self._fetchone,
            "SELECT id, thread_id, title, summary, last_message_preview, created_at, updated_at, archived "
            "FROM sessions WHERE thread_id = ?",
            (thread_id,),
        )
        return self._row_to_session(row) if row else None

    async def get_session(self, session_id: str) -> Optional[SessionRecord]:
        row = await asyncio.to_thread(
            self._fetchone,
            "SELECT id, thread_id, title, summary, last_message_preview, created_at, updated_at, archived "
            "FROM sessions WHERE id = ?",
            (session_id,),
        )
        return self._row_to_session(row) if row else None

    async def get_messages(self, session_id: str) -> list[MessageRecord]:
        rows = await asyncio.to_thread(
            self._fetchall,
            "SELECT id, session_id, role, agent, content, reasoning_content, tool_calls, metadata, seq, created_at "
            "FROM messages WHERE session_id = ? ORDER BY seq ASC",
            (session_id,),
        )
        messages: list[MessageRecord] = []
        for row in rows:
            tool_calls = json.loads(row["tool_calls"]) if row["tool_calls"] else None
            metadata = json.loads(row["metadata"]) if row["metadata"] else None
            messages.append(
                MessageRecord(
                    id=row["id"],
                    session_id=row["session_id"],
                    role=row["role"],
                    agent=row["agent"],
                    content=row["content"],
                    reasoning_content=row["reasoning_content"],
                    tool_calls=tool_calls,
                    metadata=metadata,
                    seq=row["seq"],
                    created_at=_parse_ts(row["created_at"]),
                )
            )
        return messages

    async def append_message(
        self,
        *,
        session_id: str,
        role: str,
        content: str,
        agent: Optional[str] = None,
        reasoning_content: Optional[str] = None,
        tool_calls: Optional[Iterable[dict]] = None,
        metadata: Optional[dict] = None,
    ) -> MessageRecord:
        message_id = uuid4().hex
        now = _utc_now_str()
        tool_calls_json = json.dumps(list(tool_calls)) if tool_calls else None
        metadata_json = json.dumps(metadata) if metadata else None

        async with self._write_lock:
            def _insert() -> int:
                with sqlite3.connect(self._db_path) as connection:
                    connection.row_factory = sqlite3.Row
                    _ensure_pragmas(connection)

                    cursor = connection.execute(
                        "SELECT COALESCE(MAX(seq), 0) AS max_seq FROM messages WHERE session_id = ?",
                        (session_id,),
                    )
                    row = cursor.fetchone()
                    next_seq = int(row["max_seq"] or 0) + 1

                    connection.execute(
                        "INSERT INTO messages (id, session_id, role, agent, content, reasoning_content, tool_calls, metadata, seq, created_at)"
                        " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            message_id,
                            session_id,
                            role,
                            agent,
                            content,
                            reasoning_content,
                            tool_calls_json,
                            metadata_json,
                            next_seq,
                            now,
                        ),
                    )
                    preview = content[:200]
                    connection.execute(
                        "UPDATE sessions SET updated_at = ?, last_message_preview = ? WHERE id = ?",
                        (now, preview, session_id),
                    )
                    connection.commit()
                    return next_seq

            seq = await asyncio.to_thread(_insert)

        return MessageRecord(
            id=message_id,
            session_id=session_id,
            role=role,
            agent=agent,
            content=content,
            reasoning_content=reasoning_content,
            tool_calls=json.loads(tool_calls_json) if tool_calls_json else None,
            metadata=json.loads(metadata_json) if metadata_json else None,
            seq=seq,
            created_at=_parse_ts(now),
        )

    async def update_session_title(self, session_id: str, title: str) -> None:
        now = _utc_now_str()
        async with self._write_lock:
            await asyncio.to_thread(
                self._execute,
                "UPDATE sessions SET title = ?, updated_at = ? WHERE id = ?",
                (title, now, session_id),
            )

    async def rename_session(self, session_id: str, title: str) -> SessionRecord:
        await self.update_session_title(session_id, title)
        session = await self.get_session(session_id)
        if session is None:
            raise ValueError(f"Session {session_id} not found after rename")
        return session

    async def archive_session(self, session_id: str) -> None:
        async with self._write_lock:
            await asyncio.to_thread(
                self._execute,
                "UPDATE sessions SET archived = 1 WHERE id = ?",
                (session_id,),
            )

    async def unarchive_session(self, session_id: str) -> None:
        async with self._write_lock:
            await asyncio.to_thread(
                self._execute,
                "UPDATE sessions SET archived = 0 WHERE id = ?",
                (session_id,),
            )

    async def delete_session(self, session_id: str) -> None:
        async with self._write_lock:
            await asyncio.to_thread(
                self._execute,
                "DELETE FROM sessions WHERE id = ?",
                (session_id,),
            )

    async def session_has_title(self, session_id: str) -> bool:
        row = await asyncio.to_thread(
            self._fetchone,
            "SELECT title FROM sessions WHERE id = ?",
            (session_id,),
        )
        return bool(row and row["title"])

    async def get_first_exchange(self, session_id: str) -> list[MessageRecord]:
        rows = await asyncio.to_thread(
            self._fetchall,
            "SELECT id, session_id, role, agent, content, reasoning_content, tool_calls, metadata, seq, created_at "
            "FROM messages WHERE session_id = ? ORDER BY seq ASC LIMIT 4",
            (session_id,),
        )
        records: list[MessageRecord] = []
        for row in rows:
            records.append(
                MessageRecord(
                    id=row["id"],
                    session_id=row["session_id"],
                    role=row["role"],
                    agent=row["agent"],
                    content=row["content"],
                    reasoning_content=row["reasoning_content"],
                    tool_calls=json.loads(row["tool_calls"]) if row["tool_calls"] else None,
                    metadata=json.loads(row["metadata"]) if row["metadata"] else None,
                    seq=row["seq"],
                    created_at=_parse_ts(row["created_at"]),
                )
            )
        return records

    def _execute(self, query: str, params: tuple = ()) -> None:
        with sqlite3.connect(self._db_path) as connection:
            _ensure_pragmas(connection)
            connection.execute(query, params)
            connection.commit()

    def _fetchall(self, query: str, params: tuple = ()) -> list[sqlite3.Row]:
        with sqlite3.connect(self._db_path) as connection:
            connection.row_factory = sqlite3.Row
            _ensure_pragmas(connection)
            cursor = connection.execute(query, params)
            return cursor.fetchall()

    def _fetchone(self, query: str, params: tuple = ()) -> Optional[sqlite3.Row]:
        with sqlite3.connect(self._db_path) as connection:
            connection.row_factory = sqlite3.Row
            _ensure_pragmas(connection)
            cursor = connection.execute(query, params)
            return cursor.fetchone()

    @staticmethod
    def _row_to_session(row: sqlite3.Row | None) -> Optional[SessionRecord]:
        if row is None:
            return None
        return SessionRecord(
            id=row["id"],
            thread_id=row["thread_id"],
            title=row["title"],
            summary=row["summary"],
            created_at=_parse_ts(row["created_at"]),
            updated_at=_parse_ts(row["updated_at"]),
            archived=bool(row["archived"]),
            last_message_preview=row["last_message_preview"],
        )


def _utc_now_str() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _parse_ts(value: str) -> datetime:
    if value.endswith("Z"):
        value = value[:-1]
    return datetime.fromisoformat(value)
