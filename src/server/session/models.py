from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional


@dataclass(slots=True)
class SessionRecord:
    id: str
    thread_id: str
    title: Optional[str]
    summary: Optional[str]
    created_at: datetime
    updated_at: datetime
    archived: bool
    last_message_preview: Optional[str]


@dataclass(slots=True)
class MessageRecord:
    id: str
    session_id: str
    role: str
    agent: Optional[str]
    content: str
    reasoning_content: Optional[str]
    tool_calls: Optional[list[dict[str, Any]]]
    metadata: Optional[dict[str, Any]]
    seq: int
    created_at: datetime
