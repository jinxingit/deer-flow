from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator


class SessionMessage(BaseModel):
    id: str
    role: str
    agent: Optional[str] = None
    content: str
    reasoning_content: Optional[str] = None
    tool_calls: Optional[list[dict[str, Any]]] = None
    metadata: Optional[dict[str, Any]] = None
    seq: int
    created_at: datetime


class SessionSummary(BaseModel):
    id: str
    thread_id: str
    title: Optional[str] = None
    summary: Optional[str] = None
    last_message_preview: Optional[str] = None
    updated_at: datetime
    created_at: datetime
    archived: bool = False


class SessionDetail(SessionSummary):
    messages: list[SessionMessage] = Field(default_factory=list)


class SessionCreateRequest(BaseModel):
    initial_message: Optional[str] = Field(
        default=None,
        description="Optional initial user message to seed the session.",
    )


class SessionCreateResponse(BaseModel):
    session: SessionDetail


class SessionListResponse(BaseModel):
    sessions: list[SessionSummary]


class SessionUpdateRequest(BaseModel):
    title: Optional[str] = Field(default=None, description="Manual session title override.")
    archived: Optional[bool] = Field(default=None, description="Archive/unarchive session.")

    @field_validator("title")
    @classmethod
    def validate_title_length(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return value
        value = value.strip()
        if not value:
            return None
        if len(value) > 40:
            raise ValueError("Title must be 40 characters or fewer")
        return value


class DeleteResponse(BaseModel):
    success: bool
