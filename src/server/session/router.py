from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, status

from .dependencies import get_session_store
from .models import MessageRecord, SessionRecord
from .schemas import (
    DeleteResponse,
    SessionCreateRequest,
    SessionCreateResponse,
    SessionDetail,
    SessionListResponse,
    SessionMessage,
    SessionSummary,
    SessionUpdateRequest,
)
from .store import SQLiteSessionStore

router = APIRouter(prefix="/api/sessions", tags=["sessions"])


@router.get("", response_model=SessionListResponse)
async def list_sessions(
    include_archived: bool = Query(default=False, description="Include archived sessions."),
    store: SQLiteSessionStore = Depends(get_session_store),
) -> SessionListResponse:
    records = await store.list_sessions(include_archived=include_archived)
    return SessionListResponse(sessions=[_to_summary(record) for record in records])


@router.post("", status_code=status.HTTP_201_CREATED, response_model=SessionCreateResponse)
async def create_session(
    payload: SessionCreateRequest,
    store: SQLiteSessionStore = Depends(get_session_store),
) -> SessionCreateResponse:
    session = await store.create_session()
    if payload.initial_message:
        await store.append_message(
            session_id=session.id,
            role="user",
            content=payload.initial_message.strip(),
        )
        session = await store.get_session(session.id) or session
    messages = await store.get_messages(session.id)
    return SessionCreateResponse(session=_to_detail(session, messages))


@router.get("/{session_id}", response_model=SessionDetail)
async def get_session(
    session_id: str,
    store: SQLiteSessionStore = Depends(get_session_store),
) -> SessionDetail:
    session = await store.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")
    messages = await store.get_messages(session_id)
    return _to_detail(session, messages)


@router.patch("/{session_id}", response_model=SessionDetail)
async def update_session(
    session_id: str,
    payload: SessionUpdateRequest,
    store: SQLiteSessionStore = Depends(get_session_store),
) -> SessionDetail:
    session = await store.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")

    if payload.title is not None:
        session = await store.rename_session(session_id, payload.title)

    if payload.archived is not None:
        if payload.archived:
            await store.archive_session(session_id)
        else:
            await store.unarchive_session(session_id)
        session = await store.get_session(session_id) or session

    messages = await store.get_messages(session_id)
    return _to_detail(session, messages)


@router.delete("/{session_id}", response_model=DeleteResponse)
async def delete_session(
    session_id: str,
    store: SQLiteSessionStore = Depends(get_session_store),
) -> DeleteResponse:
    session = await store.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")
    await store.delete_session(session_id)
    return DeleteResponse(success=True)


def _to_summary(record: SessionRecord) -> SessionSummary:
    return SessionSummary(
        id=record.id,
        thread_id=record.thread_id,
        title=record.title,
        summary=record.summary,
        last_message_preview=record.last_message_preview,
        updated_at=record.updated_at,
        created_at=record.created_at,
        archived=record.archived,
    )


def _to_message(record: MessageRecord) -> SessionMessage:
    return SessionMessage(
        id=record.id,
        role=record.role,
        agent=record.agent,
        content=record.content,
        reasoning_content=record.reasoning_content,
        tool_calls=record.tool_calls,
        metadata=record.metadata,
        seq=record.seq,
        created_at=record.created_at,
    )


def _to_detail(session: SessionRecord, messages: list[MessageRecord]) -> SessionDetail:
    return SessionDetail(
        **_to_summary(session).model_dump(),
        messages=[_to_message(message) for message in messages],
    )
