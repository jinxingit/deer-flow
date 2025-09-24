from __future__ import annotations

import logging
from typing import Iterable, Optional

from langchain_core.messages import HumanMessage

from src.llms.llm import get_llm_by_type

from .models import MessageRecord
from .store import SQLiteSessionStore

logger = logging.getLogger(__name__)

_MAX_TITLE_LENGTH = 20


async def ensure_session_title(store: SQLiteSessionStore, session_id: str) -> Optional[str]:
    """Generate and persist a session title if one does not already exist."""
    if await store.session_has_title(session_id):
        return None

    messages = await store.get_first_exchange(session_id)
    if not messages:
        logger.debug("Session %s has no messages yet; skipping title generation", session_id)
        return None

    fallback = _derive_fallback_title(messages)

    try:
        llm = get_llm_by_type("basic")
    except Exception as exc:  # noqa: BLE001 - LLM misconfiguration should not fail request
        logger.warning("LLM unavailable for session title generation: %s", exc)
        await store.update_session_title(session_id, fallback)
        return fallback

    prompt = _build_prompt(messages)

    try:
        ai_message = await llm.ainvoke([HumanMessage(content=prompt)])
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to generate session title via LLM: %s", exc)
        await store.update_session_title(session_id, fallback)
        return fallback

    title = ai_message.content.strip() if hasattr(ai_message, "content") else str(ai_message)
    if not title:
        title = fallback

    title = _truncate_to_limit(title)
    await store.update_session_title(session_id, title)
    return title


def _build_prompt(messages: Iterable[MessageRecord]) -> str:
    pairs: list[str] = []
    for message in messages:
        if message.role == "user":
            pairs.append(f"用户：{message.content}")
        elif message.role == "assistant":
            pairs.append(f"助手：{message.content}")
    joined = "\n".join(pairs)
    return (
        "请阅读下面的对话内容，为该对话生成一个能够概括主题的中文标题。\n"
        "要求：\n"
        "1. 标题不超过20个字；\n"
        "2. 不需要引号或句号；\n"
        "3. 如果无法概括，请返回“新会话”。\n\n"
        f"对话内容：\n{joined}\n\n"
        "请输出最终标题："
    )


def _derive_fallback_title(messages: Iterable[MessageRecord]) -> str:
    for message in messages:
        if message.role == "user" and message.content:
            return _truncate_to_limit(message.content)
    return "新会话"


def _truncate_to_limit(text: str) -> str:
    cleaned = text.strip()
    if len(cleaned) <= _MAX_TITLE_LENGTH:
        return cleaned or "新会话"
    trimmed = cleaned[: _MAX_TITLE_LENGTH - 1].rstrip()
    return f"{trimmed}…"
