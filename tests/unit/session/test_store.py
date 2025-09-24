import pytest

from src.server.session.store import SQLiteSessionStore


@pytest.mark.asyncio
async def test_create_session_and_messages(tmp_path):
    db_path = tmp_path / "session_store.db"
    store = SQLiteSessionStore(str(db_path))
    await store.init()

    session = await store.create_session()
    assert session.id
    assert session.thread_id
    assert session.title is None

    stored = await store.get_session(session.id)
    assert stored is not None
    assert stored.last_message_preview is None

    await store.append_message(session_id=session.id, role="user", content="你好")
    await store.append_message(
        session_id=session.id,
        role="assistant",
        content="你好，请问有什么可以帮你？",
        reasoning_content="reasoning",
    )

    messages = await store.get_messages(session.id)
    assert len(messages) == 2
    assert messages[0].content == "你好"
    assert messages[1].reasoning_content == "reasoning"

    sessions = await store.list_sessions()
    assert len(sessions) == 1
    assert sessions[0].last_message_preview.startswith("你好")


@pytest.mark.asyncio
async def test_rename_and_archive_session(tmp_path):
    db_path = tmp_path / "session_store.db"
    store = SQLiteSessionStore(str(db_path))
    await store.init()

    session = await store.create_session()
    await store.append_message(session_id=session.id, role="user", content="test")

    renamed = await store.rename_session(session.id, "新的会话名称")
    assert renamed.title == "新的会话名称"

    await store.archive_session(session.id)
    archived = await store.get_session(session.id)
    assert archived is not None
    assert archived.archived is True

    await store.unarchive_session(session.id)
    restored = await store.get_session(session.id)
    assert restored is not None
    assert restored.archived is False

    await store.delete_session(session.id)
    assert await store.get_session(session.id) is None
