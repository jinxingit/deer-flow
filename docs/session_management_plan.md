# Session Management Feature Implementation Plan

## Overview
Introduce persistent multi-session chat support backed by SQLite, enabling users to create, switch, rename, and archive sessions while continuing conversations seamlessly across refreshes.

## Milestones
1. **Database & Infrastructure**
   - Implement lightweight SQLite session/message schema and configuration (`SESSION_DB_PATH`).
   - Provide an async-friendly repository abstraction (`SessionStore`) with CRUD helpers backed by the standard library `sqlite3` + `asyncio`.
2. **Backend API Layer**
   - Define Pydantic models for session DTOs and pagination cursors.
   - Add FastAPI router (`/api/sessions`) covering list/create/get/patch/delete operations.
   - Integrate repository into application startup and dependency injection.
3. **Chat Stream Integration**
   - Validate incoming `thread_id` against active sessions.
   - Persist streaming message chunks into SQLite once responses finalise.
   - Implement async session title generation using existing LLM abstraction; enforce 20-char limit and fallbacks.
4. **Frontend State & API**
   - Extend Zustand store to track `sessions`, `activeSessionId`, and `messagesBySession`.
   - Build session sidebar UI with create/switch/rename/archive interactions.
   - Update chat sending logic to ensure `thread_id` sourced from selected session and history loading works.
5. **Quality Assurance**
   - Add backend unit/integration tests covering repositories and session API workflows (using in-memory SQLite).
   - Implement frontend tests for session list behaviour and regression coverage.
   - Update documentation (README / configuration guide) for new environment variables and usage instructions.

## Tracking
- Use this document to log progress and adjust scope as implementation proceeds.
