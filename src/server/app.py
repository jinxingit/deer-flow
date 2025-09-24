# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

import asyncio
import base64
import json
import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Annotated, Any, Dict, Iterable, List, Optional, cast
from uuid import uuid4

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from langchain_core.messages import AIMessageChunk, BaseMessage, ToolMessage
from langgraph.checkpoint.mongodb import AsyncMongoDBSaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command
from psycopg_pool import AsyncConnectionPool

from src.config.configuration import get_recursion_limit
from src.config.loader import get_bool_env, get_str_env
from src.config.report_style import ReportStyle
from src.config.tools import SELECTED_RAG_PROVIDER
from src.graph.builder import build_graph_with_memory
from src.graph.checkpoint import chat_stream_message
from src.llms.llm import get_configured_llm_models
from src.podcast.graph.builder import build_graph as build_podcast_graph
from src.ppt.graph.builder import build_graph as build_ppt_graph
from src.prompt_enhancer.graph.builder import build_graph as build_prompt_enhancer_graph
from src.prose.graph.builder import build_graph as build_prose_graph
from src.rag.builder import build_retriever
from src.rag.milvus import load_examples
from src.rag.retriever import Resource
from src.server.chat_request import (
    ChatRequest,
    EnhancePromptRequest,
    GeneratePodcastRequest,
    GeneratePPTRequest,
    GenerateProseRequest,
    TTSRequest,
)
from src.server.config_request import ConfigResponse
from src.server.mcp_request import MCPServerMetadataRequest, MCPServerMetadataResponse
from src.server.mcp_utils import load_mcp_tools
from src.server.rag_request import (  # type: ignore[import-untyped]
    RAGConfigResponse,
    RAGResourceRequest,
    RAGResourcesResponse,
)
from src.server.session.dependencies import (
    get_session_store,
    initialise_session_store,
    set_session_store,
)
from src.server.session.router import router as session_router
from src.server.session.store import SQLiteSessionStore
from src.server.session.title import ensure_session_title
from src.tools import VolcengineTTS
from src.utils.json_utils import sanitize_args

logger = logging.getLogger(__name__)

INTERNAL_SERVER_ERROR_DETAIL = "Internal Server Error"


@asynccontextmanager
async def lifespan(_: FastAPI):
    session_store = initialise_session_store()
    await session_store.init()
    set_session_store(session_store)
    try:
        yield
    finally:
        await session_store.close()


app = FastAPI(
    title="DeerFlow API",
    description="API for Deer",
    version="0.1.0",
    lifespan=lifespan,
)

# Add CORS middleware
# It's recommended to load the allowed origins from an environment variable
# for better security and flexibility across different environments.
allowed_origins_str = get_str_env("ALLOWED_ORIGINS", "http://localhost:3000")
allowed_origins = [origin.strip() for origin in allowed_origins_str.split(",")]

logger.info(f"Allowed origins: {allowed_origins}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,  # Restrict to specific origins
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],  # Use the configured list of methods
    allow_headers=["*"],  # Now allow all headers, but can be restricted further
)

app.include_router(session_router)

# Load examples into Milvus if configured
load_examples()

in_memory_store = InMemoryStore()
graph = build_graph_with_memory()


@dataclass(slots=True)
class _AssistantMessageBuilder:
    agent: Optional[str]
    content_chunks: List[str] = field(default_factory=list)
    reasoning_chunks: List[str] = field(default_factory=list)
    tool_calls: Optional[List[dict[str, Any]]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class StreamPersistenceContext:
    """Accumulates streaming events and persists final messages to the session store."""

    def __init__(self, store: SQLiteSessionStore, session_id: str) -> None:
        self._store = store
        self._session_id = session_id
        self._builders: Dict[str, _AssistantMessageBuilder] = {}
        self._lock = asyncio.Lock()
        self._assistant_written = False

    async def handle_ai_event(self, event: Dict[str, Any]) -> None:
        message_id = event.get("id")
        if not message_id:
            return

        finalize: Optional[_AssistantMessageBuilder] = None
        async with self._lock:
            builder = self._builders.get(message_id)
            if builder is None:
                builder = _AssistantMessageBuilder(agent=event.get("agent"))
                self._builders[message_id] = builder

            if event.get("agent"):
                builder.agent = event["agent"]

            content = event.get("content")
            if content:
                builder.content_chunks.append(str(content))

            reasoning = event.get("reasoning_content")
            if reasoning:
                builder.reasoning_chunks.append(str(reasoning))

            tool_calls = event.get("tool_calls")
            if tool_calls:
                builder.tool_calls = _serialize_tool_calls(tool_calls)

            metadata = {
                key: event.get(key)
                for key in ("langgraph_node", "langgraph_path", "langgraph_step", "checkpoint_ns")
                if event.get(key)
            }
            if metadata:
                builder.metadata.update(metadata)

            finish_reason = event.get("finish_reason")
            if finish_reason in {"stop", "interrupt"}:
                builder.metadata.setdefault("finish_reason", finish_reason)
                finalize = self._builders.pop(message_id, None)
                if finalize is None:
                    finalize = builder

        if finalize is not None:
            await self._persist_builder(finalize)

    async def handle_tool_result(self, event: Dict[str, Any]) -> None:
        content = event.get("content")
        if content is None:
            return
        metadata = {
            key: event.get(key)
            for key in ("tool_call_id", "langgraph_node", "langgraph_path", "langgraph_step")
            if event.get(key)
        }
        await self._store.append_message(
            session_id=self._session_id,
            role="tool",
            agent=event.get("agent"),
            content=str(content),
            metadata=metadata or None,
        )

    async def handle_interrupt(self, event: Dict[str, Any]) -> None:
        content = event.get("content")
        if not content:
            return
        metadata = {
            key: event.get(key)
            for key in ("finish_reason", "options")
            if event.get(key)
        }
        await self._store.append_message(
            session_id=self._session_id,
            role="assistant",
            agent=event.get("agent"),
            content=str(content),
            metadata=metadata or None,
        )
        self._assistant_written = True

    async def finalize(self) -> None:
        async with self._lock:
            pending = list(self._builders.values())
            self._builders.clear()
        for builder in pending:
            await self._persist_builder(builder)
        if self._assistant_written:
            await ensure_session_title(self._store, self._session_id)

    async def _persist_builder(self, builder: _AssistantMessageBuilder) -> None:
        content = "".join(builder.content_chunks).strip()
        reasoning = "".join(builder.reasoning_chunks).strip() or None
        await self._store.append_message(
            session_id=self._session_id,
            role="assistant",
            agent=builder.agent,
            content=content,
            reasoning_content=reasoning,
            tool_calls=builder.tool_calls,
            metadata=builder.metadata or None,
        )
        self._assistant_written = True


def _serialize_tool_calls(tool_calls: Iterable[Any]) -> list[dict[str, Any]]:
    serializable: list[dict[str, Any]] = []
    for call in tool_calls:
        if isinstance(call, dict):
            data = dict(call)
            if "args" in data:
                data["args"] = sanitize_args(data["args"])
            serializable.append(data)
        else:
            data = {
                "id": getattr(call, "id", ""),
                "name": getattr(call, "name", ""),
                "args": sanitize_args(getattr(call, "args", {})),
            }
            serializable.append(data)
    return serializable


@app.post("/api/chat/stream")
async def chat_stream(
    request: ChatRequest,
    session_store: SQLiteSessionStore = Depends(get_session_store),
):
    # Check if MCP server configuration is enabled
    mcp_enabled = get_bool_env("ENABLE_MCP_SERVER_CONFIGURATION", False)

    # Validate MCP settings if provided
    if request.mcp_settings and not mcp_enabled:
        raise HTTPException(
            status_code=403,
            detail="MCP server configuration is disabled. Set ENABLE_MCP_SERVER_CONFIGURATION=true to enable MCP features.",
        )

    thread_id = request.thread_id
    if not thread_id:
        raise HTTPException(status_code=400, detail="thread_id is required")

    session = await session_store.get_session_by_thread(thread_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    if session.archived:
        raise HTTPException(status_code=403, detail="Session has been archived")

    incoming_messages = request.messages or []
    if incoming_messages:
        last_message = incoming_messages[-1]
        if last_message.role == "user":
            content = _stringify_message_content(last_message.content)
            if content:
                await session_store.append_message(
                    session_id=session.id,
                    role="user",
                    content=content,
                )

    persistence = StreamPersistenceContext(session_store, session.id)
    request_payload = request.model_dump()

    return StreamingResponse(
        _astream_workflow_generator(
            request_payload["messages"],
            thread_id,
            request.resources,
            request.max_plan_iterations,
            request.max_step_num,
            request.max_search_results,
            request.auto_accepted_plan,
            request.interrupt_feedback,
            request.mcp_settings if mcp_enabled else {},
            request.enable_background_investigation,
            request.report_style,
            request.enable_deep_thinking,
            persistence,
        ),
        media_type="text/event-stream",
    )


def _process_tool_call_chunks(tool_call_chunks):
    """Process tool call chunks and sanitize arguments."""
    chunks = []
    for chunk in tool_call_chunks:
        chunks.append(
            {
                "name": chunk.get("name", ""),
                "args": sanitize_args(chunk.get("args", "")),
                "id": chunk.get("id", ""),
                "index": chunk.get("index", 0),
                "type": chunk.get("type", ""),
            }
        )
    return chunks


def _get_agent_name(agent, message_metadata):
    """Extract agent name from agent tuple."""
    agent_name = "unknown"
    if agent and len(agent) > 0:
        agent_name = agent[0].split(":")[0] if ":" in agent[0] else agent[0]
    else:
        agent_name = message_metadata.get("langgraph_node", "unknown")
    return agent_name


def _create_event_stream_message(
    message_chunk, message_metadata, thread_id, agent_name
):
    """Create base event stream message."""
    event_stream_message = {
        "thread_id": thread_id,
        "agent": agent_name,
        "id": message_chunk.id,
        "role": "assistant",
        "checkpoint_ns": message_metadata.get("checkpoint_ns", ""),
        "langgraph_node": message_metadata.get("langgraph_node", ""),
        "langgraph_path": message_metadata.get("langgraph_path", ""),
        "langgraph_step": message_metadata.get("langgraph_step", ""),
        "content": message_chunk.content,
    }

    # Add optional fields
    if message_chunk.additional_kwargs.get("reasoning_content"):
        event_stream_message["reasoning_content"] = message_chunk.additional_kwargs[
            "reasoning_content"
        ]

    if message_chunk.response_metadata.get("finish_reason"):
        event_stream_message["finish_reason"] = message_chunk.response_metadata.get(
            "finish_reason"
        )

    return event_stream_message


async def _create_interrupt_event(
    thread_id, event_data, persistence: Optional[StreamPersistenceContext] = None
):
    """Create interrupt event."""
    payload = {
        "thread_id": thread_id,
        "id": event_data["__interrupt__"][0].ns[0],
        "role": "assistant",
        "content": event_data["__interrupt__"][0].value,
        "finish_reason": "interrupt",
        "options": [
            {"text": "Edit plan", "value": "edit_plan"},
            {"text": "Start research", "value": "accepted"},
        ],
    }
    if persistence:
        await persistence.handle_interrupt(payload)
    return _make_event("interrupt", payload)


def _process_initial_messages(message, thread_id):
    """Process initial messages and yield formatted events."""
    json_data = json.dumps(
        {
            "thread_id": thread_id,
            "id": "run--" + message.get("id", uuid4().hex),
            "role": "user",
            "content": message.get("content", ""),
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )
    chat_stream_message(
        thread_id, f"event: message_chunk\ndata: {json_data}\n\n", "none"
    )


def _stringify_message_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            item_type = getattr(item, "type", None) or item.get("type") if isinstance(item, dict) else None
            if item_type == "text":
                text = getattr(item, "text", None) if not isinstance(item, dict) else item.get("text")
                if text:
                    parts.append(str(text))
            elif item_type == "image":
                url = getattr(item, "image_url", None) if not isinstance(item, dict) else item.get("image_url")
                if url:
                    parts.append(f"[image:{url}]")
        return "\n".join(parts)
    try:
        return json.dumps(content, ensure_ascii=False)
    except TypeError:
        return str(content)


async def _process_message_chunk(
    message_chunk,
    message_metadata,
    thread_id,
    agent,
    persistence: Optional[StreamPersistenceContext] = None,
):
    """Process a single message chunk and yield appropriate events."""
    agent_name = _get_agent_name(agent, message_metadata)
    event_stream_message = _create_event_stream_message(
        message_chunk, message_metadata, thread_id, agent_name
    )

    if isinstance(message_chunk, ToolMessage):
        # Tool Message - Return the result of the tool call
        event_stream_message["tool_call_id"] = message_chunk.tool_call_id
        event_stream_message["role"] = "tool"
        if persistence:
            await persistence.handle_tool_result(event_stream_message)
        yield _make_event("tool_call_result", event_stream_message)
    elif isinstance(message_chunk, AIMessageChunk):
        # AI Message - Raw message tokens
        if message_chunk.tool_calls:
            # AI Message - Tool Call
            event_stream_message["tool_calls"] = message_chunk.tool_calls
            event_stream_message["tool_call_chunks"] = _process_tool_call_chunks(
                message_chunk.tool_call_chunks
            )
            if persistence:
                await persistence.handle_ai_event(event_stream_message)
            yield _make_event("tool_calls", event_stream_message)
        elif message_chunk.tool_call_chunks:
            # AI Message - Tool Call Chunks
            event_stream_message["tool_call_chunks"] = _process_tool_call_chunks(
                message_chunk.tool_call_chunks
            )
            if persistence:
                await persistence.handle_ai_event(event_stream_message)
            yield _make_event("tool_call_chunks", event_stream_message)
        else:
            # AI Message - Raw message tokens
            if persistence:
                await persistence.handle_ai_event(event_stream_message)
            yield _make_event("message_chunk", event_stream_message)


async def _stream_graph_events(
    graph_instance,
    workflow_input,
    workflow_config,
    thread_id,
    persistence: Optional[StreamPersistenceContext] = None,
):
    """Stream events from the graph and process them."""
    try:
        async for agent, _, event_data in graph_instance.astream(
            workflow_input,
            config=workflow_config,
            stream_mode=["messages", "updates"],
            subgraphs=True,
        ):
            if isinstance(event_data, dict):
                if "__interrupt__" in event_data:
                    yield await _create_interrupt_event(
                        thread_id, event_data, persistence
                    )
                continue

            message_chunk, message_metadata = cast(
                tuple[BaseMessage, dict[str, Any]], event_data
            )

            async for event in _process_message_chunk(
                message_chunk,
                message_metadata,
                thread_id,
                agent,
                persistence,
            ):
                yield event
    except Exception as e:
        logger.exception("Error during graph execution")
        yield _make_event(
            "error",
            {
                "thread_id": thread_id,
                "error": "Error during graph execution",
            },
        )



async def _astream_workflow_generator(
    messages: List[dict],
    thread_id: str,
    resources: List[Resource],
    max_plan_iterations: int,
    max_step_num: int,
    max_search_results: int,
    auto_accepted_plan: bool,
    interrupt_feedback: str,
    mcp_settings: dict,
    enable_background_investigation: bool,
    report_style: ReportStyle,
    enable_deep_thinking: bool,
    persistence: StreamPersistenceContext,
):
    # Process initial messages
    for message in messages:
        if isinstance(message, dict) and "content" in message:
            _process_initial_messages(message, thread_id)

    # Prepare workflow input
    workflow_input = {
        "messages": messages,
        "plan_iterations": 0,
        "final_report": "",
        "current_plan": None,
        "observations": [],
        "auto_accepted_plan": auto_accepted_plan,
        "enable_background_investigation": enable_background_investigation,
        "research_topic": messages[-1]["content"] if messages else "",
    }

    if not auto_accepted_plan and interrupt_feedback:
        resume_msg = f"[{interrupt_feedback}]"
        if messages:
            resume_msg += f" {messages[-1]['content']}"
        workflow_input = Command(resume=resume_msg)

    # Prepare workflow config
    workflow_config = {
        "thread_id": thread_id,
        "resources": resources,
        "max_plan_iterations": max_plan_iterations,
        "max_step_num": max_step_num,
        "max_search_results": max_search_results,
        "mcp_settings": mcp_settings,
        "report_style": report_style.value,
        "enable_deep_thinking": enable_deep_thinking,
        "recursion_limit": get_recursion_limit(),
    }

    checkpoint_saver = get_bool_env("LANGGRAPH_CHECKPOINT_SAVER", False)
    checkpoint_url = get_str_env("LANGGRAPH_CHECKPOINT_DB_URL", "")
    # Handle checkpointer if configured
    connection_kwargs = {
        "autocommit": True,
        "row_factory": "dict_row",
        "prepare_threshold": 0,
    }
    try:
        if checkpoint_saver and checkpoint_url != "":
            if checkpoint_url.startswith("postgresql://"):
                logger.info("start async postgres checkpointer.")
                async with AsyncConnectionPool(
                    checkpoint_url, kwargs=connection_kwargs
                ) as conn:
                    checkpointer = AsyncPostgresSaver(conn)
                    await checkpointer.setup()
                    graph.checkpointer = checkpointer
                    graph.store = in_memory_store
                    async for event in _stream_graph_events(
                        graph,
                        workflow_input,
                        workflow_config,
                        thread_id,
                        persistence,
                    ):
                        yield event

            if checkpoint_url.startswith("mongodb://"):
                logger.info("start async mongodb checkpointer.")
                async with AsyncMongoDBSaver.from_conn_string(
                    checkpoint_url
                ) as checkpointer:
                    graph.checkpointer = checkpointer
                    graph.store = in_memory_store
                    async for event in _stream_graph_events(
                        graph,
                        workflow_input,
                        workflow_config,
                        thread_id,
                        persistence,
                    ):
                        yield event
        else:
            # Use graph without MongoDB checkpointer
            async for event in _stream_graph_events(
                graph, workflow_input, workflow_config, thread_id, persistence
            ):
                yield event
    finally:
        await persistence.finalize()


def _make_event(event_type: str, data: dict[str, any]):
    if data.get("content") == "":
        data.pop("content")
    # Ensure JSON serialization with proper encoding
    try:
        json_data = json.dumps(data, ensure_ascii=False)

        finish_reason = data.get("finish_reason", "")
        chat_stream_message(
            data.get("thread_id", ""),
            f"event: {event_type}\ndata: {json_data}\n\n",
            finish_reason,
        )

        return f"event: {event_type}\ndata: {json_data}\n\n"
    except (TypeError, ValueError) as e:
        logger.error(f"Error serializing event data: {e}")
        # Return a safe error event
        error_data = json.dumps({"error": "Serialization failed"}, ensure_ascii=False)
        return f"event: error\ndata: {error_data}\n\n"


@app.post("/api/tts")
async def text_to_speech(request: TTSRequest):
    """Convert text to speech using volcengine TTS API."""
    app_id = get_str_env("VOLCENGINE_TTS_APPID", "")
    if not app_id:
        raise HTTPException(status_code=400, detail="VOLCENGINE_TTS_APPID is not set")
    access_token = get_str_env("VOLCENGINE_TTS_ACCESS_TOKEN", "")
    if not access_token:
        raise HTTPException(
            status_code=400, detail="VOLCENGINE_TTS_ACCESS_TOKEN is not set"
        )

    try:
        cluster = get_str_env("VOLCENGINE_TTS_CLUSTER", "volcano_tts")
        voice_type = get_str_env("VOLCENGINE_TTS_VOICE_TYPE", "BV700_V2_streaming")

        tts_client = VolcengineTTS(
            appid=app_id,
            access_token=access_token,
            cluster=cluster,
            voice_type=voice_type,
        )
        # Call the TTS API
        result = tts_client.text_to_speech(
            text=request.text[:1024],
            encoding=request.encoding,
            speed_ratio=request.speed_ratio,
            volume_ratio=request.volume_ratio,
            pitch_ratio=request.pitch_ratio,
            text_type=request.text_type,
            with_frontend=request.with_frontend,
            frontend_type=request.frontend_type,
        )

        if not result["success"]:
            raise HTTPException(status_code=500, detail=str(result["error"]))

        # Decode the base64 audio data
        audio_data = base64.b64decode(result["audio_data"])

        # Return the audio file
        return Response(
            content=audio_data,
            media_type=f"audio/{request.encoding}",
            headers={
                "Content-Disposition": (
                    f"attachment; filename=tts_output.{request.encoding}"
                )
            },
        )

    except Exception as e:
        logger.exception(f"Error in TTS endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=INTERNAL_SERVER_ERROR_DETAIL)


@app.post("/api/podcast/generate")
async def generate_podcast(request: GeneratePodcastRequest):
    try:
        report_content = request.content
        print(report_content)
        workflow = build_podcast_graph()
        final_state = workflow.invoke({"input": report_content})
        audio_bytes = final_state["output"]
        return Response(content=audio_bytes, media_type="audio/mp3")
    except Exception as e:
        logger.exception(f"Error occurred during podcast generation: {str(e)}")
        raise HTTPException(status_code=500, detail=INTERNAL_SERVER_ERROR_DETAIL)


@app.post("/api/ppt/generate")
async def generate_ppt(request: GeneratePPTRequest):
    try:
        report_content = request.content
        print(report_content)
        workflow = build_ppt_graph()
        final_state = workflow.invoke({"input": report_content})
        generated_file_path = final_state["generated_file_path"]
        with open(generated_file_path, "rb") as f:
            ppt_bytes = f.read()
        return Response(
            content=ppt_bytes,
            media_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
        )
    except Exception as e:
        logger.exception(f"Error occurred during ppt generation: {str(e)}")
        raise HTTPException(status_code=500, detail=INTERNAL_SERVER_ERROR_DETAIL)


@app.post("/api/prose/generate")
async def generate_prose(request: GenerateProseRequest):
    try:
        sanitized_prompt = request.prompt.replace("\r\n", "").replace("\n", "")
        logger.info(f"Generating prose for prompt: {sanitized_prompt}")
        workflow = build_prose_graph()
        events = workflow.astream(
            {
                "content": request.prompt,
                "option": request.option,
                "command": request.command,
            },
            stream_mode="messages",
            subgraphs=True,
        )
        return StreamingResponse(
            (f"data: {event[0].content}\n\n" async for _, event in events),
            media_type="text/event-stream",
        )
    except Exception as e:
        logger.exception(f"Error occurred during prose generation: {str(e)}")
        raise HTTPException(status_code=500, detail=INTERNAL_SERVER_ERROR_DETAIL)


@app.post("/api/prompt/enhance")
async def enhance_prompt(request: EnhancePromptRequest):
    try:
        sanitized_prompt = request.prompt.replace("\r\n", "").replace("\n", "")
        logger.info(f"Enhancing prompt: {sanitized_prompt}")

        # Convert string report_style to ReportStyle enum
        report_style = None
        if request.report_style:
            try:
                # Handle both uppercase and lowercase input
                style_mapping = {
                    "ACADEMIC": ReportStyle.ACADEMIC,
                    "POPULAR_SCIENCE": ReportStyle.POPULAR_SCIENCE,
                    "NEWS": ReportStyle.NEWS,
                    "SOCIAL_MEDIA": ReportStyle.SOCIAL_MEDIA,
                }
                report_style = style_mapping.get(
                    request.report_style.upper(), ReportStyle.ACADEMIC
                )
            except Exception:
                # If invalid style, default to ACADEMIC
                report_style = ReportStyle.ACADEMIC
        else:
            report_style = ReportStyle.ACADEMIC

        workflow = build_prompt_enhancer_graph()
        final_state = workflow.invoke(
            {
                "prompt": request.prompt,
                "context": request.context,
                "report_style": report_style,
            }
        )
        return {"result": final_state["output"]}
    except Exception as e:
        logger.exception(f"Error occurred during prompt enhancement: {str(e)}")
        raise HTTPException(status_code=500, detail=INTERNAL_SERVER_ERROR_DETAIL)


@app.post("/api/mcp/server/metadata", response_model=MCPServerMetadataResponse)
async def mcp_server_metadata(request: MCPServerMetadataRequest):
    """Get information about an MCP server."""
    # Check if MCP server configuration is enabled
    if not get_bool_env("ENABLE_MCP_SERVER_CONFIGURATION", False):
        raise HTTPException(
            status_code=403,
            detail="MCP server configuration is disabled. Set ENABLE_MCP_SERVER_CONFIGURATION=true to enable MCP features.",
        )

    try:
        # Set default timeout with a longer value for this endpoint
        timeout = 300  # Default to 300 seconds for this endpoint

        # Use custom timeout from request if provided
        if request.timeout_seconds is not None:
            timeout = request.timeout_seconds

        # Load tools from the MCP server using the utility function
        tools = await load_mcp_tools(
            server_type=request.transport,
            command=request.command,
            args=request.args,
            url=request.url,
            env=request.env,
            headers=request.headers,
            timeout_seconds=timeout,
        )

        # Create the response with tools
        response = MCPServerMetadataResponse(
            transport=request.transport,
            command=request.command,
            args=request.args,
            url=request.url,
            env=request.env,
            headers=request.headers,
            tools=tools,
        )

        return response
    except Exception as e:
        logger.exception(f"Error in MCP server metadata endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=INTERNAL_SERVER_ERROR_DETAIL)


@app.get("/api/rag/config", response_model=RAGConfigResponse)
async def rag_config():
    """Get the config of the RAG."""
    return RAGConfigResponse(provider=SELECTED_RAG_PROVIDER)


@app.get("/api/rag/resources", response_model=RAGResourcesResponse)
async def rag_resources(request: Annotated[RAGResourceRequest, Query()]):
    """Get the resources of the RAG."""
    retriever = build_retriever()
    if retriever:
        return RAGResourcesResponse(resources=retriever.list_resources(request.query))
    return RAGResourcesResponse(resources=[])


@app.get("/api/config", response_model=ConfigResponse)
async def config():
    """Get the config of the server."""
    return ConfigResponse(
        rag=RAGConfigResponse(provider=SELECTED_RAG_PROVIDER),
        models=get_configured_llm_models(),
    )
