import { resolveServiceURL } from "./resolve-service-url";
import type {
  SessionDetailDTO,
  SessionMessageDTO,
  SessionSummaryDTO,
} from "./types";
import type { MessageRole, ToolCallRuntime } from "../messages";

export interface SessionSummary {
  id: string;
  threadId: string;
  title?: string | null;
  summary?: string | null;
  lastMessagePreview?: string | null;
  updatedAt: string;
  createdAt: string;
  archived: boolean;
}

export interface SessionMessage {
  id: string;
  role: MessageRole;
  agent?: string;
  content: string;
  reasoningContent?: string;
  toolCalls?: ToolCallRuntime[];
  metadata?: Record<string, unknown>;
  seq: number;
  createdAt: string;
}

export interface SessionDetail extends SessionSummary {
  messages: Array<SessionMessage>;
}

function transformSummary(dto: SessionSummaryDTO): SessionSummary {
  return {
    id: dto.id,
    threadId: dto.thread_id,
    title: dto.title ?? undefined,
    summary: dto.summary ?? undefined,
    lastMessagePreview: dto.last_message_preview ?? undefined,
    updatedAt: dto.updated_at,
    createdAt: dto.created_at,
    archived: dto.archived,
  };
}

function transformMessage(dto: SessionMessageDTO): SessionMessage {
  return {
    id: dto.id,
    role: dto.role,
    agent: dto.agent ?? undefined,
    content: dto.content,
    reasoningContent: dto.reasoning_content ?? undefined,
    toolCalls: dto.tool_calls ?? undefined,
    metadata: dto.metadata ?? undefined,
    seq: dto.seq,
    createdAt: dto.created_at,
  };
}

function transformDetail(dto: SessionDetailDTO): SessionDetail {
  return {
    ...transformSummary(dto),
    messages: dto.messages.map(transformMessage),
  };
}

async function handleResponse<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || `Request failed with status ${res.status}`);
  }
  return (await res.json()) as T;
}

export async function fetchSessions(): Promise<Array<SessionSummary>> {
  const data = await handleResponse<{ sessions: Array<SessionSummaryDTO> }>(
    await fetch(resolveServiceURL("sessions"), {
      method: "GET",
      headers: { "Content-Type": "application/json" },
    }),
  );
  return data.sessions.map(transformSummary);
}

export async function fetchSession(sessionId: string): Promise<SessionDetail> {
  const data = await handleResponse<SessionDetailDTO>(
    await fetch(resolveServiceURL(`sessions/${sessionId}`), {
      method: "GET",
      headers: { "Content-Type": "application/json" },
    }),
  );
  return transformDetail(data);
}

export async function createSession(
  initialMessage?: string,
): Promise<SessionDetail> {
  const payload = initialMessage ? { initial_message: initialMessage } : {};
  const data = await handleResponse<{ session: SessionDetailDTO }>(
    await fetch(resolveServiceURL("sessions"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }),
  );
  return transformDetail(data.session);
}

export async function updateSession(
  sessionId: string,
  payload: { title?: string; archived?: boolean },
): Promise<SessionDetail> {
  const data = await handleResponse<SessionDetailDTO>(
    await fetch(resolveServiceURL(`sessions/${sessionId}`), {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        title: payload.title,
        archived: payload.archived,
      }),
    }),
  );
  return transformDetail(data);
}

export async function deleteSession(sessionId: string): Promise<void> {
  await handleResponse<{ success: boolean }>(
    await fetch(resolveServiceURL(`sessions/${sessionId}`), {
      method: "DELETE",
      headers: { "Content-Type": "application/json" },
    }),
  );
}
