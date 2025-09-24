// Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
// SPDX-License-Identifier: MIT

import { nanoid } from "nanoid";
import { toast } from "sonner";
import { create } from "zustand";
import { useShallow } from "zustand/react/shallow";

import {
  chatStream,
  createSession as apiCreateSession,
  deleteSession as apiDeleteSession,
  fetchSession as apiFetchSession,
  fetchSessions as apiFetchSessions,
  generatePodcast,
  updateSession as apiUpdateSession,
  type SessionDetail,
  type SessionMessage as SessionMessagePayload,
  type SessionSummary as SessionSummaryPayload,
} from "../api";
import type { Message, Resource, ToolCallRuntime } from "../messages";
import { mergeMessage } from "../messages";
import { parseJSON } from "../utils";

import { getChatStreamSettings } from "./settings-store";

const ACTIVE_SESSION_STORAGE_KEY = "deerflow/activeSessionId";
let pendingSessionCreation: Promise<void> | null = null;

interface StoreState {
  responding: boolean;
  sessions: SessionSummaryPayload[];
  sessionLoading: boolean;
  activeSessionId: string | null;
  activeThreadId: string | null;
  messageIds: string[];
  messages: Map<string, Message>;
  researchIds: string[];
  researchPlanIds: Map<string, string>;
  researchReportIds: Map<string, string>;
  researchActivityIds: Map<string, string[]>;
  ongoingResearchId: string | null;
  openResearchId: string | null;

  setResponding: (value: boolean) => void;
  setSessions: (sessions: SessionSummaryPayload[]) => void;
  setSessionLoading: (value: boolean) => void;
  setActiveSessionState: (sessionId: string | null, threadId: string | null) => void;
  hydrateMessages: (messages: Message[]) => void;
  appendMessage: (message: Message) => void;
  updateMessage: (message: Message) => void;
  openResearch: (researchId: string | null) => void;
  closeResearch: () => void;
  setOngoingResearch: (researchId: string | null) => void;
}

export const useStore = create<StoreState>((set) => ({
  responding: false,
  sessions: [],
  sessionLoading: false,
  activeSessionId: null,
  activeThreadId: null,
  messageIds: [],
  messages: new Map<string, Message>(),
  researchIds: [],
  researchPlanIds: new Map<string, string>(),
  researchReportIds: new Map<string, string>(),
  researchActivityIds: new Map<string, string[]>(),
  ongoingResearchId: null,
  openResearchId: null,

  setResponding(value) {
    set({ responding: value });
  },
  setSessions(sessions) {
    set({ sessions });
  },
  setSessionLoading(value) {
    set({ sessionLoading: value });
  },
  setActiveSessionState(sessionId, threadId) {
    if (typeof window !== "undefined") {
      if (sessionId) {
        localStorage.setItem(ACTIVE_SESSION_STORAGE_KEY, sessionId);
      } else {
        localStorage.removeItem(ACTIVE_SESSION_STORAGE_KEY);
      }
    }
    set({ activeSessionId: sessionId, activeThreadId: threadId });
  },
  hydrateMessages(messages) {
    const computed = buildStateFromMessages(messages);
    set({
      messageIds: computed.messageIds,
      messages: computed.messageMap,
      researchIds: computed.researchIds,
      researchPlanIds: computed.researchPlanIds,
      researchReportIds: computed.researchReportIds,
      researchActivityIds: computed.researchActivityIds,
      ongoingResearchId: null,
      openResearchId: null,
    });
  },
  appendMessage(message) {
    set((state) => {
      const nextMessages = new Map(state.messages);
      nextMessages.set(message.id, message);
      const nextMessageIds = [...state.messageIds, message.id];

      const updates: Partial<StoreState> = {
        messageIds: nextMessageIds,
        messages: nextMessages,
      };

      if (
        message.agent === "coder" ||
        message.agent === "reporter" ||
        message.agent === "researcher"
      ) {
        const {
          researchIds,
          researchPlanIds,
          researchReportIds,
          researchActivityIds,
          ongoingResearchId,
        } = updateResearchStateFromMessage(state, message);
        updates.researchIds = researchIds;
        updates.researchPlanIds = researchPlanIds;
        updates.researchReportIds = researchReportIds;
        updates.researchActivityIds = researchActivityIds;
        updates.ongoingResearchId = ongoingResearchId;
        if (message.agent === "coder" || message.agent === "researcher") {
          updates.openResearchId = ongoingResearchId;
        }
        if (message.agent === "reporter") {
          updates.openResearchId = null;
        }
      }
      if (updates.ongoingResearchId === undefined) {
        updates.ongoingResearchId = state.ongoingResearchId;
      }
      if (updates.openResearchId === undefined) {
        updates.openResearchId = state.openResearchId;
      }
      return updates;
    });
  },
  updateMessage(message) {
    set((state) => {
      const nextMessages = new Map(state.messages).set(message.id, message);
      let ongoingResearchId = state.ongoingResearchId;
      if (
        ongoingResearchId &&
        message.agent === "reporter" &&
        !message.isStreaming
      ) {
        ongoingResearchId = null;
      }
      return {
        messages: nextMessages,
        ongoingResearchId,
      } as Partial<StoreState>;
    });
  },
  openResearch(researchId) {
    set({ openResearchId: researchId });
  },
  closeResearch() {
    set({ openResearchId: null });
  },
  setOngoingResearch(researchId) {
    set({ ongoingResearchId: researchId });
  },
}));

export async function loadSessions() {
  useStore.getState().setSessionLoading(true);
  try {
    const sessions = await apiFetchSessions();
    useStore.getState().setSessions(sessions);
    const storedId =
      typeof window !== "undefined"
        ? localStorage.getItem(ACTIVE_SESSION_STORAGE_KEY)
        : null;
    const preferred = sessions.find((session) => session.id === storedId);
    const fallback = sessions[0];
    if (preferred) {
      await activateSession(preferred.id);
    } else if (fallback) {
      await activateSession(fallback.id);
    } else {
      await createAndActivateSession();
    }
  } catch (error) {
    console.error(error);
    toast("加载会话列表失败，请稍后重试。");
  } finally {
    useStore.getState().setSessionLoading(false);
  }
}

export async function activateSession(sessionId: string) {
  useStore.getState().setSessionLoading(true);
  try {
    const detail = await apiFetchSession(sessionId);
    const messages = detail.messages.map((message) =>
      toMessage(detail.threadId, message),
    );
    useStore.getState().hydrateMessages(messages);
    useStore.getState().setActiveSessionState(detail.id, detail.threadId);
    useStore.getState().setSessions(updateSummaryList(detail));
  } catch (error) {
    console.error(error);
    toast("加载会话失败，请稍后重试。");
  } finally {
    useStore.getState().setSessionLoading(false);
  }
}

export async function createAndActivateSession(initialMessage?: string) {
  if (pendingSessionCreation) {
    await pendingSessionCreation;
    return;
  }
  pendingSessionCreation = (async () => {
    useStore.getState().setSessionLoading(true);
    try {
      const detail = await apiCreateSession(initialMessage);
      useStore.getState().setSessions([
        transformDetailToSummary(detail),
        ...useStore.getState().sessions,
      ]);
      const messages = detail.messages.map((message) =>
        toMessage(detail.threadId, message),
      );
      useStore.getState().hydrateMessages(messages);
      useStore.getState().setActiveSessionState(detail.id, detail.threadId);
    } catch (error) {
      console.error(error);
      toast("创建会话失败，请稍后重试。");
    } finally {
      useStore.getState().setSessionLoading(false);
      pendingSessionCreation = null;
    }
  })();
  await pendingSessionCreation;
}

export async function renameSession(sessionId: string, title: string) {
  try {
    const detail = await apiUpdateSession(sessionId, { title });
    useStore.getState().setSessions(updateSummaryList(detail));
    if (useStore.getState().activeSessionId === sessionId) {
      const messages = detail.messages.map((message) =>
        toMessage(detail.threadId, message),
      );
      useStore.getState().hydrateMessages(messages);
      useStore.getState().setActiveSessionState(detail.id, detail.threadId);
    }
  } catch (error) {
    console.error(error);
    toast("重命名会话失败，请稍后重试。");
  }
}

export async function removeSession(sessionId: string) {
  try {
    await apiDeleteSession(sessionId);
    const remaining = useStore
      .getState()
      .sessions.filter((session) => session.id !== sessionId);
    useStore.getState().setSessions(remaining);
    if (useStore.getState().activeSessionId === sessionId) {
      if (remaining.length > 0) {
        await activateSession(remaining[0]!.id);
      } else {
        useStore.getState().hydrateMessages([]);
        useStore.getState().setActiveSessionState(null, null);
      }
    }
  } catch (error) {
    console.error(error);
    toast("删除会话失败，请稍后重试。");
  }
}

export async function sendMessage(
  content?: string,
  {
    interruptFeedback,
    resources,
  }: {
    interruptFeedback?: string;
    resources?: Array<Resource>;
  } = {},
  options: { abortSignal?: AbortSignal } = {},
) {
  const { sessionId, threadId } = await ensureActiveSession();

  if (content != null) {
    appendMessage({
      id: nanoid(),
      threadId,
      role: "user",
      content,
      contentChunks: [content],
      resources,
    });
  }

  const settings = getChatStreamSettings();
  const stream = chatStream(
    content ?? "[REPLAY]",
    {
      thread_id: threadId,
      interrupt_feedback: interruptFeedback,
      resources,
      auto_accepted_plan: settings.autoAcceptedPlan,
      enable_deep_thinking: settings.enableDeepThinking ?? false,
      enable_background_investigation:
        settings.enableBackgroundInvestigation ?? true,
      max_plan_iterations: settings.maxPlanIterations,
      max_step_num: settings.maxStepNum,
      max_search_results: settings.maxSearchResults,
      report_style: settings.reportStyle,
      mcp_settings: settings.mcpSettings,
    },
    options,
  );

  setResponding(true);
  let messageId: string | undefined;
  try {
    for await (const event of stream) {
      const { type, data } = event;
      messageId = data.id;
      let message: Message | undefined;
      if (type === "tool_call_result") {
        message = findMessageByToolCallId(data.tool_call_id);
      } else if (!existsMessage(messageId)) {
        message = {
          id: messageId,
          threadId: data.thread_id,
          agent: data.agent,
          role: data.role,
          content: "",
          contentChunks: [],
          reasoningContent: "",
          reasoningContentChunks: [],
          isStreaming: true,
          interruptFeedback,
        };
        appendMessage(message);
      }
      message ??= getMessage(messageId);
      if (message) {
        message = mergeMessage(message, event);
        updateMessage(message);
      }
    }
    await refreshSessionSummary(sessionId);
  } catch (error) {
    console.error(error);
    toast("生成回复时出现错误，请稍后再试。");
    if (messageId != null) {
      const message = getMessage(messageId);
      if (message?.isStreaming) {
        message.isStreaming = false;
        useStore.getState().updateMessage(message);
      }
    }
    useStore.getState().setOngoingResearch(null);
  } finally {
    setResponding(false);
  }
}

export async function refreshSessionSummary(sessionId: string) {
  try {
    const detail = await apiFetchSession(sessionId);
    useStore.getState().setSessions(updateSummaryList(detail));
  } catch (error) {
    console.error(error);
  }
}

function setResponding(value: boolean) {
  useStore.getState().setResponding(value);
}

function existsMessage(id: string) {
  return useStore.getState().messageIds.includes(id);
}

function getMessage(id: string) {
  return useStore.getState().messages.get(id);
}

function findMessageByToolCallId(toolCallId: string) {
  return Array.from(useStore.getState().messages.values())
    .reverse()
    .find((message) =>
      message.toolCalls?.some((toolCall: ToolCallRuntime) => toolCall.id === toolCallId),
    );
}

function appendMessage(message: Message) {
  useStore.getState().appendMessage(message);
}

function updateMessage(message: Message) {
  useStore.getState().updateMessage(message);
}

function getOngoingResearchId() {
  return useStore.getState().ongoingResearchId;
}

function updateResearchStateFromMessage(state: StoreState, message: Message) {
  const researchIds = [...state.researchIds];
  const researchPlanIds = new Map(state.researchPlanIds);
  const researchReportIds = new Map(state.researchReportIds);
  const researchActivityIds = new Map(state.researchActivityIds);
  let ongoingResearchId = state.ongoingResearchId;
  let lastPlannerId: string | null = null;

  for (let i = state.messageIds.length - 1; i >= 0; i -= 1) {
    const id = state.messageIds[i]!;
    const msg = state.messages.get(id);
    if (msg?.agent === "planner") {
      lastPlannerId = msg.id;
      break;
    }
  }

  if (!ongoingResearchId && lastPlannerId) {
    ongoingResearchId = message.id;
    researchIds.push(message.id);
    researchPlanIds.set(message.id, lastPlannerId);
    researchActivityIds.set(message.id, [lastPlannerId, message.id]);
  } else if (ongoingResearchId) {
    const activity = researchActivityIds.get(ongoingResearchId) ?? [];
    if (!activity.includes(message.id)) {
      researchActivityIds.set(ongoingResearchId, [...activity, message.id]);
    }
  }

  if (ongoingResearchId && message.agent === "reporter") {
    researchReportIds.set(ongoingResearchId, message.id);
  }

  return {
    researchIds,
    researchPlanIds,
    researchReportIds,
    researchActivityIds,
    ongoingResearchId,
  };
}

function buildStateFromMessages(messages: Message[]) {
  const messageIds: string[] = [];
  const messageMap = new Map<string, Message>();
  const researchIds: string[] = [];
  const researchPlanIds = new Map<string, string>();
  const researchReportIds = new Map<string, string>();
  const researchActivityIds = new Map<string, string[]>();
  let lastPlannerId: string | null = null;
  let currentResearchId: string | null = null;

  for (const message of messages) {
    messageIds.push(message.id);
    messageMap.set(message.id, message);

    if (message.agent === "planner") {
      lastPlannerId = message.id;
      continue;
    }

    if (
      message.agent === "coder" ||
      message.agent === "reporter" ||
      message.agent === "researcher"
    ) {
      if (!currentResearchId && lastPlannerId) {
        currentResearchId = message.id;
        researchIds.push(message.id);
        researchPlanIds.set(message.id, lastPlannerId);
        researchActivityIds.set(message.id, [lastPlannerId, message.id]);
      } else if (currentResearchId) {
        const activity = researchActivityIds.get(currentResearchId) ?? [];
        if (!activity.includes(message.id)) {
          researchActivityIds.set(currentResearchId, [...activity, message.id]);
        }
      }
      if (currentResearchId && message.agent === "reporter") {
        researchReportIds.set(currentResearchId, message.id);
        currentResearchId = null;
      }
    }
  }

  return {
    messageIds,
    messageMap,
    researchIds,
    researchPlanIds,
    researchReportIds,
    researchActivityIds,
  };
}

function transformDetailToSummary(detail: SessionDetail): SessionSummaryPayload {
  const { messages, ...summary } = detail;
  return summary;
}

function updateSummaryList(detail: SessionDetail) {
  const summaries = useStore.getState().sessions;
  const summary = transformDetailToSummary(detail);
  const filtered = summaries.filter((item) => item.id !== summary.id);
  return [summary, ...filtered];
}

function toMessage(threadId: string, message: SessionMessagePayload): Message {
  const reasoningContent = message.reasoningContent ?? undefined;
  return {
    id: message.id,
    threadId,
    agent: message.agent as Message["agent"],
    role: message.role,
    content: message.content,
    contentChunks: [message.content],
    reasoningContent,
    reasoningContentChunks: reasoningContent ? [reasoningContent] : [],
    toolCalls: message.toolCalls ?? undefined,
    finishReason: "stop",
  };
}

async function ensureActiveSession() {
  const state = useStore.getState();
  if (state.activeSessionId && state.activeThreadId) {
    return { sessionId: state.activeSessionId, threadId: state.activeThreadId };
  }
  await createAndActivateSession();
  const refreshed = useStore.getState();
  if (!refreshed.activeSessionId || !refreshed.activeThreadId) {
    throw new Error("Failed to obtain active session");
  }
  return {
    sessionId: refreshed.activeSessionId,
    threadId: refreshed.activeThreadId,
  };
}

export function openResearch(researchId: string | null) {
  useStore.getState().openResearch(researchId);
}

export function closeResearch() {
  useStore.getState().closeResearch();
}

export async function listenToPodcast(researchId: string) {
  const planMessageId = useStore.getState().researchPlanIds.get(researchId);
  const reportMessageId = useStore.getState().researchReportIds.get(researchId);
  const threadId = useStore.getState().activeThreadId;
  if (!threadId) {
    toast("请先选择一个会话。");
    return;
  }
  if (planMessageId && reportMessageId) {
    const planMessage = getMessage(planMessageId)!;
    const title = parseJSON(planMessage.content, { title: "Untitled" }).title;
    const reportMessage = getMessage(reportMessageId);
    if (reportMessage?.content) {
      appendMessage({
        id: nanoid(),
        threadId,
        role: "user",
        content: "Please generate a podcast for the above research.",
        contentChunks: [],
      });
      const podCastMessageId = nanoid();
      const podcastObject = { title, researchId };
      const podcastMessage: Message = {
        id: podCastMessageId,
        threadId,
        role: "assistant",
        agent: "podcast",
        content: JSON.stringify(podcastObject),
        contentChunks: [],
        reasoningContent: "",
        reasoningContentChunks: [],
        isStreaming: true,
      };
      appendMessage(podcastMessage);
      let audioUrl: string | undefined;
      try {
        audioUrl = await generatePodcast(reportMessage.content);
      } catch (error) {
        console.error(error);
        useStore.setState((state) => ({
          messages: new Map(state.messages).set(podCastMessageId, {
            ...state.messages.get(podCastMessageId)!,
            content: JSON.stringify({
              ...podcastObject,
              error: error instanceof Error ? error.message : "Unknown error",
            }),
            isStreaming: false,
          }),
        }));
        toast("生成播客时出现错误，请稍后再试。");
        return;
      }
      useStore.setState((state) => ({
        messages: new Map(state.messages).set(podCastMessageId, {
          ...state.messages.get(podCastMessageId)!,
          content: JSON.stringify({ ...podcastObject, audioUrl }),
          isStreaming: false,
        }),
      }));
    }
  }
}

export function useResearchMessage(researchId: string) {
  return useStore(
    useShallow((state) => {
      const messageId = state.researchPlanIds.get(researchId);
      return messageId ? state.messages.get(messageId) : undefined;
    }),
  );
}

export function useMessage(messageId: string | null | undefined) {
  return useStore(
    useShallow((state) =>
      messageId ? state.messages.get(messageId) : undefined,
    ),
  );
}

export function useMessageIds() {
  return useStore(useShallow((state) => state.messageIds));
}

export function useLastInterruptMessage() {
  return useStore(
    useShallow((state) => {
      if (state.messageIds.length >= 2) {
        const lastMessage = state.messages.get(
          state.messageIds[state.messageIds.length - 1]!,
        );
        return lastMessage?.finishReason === "interrupt" ? lastMessage : null;
      }
      return null;
    }),
  );
}

export function useLastFeedbackMessageId() {
  const waitingForFeedbackMessageId = useStore(
    useShallow((state) => {
      if (state.messageIds.length >= 2) {
        const lastMessage = state.messages.get(
          state.messageIds[state.messageIds.length - 1]!,
        );
        if (lastMessage && lastMessage.finishReason === "interrupt") {
          return state.messageIds[state.messageIds.length - 2];
        }
      }
      return null;
    }),
  );
  return waitingForFeedbackMessageId;
}

export function useToolCalls() {
  return useStore(
    useShallow((state) => {
      return state.messageIds
        ?.map((id) => getMessage(id)?.toolCalls)
        .filter((toolCalls) => toolCalls != null)
        .flat();
    }),
  );
}

export function useSessionsList() {
  return useStore(useShallow((state) => state.sessions));
}

export function useActiveSessionMeta() {
  return useStore(
    useShallow((state) => ({
      sessionId: state.activeSessionId,
      threadId: state.activeThreadId,
    })),
  );
}

export function useSessionLoading() {
  return useStore((state) => state.sessionLoading);
}

export function useResponding() {
  return useStore((state) => state.responding);
}
