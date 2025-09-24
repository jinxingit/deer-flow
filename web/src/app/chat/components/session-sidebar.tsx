"use client";

import { Plus, Trash2, Pencil } from "lucide-react";
import { useEffect, useMemo, useRef } from "react";

import { Button } from "~/components/ui/button";
import { useActiveSessionMeta, useResponding, useSessionLoading, useSessionsList } from "~/core/store";
import {
  activateSession,
  createAndActivateSession,
  loadSessions,
  removeSession,
  renameSession,
} from "~/core/store";

function formatTimestamp(value?: string | null) {
  if (!value) return "";
  try {
    const date = new Date(value);
    return new Intl.DateTimeFormat(undefined, {
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    }).format(date);
  } catch {
    return "";
  }
}

export function SessionsSidebar() {
  const initialized = useRef(false);
  const sessions = useSessionsList();
  const { sessionId: activeSessionId } = useActiveSessionMeta();
  const loading = useSessionLoading();
  const responding = useResponding();

  useEffect(() => {
    if (initialized.current) return;
    initialized.current = true;
    void loadSessions();
  }, []);

  const hasSessions = sessions.length > 0;

  const headerStatus = useMemo(() => {
    if (loading) return "加载中";
    if (responding) return "生成中";
    return "";
  }, [loading, responding]);

  return (
    <aside className="flex h-full w-72 shrink-0 flex-col border-r border-border bg-background/60 pt-12 backdrop-blur">
      <div className="flex items-center justify-between px-4 py-3">
        <div>
          <div className="text-sm font-medium">会话</div>
          {headerStatus && (
            <div className="text-xs text-muted-foreground">{headerStatus}</div>
          )}
        </div>
        <Button size="icon" variant="outline" onClick={() => createAndActivateSession()}>
          <Plus className="h-4 w-4" />
        </Button>
      </div>
      <div className="flex-1 overflow-y-auto">
        {hasSessions ? (
          <ul className="space-y-1 px-3 pb-4">
            {sessions.map((session) => {
              const isActive = session.id === activeSessionId;
              const title = session.title ?? "未命名会话";
              const subtitle = session.lastMessagePreview ?? "开始新的研究";
              return (
                <li key={session.id}>
                  <div
                    role="button"
                    tabIndex={0}
                    onClick={() => {
                      if (session.id !== activeSessionId) {
                        void activateSession(session.id);
                      }
                    }}
                    onKeyDown={(event) => {
                      if (event.key === "Enter" || event.key === " ") {
                        event.preventDefault();
                        if (session.id !== activeSessionId) {
                          void activateSession(session.id);
                        }
                      }
                    }}
                    className={`group flex w-full items-start justify-between rounded-md border border-transparent px-3 py-2 text-left transition hover:bg-secondary focus:outline-none focus-visible:ring-2 focus-visible:ring-ring ${isActive ? "bg-secondary" : ""}`}
                  >
                    <div className="flex-1 pr-3">
                      <div className="text-sm font-medium leading-tight text-foreground">
                        {title}
                      </div>
                      <div className="text-xs text-muted-foreground line-clamp-1">
                        {subtitle}
                      </div>
                      <div className="text-[10px] uppercase text-muted-foreground/70">
                        {formatTimestamp(session.updatedAt)}
                      </div>
                    </div>
                    <div className="flex gap-2 opacity-0 transition group-hover:opacity-100">
                      <Button
                        variant="ghost"
                        size="icon"
                        className="h-7 w-7 rounded-sm text-muted-foreground hover:bg-muted"
                        onClick={(event) => {
                          event.stopPropagation();
                          const nextTitle = window.prompt("输入新的会话名称", title);
                          if (nextTitle && nextTitle.trim()) {
                            void renameSession(session.id, nextTitle.trim());
                          }
                        }}
                      >
                        <Pencil className="h-4 w-4" />
                      </Button>
                      <Button
                        variant="ghost"
                        size="icon"
                        className="h-7 w-7 rounded-sm text-muted-foreground hover:bg-muted"
                        onClick={(event) => {
                          event.stopPropagation();
                          if (
                            window.confirm(
                              `确认删除会话“${title}”吗？该操作不可撤销。`,
                            )
                          ) {
                            void removeSession(session.id);
                          }
                        }}
                      >
                        <Trash2 className="h-4 w-4" />
                      </Button>
                    </div>
                  </div>
                </li>
              );
            })}
          </ul>
        ) : (
          <div className="flex h-full flex-col items-center justify-center px-6 text-center text-sm text-muted-foreground">
            <p>暂无会话，点击右上角按钮开始新的深度研究。</p>
          </div>
        )}
      </div>
    </aside>
  );
}
