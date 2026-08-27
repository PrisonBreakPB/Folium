export type Completion = {command: string; description: string};

export type MessageRole = "user" | "assistant" | "tool" | "system" | "command" | "error";

export type UiMessage = {
  id: string;
  role: MessageRole;
  content: string;
  kind?: string;
};

export type Approval = {
  request_id: string;
  proposal: {
    title?: string;
    path?: string;
    diff?: string;
    files?: Array<{path: string; title?: string; diff?: string; additions?: number; deletions?: number}>;
  };
};

export type EventMessage = {
  type: string;
  request_id?: string;
  session_id?: string | null;
  model?: string;
  mode?: string;
  workspace?: string;
  content?: string;
  response?: string;
  text?: string;
  level?: string;
  event?: {type?: string; name?: string; status?: string; message?: string; arguments_preview?: string; preview?: string; diff?: string; estimated_context_tokens?: number; prompt_tokens?: number; completion_tokens?: number};
  kind?: string;
  data?: Record<string, unknown>;
  skills?: string[];
  proposal?: Approval["proposal"];
  [key: string]: unknown;
};

export const COMMANDS: Completion[] = [
  ["help", "Show help"], ["new", "Start a new conversation"], ["reset", "Clear conversation history"],
  ["model", "Show or switch model"], ["mode", "Show or switch agent mode"], ["skills", "List available skills"],
  ["status", "Show runtime status"], ["context", "Show context and token usage"], ["usage", "Alias for status"],
  ["workspace", "Show workspace paths"], ["todos", "Show todo list"], ["tokens", "Show token usage"],
  ["compact", "Compress conversation context"], ["diff", "Show modified files"], ["save", "Save session"],
  ["sessions", "List saved sessions"], ["switch", "Switch session"], ["delete", "Delete session"],
  ["traces", "List execution traces"], ["trace", "Show a trace"], ["quit", "Exit Folium"], ["exit", "Exit Folium"],
].map(([command, description]) => ({command: `/${command}`, description}));

function formatStructured(kind: string, data: Record<string, unknown>): string[] {
  if (kind === "context") {
    const n = (key: string) => Number(data[key] || 0).toLocaleString();
    const pct = (key: string) => `${(Number(data[key] || 0) * 100).toFixed(1)}%`;
    return [
      `Context window: ${n("estimated_context_tokens")} / ${n("max_context_tokens")} (${pct("context_usage_ratio")})`,
      `Input budget: ${n("estimated_context_tokens")} / ${n("input_budget_tokens")} (${pct("input_budget_usage_ratio")})`,
      `Output reserve: ${n("reserved_output_tokens")} | API max output: ${n("api_max_output_tokens")}`,
      `Last turn: ${n("last_prompt_tokens")} prompt + ${n("last_completion_tokens")} completion + ${n("last_cached_tokens")} cached = ${n("last_total_tokens")} total`,
      `Session total: ${n("total_prompt_tokens")} prompt + ${n("total_completion_tokens")} completion = ${n("total_tokens")}`,
      `Cache hit rate: ${pct("cache_hit_rate")} (${n("total_cached_tokens")} cached)`,
      `Estimated cost: ${data.estimated_cost == null ? "unavailable" : `$${Number(data.estimated_cost).toFixed(6)}`}`,
      `Cost budget: $${Number(data.budget_spent || 0).toFixed(6)} spent / $${Number(data.budget_usd || 0).toFixed(6)} budget`,
    ];
  }
  return [
    `Session: ${String(data.session_id || "(unsaved)")}`,
    `Model: ${String(data.model || "")}`,
    `Mode: ${String(data.mode || "")}`,
    `Workspace: ${String(data.workspace || "")}`,
  ];
}

export function formatEvent(message: EventMessage): string[] {
  if (message.type === "agent_event" && message.event) {
    const event = message.event;
    if (event.type === "tool_start") return [`> ${event.name || "tool"}(${event.arguments_preview || ""})`];
    if (event.type === "tool_result" || event.type === "tool_error") {
      const preview = event.preview ? ` ${event.preview}` : "";
      return [`< ${event.name || "tool"} status=${event.status || "ok"}${preview}`];
    }
    if (event.type === "context_update") return [`context: ${String(event.estimated_context_tokens || 0)} tokens`];
    if (event.type === "usage_update") return [`usage: ${String(event.prompt_tokens || 0)} in + ${String(event.completion_tokens || 0)} out`];
    if (event.type === "agent_status" && event.message) return [event.message];
    return [];
  }
  if (message.type === "command_result") {
    if (message.kind && message.data) return formatStructured(message.kind, message.data);
    return message.text ? message.text.split("\n") : [];
  }
  if (message.type === "error") return [`Error: ${message.message || "unknown error"}`];
  return [];
}
