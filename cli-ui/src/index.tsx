import React, {useEffect, useMemo, useRef, useState} from "react";
import {render, Text, Box, useInput, useApp} from "ink";
import {spawn, ChildProcess} from "node:child_process";
import {createInterface} from "node:readline";

type Completion = {command: string; description: string};
type Approval = {
  request_id: string;
  proposal: {
    title?: string;
    path?: string;
    diff?: string;
    files?: Array<{path: string; title?: string; diff?: string; additions?: number; deletions?: number}>;
  };
};
type EventMessage = {
  type: string;
  request_id?: string;
  content?: string;
  response?: string;
  text?: string;
  level?: string;
  event?: {type?: string; name?: string; status?: string; message?: string; arguments_preview?: string; preview?: string; diff?: string};
  kind?: string;
  data?: Record<string, unknown>;
  skills?: string[];
  proposal?: Approval["proposal"];
  [key: string]: unknown;
};

const COMMANDS: Completion[] = [
  ["help", "Show help"], ["new", "Start a new conversation"], ["reset", "Clear conversation history"],
  ["model", "Show or switch model"], ["mode", "Show or switch agent mode"], ["skills", "List available skills"],
  ["status", "Show runtime status"], ["context", "Show context and token usage"], ["usage", "Alias for status"],
  ["workspace", "Show workspace paths"], ["todos", "Show todo list"], ["tokens", "Show token usage"],
  ["compact", "Compress conversation context"], ["diff", "Show modified files"], ["save", "Save session"],
  ["sessions", "List saved sessions"], ["switch", "Switch session"], ["delete", "Delete session"],
  ["traces", "List execution traces"], ["trace", "Show a trace"], ["quit", "Exit Folium"], ["exit", "Exit Folium"],
].map(([command, description]) => ({command: `/${command}`, description}));

function pythonCommand(): string {
  return process.env.FOLIUM_PYTHON || (process.platform === "win32" ? "python" : "python3");
}

function formatEvent(message: EventMessage): string[] {
  if (message.type === "agent_event" && message.event) {
    const event = message.event;
    if (event.type === "tool_start") return [`> ${event.name || "tool"}(${event.arguments_preview || ""})`];
    if (event.type === "tool_result" || event.type === "tool_error") {
      const preview = event.preview ? ` ${event.preview}` : "";
      return [`< ${event.name || "tool"} status=${event.status || "ok"}${preview}`];
    }
    if (event.type === "context_update") {
      return [`context: ${String((message.event as Record<string, unknown>).estimated_context_tokens || 0)} tokens`];
    }
    if (event.type === "usage_update") {
      return [`usage: ${String((message.event as Record<string, unknown>).prompt_tokens || 0)} in + ${String((message.event as Record<string, unknown>).completion_tokens || 0)} out`];
    }
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

function App(): React.ReactElement {
  const {exit} = useApp();
  const [input, setInput] = useState("");
  const [lines, setLines] = useState<string[]>([]);
  const [completions, setCompletions] = useState<Completion[]>([]);
  const [selected, setSelected] = useState(0);
  const [approval, setApproval] = useState<Approval | null>(null);
  const [ready, setReady] = useState<EventMessage | null>(null);
  const [busy, setBusy] = useState(false);
  const streamRef = useRef("");
  const requestCounter = useRef(0);
  const childRef = useRef<ChildProcess | null>(null);

  useEffect(() => {
    const child = spawn(pythonCommand(), ["-m", "folium.cli", "--jsonl", ...process.argv.slice(2)], {
      cwd: process.cwd(),
      env: process.env,
      stdio: ["pipe", "pipe", "inherit"],
    });
    childRef.current = child;
    const reader = createInterface({input: child.stdout});
    reader.on("line", (line) => {
      try {
        const message = JSON.parse(line) as EventMessage;
        if (message.type === "ready") {
          setReady(message);
          const initialArgs = process.argv.slice(2);
          const initialPrompt = initialArgs.reduce<string | null>((value, argument, index, args) => {
            if (value !== null) return value;
            if ((argument === "-p" || argument === "--prompt") && args[index + 1]) return args[index + 1];
            return null;
          }, null);
          if (initialPrompt) {
            sendRequest({type: "message", request_id: "initial", content: initialPrompt});
            setLines((previous) => [...previous, `YOU >> ${initialPrompt}`]);
            setBusy(true);
          }
          return;
        }
        if (message.type === "approval_required") {
          setApproval({request_id: String(message.request_id), proposal: message.proposal || {}});
          setBusy(true);
          return;
        }
        if (message.type === "token") {
          streamRef.current += message.content || "";
          setLines((previous) => {
            const next = [...previous];
            const last = next.length - 1;
            if (last >= 0 && next[last].startsWith("AGENT >> ")) {
              next[last] = `AGENT >> ${streamRef.current}`;
            } else {
              next.push(`AGENT >> ${streamRef.current}`);
            }
            return next;
          });
          return;
        }
        if (message.type === "done") {
          if (!message.streamed && message.response) setLines((previous) => [...previous, message.response || ""]);
          streamRef.current = "";
          setBusy(false);
          return;
        }
        if (message.type === "bye") {
          exit();
          return;
        }
        const rendered = formatEvent(message);
        if (rendered.length) setLines((previous) => [...previous, ...rendered]);
      } catch {
        setLines((previous) => [...previous, line]);
      }
    });
    child.on("exit", () => exit());
    return () => {
      reader.close();
      child.kill();
    };
  }, [exit]);

  const sendRequest = (request: Record<string, unknown>) => {
    const stdin = childRef.current?.stdin;
    if (stdin) stdin.write(`${JSON.stringify(request)}\n`);
  };

  const writeRequest = (request: Record<string, unknown>) => {
    sendRequest(request);
  };

  const skillCommands = useMemo(() => (
    Array.isArray(ready?.skills)
      ? ready.skills.map((name) => ({command: `/${String(name)}`, description: "Skill"}))
      : []
  ), [ready]);

  const matches = useMemo(() => {
    if (!input.startsWith("/") || /\s/.test(input)) return [];
    return [...COMMANDS, ...skillCommands]
      .filter((item) => item.command.toLowerCase().startsWith(input.toLowerCase()));
  }, [input, skillCommands]);

  useEffect(() => {
    setCompletions(matches);
    setSelected(0);
  }, [matches]);

  const send = (type: "message" | "command", value: string) => {
    const child = childRef.current;
    if (!child?.stdin || !value.trim() || busy) return;
    const request_id = String(++requestCounter.current);
    writeRequest({type, request_id, [type === "message" ? "content" : "command"]: value});
    setLines((previous) => [...previous, `YOU >> ${value}`]);
    setInput("");
    setCompletions([]);
    setBusy(type === "message");
  };

  useInput((value, key) => {
    if (approval) {
      if (value.toLowerCase() === "y" || value.toLowerCase() === "n") {
        writeRequest({type: "approve", request_id: approval.request_id, decision: value.toLowerCase() === "y" ? "approved" : "rejected"});
        setApproval(null);
        return;
      }
      if (value.toLowerCase() === "e") {
        writeRequest({type: "approve", request_id: approval.request_id, decision: "revision_requested", feedback: "Please revise the change."});
        setApproval(null);
      }
      return;
    }
    if (key.ctrl && value === "c") {
      writeRequest({type: "shutdown"});
      return;
    }
    if (key.upArrow && completions.length) {
      setSelected((index) => (index - 1 + completions.length) % completions.length);
      return;
    }
    if (key.downArrow && completions.length) {
      setSelected((index) => (index + 1) % completions.length);
      return;
    }
    if (key.return) {
      if (completions.length) {
        setInput(completions[selected].command);
        setCompletions([]);
      } else if (input.startsWith("/")) {
        send("command", input);
      } else {
        send("message", input);
      }
      return;
    }
    if (key.backspace || key.delete) {
      setInput((text) => text.slice(0, -1));
      return;
    }
    if (!key.ctrl && !key.meta && value) setInput((text) => text + value);
  });

  const proposalLines = approval ? [
    `Approval required: ${approval.proposal.title || "Review change"}`,
    ...(approval.proposal.files || [{path: approval.proposal.path || "", diff: approval.proposal.diff || ""}]).flatMap((file) => [
      file.path,
      ...(file.diff || "").split("\n").map((line) => line.startsWith("+") ? `+ ${line}` : line.startsWith("-") ? `- ${line}` : `  ${line}`),
    ]),
    "Press y to approve, n to reject, e to request revision.",
  ] : [];

  return (
    <Box flexDirection="column" paddingX={1}>
      <Box borderStyle="round" borderColor="cyan" paddingX={1}>
        <Text color="cyan" bold>FOLIUM</Text><Text> / RESEARCH AGENT / v0.3.0</Text>
      </Box>
      <Text dimColor>{ready ? `Model: ${String(ready.model)}  Mode: ${String(ready.mode)}  Workspace: ${String(ready.workspace)}` : "Connecting to Python Agent..."}</Text>
      <Box flexDirection="column" height={12} overflow="hidden" marginTop={1}>
        {lines.slice(-12).map((line, index) => <Text key={`${index}-${line}`}>{line}</Text>)}
      </Box>
      {approval && <Box flexDirection="column" borderStyle="round" borderColor="yellow" paddingX={1}>{proposalLines.map((line, index) => <Text key={`${index}-${line}`} color={line.startsWith("+") ? "green" : line.startsWith("-") ? "red" : undefined}>{line}</Text>)}</Box>}
      {completions.length > 0 && <Box flexDirection="column" borderStyle="single" borderColor="blue" paddingX={1}>{completions.map((item, index) => <Text key={item.command} color={index === selected ? "cyan" : undefined}>{index === selected ? ">" : " "} {item.command}  <Text dimColor>{item.description}</Text></Text>)}</Box>}
      <Box marginTop={1}><Text color="yellow" bold>YOU &gt;&gt; </Text><Text>{input}</Text><Text color="gray">{busy ? "  [processing]" : ""}</Text></Box>
    </Box>
  );
}

render(<App />);
