import React, {useEffect, useRef, useState} from "react";
import {render, Box, Text, useApp, useStdout} from "ink";
import {spawn, ChildProcess} from "node:child_process";
import {createInterface} from "node:readline";
import MessageViewport from "./MessageViewport.js";
import PromptInput from "./PromptInput.js";
import {formatEvent, type Approval, type EventMessage, type MessageRole, type UiMessage} from "./protocol.js";

function pythonCommand(): string {
  return process.env.FOLIUM_PYTHON || (process.platform === "win32" ? "python" : "python3");
}

function App(): React.ReactElement {
  const {exit} = useApp();
  const {stdout} = useStdout();
  const [terminalRows, setTerminalRows] = useState(Math.max(stdout.rows || 24, 12));
  const [messages, setMessages] = useState<UiMessage[]>([]);
  const [approval, setApproval] = useState<Approval | null>(null);
  const [ready, setReady] = useState<EventMessage | null>(null);
  const [busy, setBusy] = useState(false);
  const streamRef = useRef("");
  const requestCounter = useRef(0);
  const childRef = useRef<ChildProcess | null>(null);
  const streamMessageId = useRef<string | null>(null);
  const messageCounter = useRef(0);

  const appendMessage = (role: MessageRole, content: string, kind?: string) => {
    if (!content) return;
    const id = `${role}-${++messageCounter.current}`;
    setMessages((previous) => [...previous, {id, role, content, kind}]);
    return id;
  };

  useEffect(() => {
    if (stdout.isTTY) stdout.write("\u001b[?25l");
    const onResize = () => setTerminalRows(Math.max(stdout.rows || 24, 12));
    stdout.on("resize", onResize);
    return () => {
      stdout.removeListener("resize", onResize);
      if (stdout.isTTY) stdout.write("\u001b[?25h");
    };
  }, [stdout]);

  const sendRequest = (request: Record<string, unknown>) => {
    const stdin = childRef.current?.stdin;
    if (stdin && !stdin.destroyed) stdin.write(`${JSON.stringify(request)}\n`);
  };

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
          const args = process.argv.slice(2);
          const promptIndex = args.findIndex((argument) => argument === "-p" || argument === "--prompt");
          const initialPrompt = promptIndex >= 0 ? args[promptIndex + 1] : undefined;
          if (initialPrompt) {
            sendRequest({type: "message", request_id: "initial", content: initialPrompt});
            appendMessage("user", initialPrompt);
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
          setMessages((previous) => {
            if (streamMessageId.current) {
              return previous.map((item) => item.id === streamMessageId.current ? {...item, content: streamRef.current} : item);
            }
            const id = `assistant-${++messageCounter.current}`;
            streamMessageId.current = id;
            return [...previous, {id, role: "assistant", content: streamRef.current}];
          });
          return;
        }
        if (message.type === "done") {
          if (!message.streamed && message.response) appendMessage("assistant", message.response);
          streamRef.current = "";
          streamMessageId.current = null;
          if (message.model) setReady((previous) => previous ? {...previous, model: message.model} : previous);
          setBusy(false);
          return;
        }
        if (message.type === "error") {
          streamRef.current = "";
          streamMessageId.current = null;
          setBusy(false);
        }
        if (message.type === "command_result" && message.session_id !== undefined) {
          setReady((previous) => previous ? {
            ...previous,
            session_id: message.session_id,
            model: message.model ?? previous.model,
            mode: message.mode ?? previous.mode,
            workspace: message.workspace ?? previous.workspace,
          } : previous);
        }
        if (message.type === "bye") {
          exit();
          return;
        }
        const rendered = formatEvent(message);
        if (rendered.length) {
          const role: MessageRole = message.type === "error" ? "error" : message.type === "agent_event" ? "tool" : "system";
          rendered.forEach((line) => appendMessage(role, line, message.event?.type || message.kind));
        }
      } catch {
        appendMessage("system", line);
      }
    });
    child.on("error", (error) => {
      appendMessage("error", `Process error: ${error.message}`);
      setBusy(false);
    });
    child.on("exit", () => exit());
    return () => {
      reader.close();
      child.kill();
    };
  }, [exit]);

  const submit = (type: "message" | "command", value: string) => {
    const request_id = String(++requestCounter.current);
    sendRequest({type, request_id, [type === "message" ? "content" : "command"]: value});
    appendMessage(type === "message" ? "user" : "command", value);
    setBusy(type === "message");
  };

  const approve = (decision: "approved" | "rejected" | "revision_requested") => {
    sendRequest({type: "approve", request_id: approval?.request_id, decision, ...(decision === "revision_requested" ? {feedback: "Please revise the change."} : {})});
    setApproval(null);
  };

  const shutdown = () => sendRequest({type: "shutdown"});
  const terminalColumns = Math.max(stdout.columns || 80, 40);
  const messageHeight = Math.max(4, terminalRows - 10);

  return (
    <Box flexDirection="column" height={terminalRows} paddingX={1}>
      <Box borderStyle="round" borderColor="cyan" paddingX={1}>
        <Text color="cyan" bold>FOLIUM</Text><Text> / RESEARCH AGENT / v0.3.0</Text>
      </Box>
      <MessageViewport messages={messages} height={messageHeight} columns={terminalColumns} />
      <PromptInput ready={ready} skills={ready?.skills || []} busy={busy} approval={approval} onRequest={submit} onApproval={approve} onShutdown={shutdown} />
    </Box>
  );
}

render(<App />);
