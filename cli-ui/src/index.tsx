import React, {useEffect, useRef, useState} from "react";
import {render, Box, Text, useApp, useStdout} from "ink";
import {spawn, ChildProcess} from "node:child_process";
import {createInterface} from "node:readline";
import MessageViewport from "./MessageViewport.js";
import PromptInput from "./PromptInput.js";
import {formatEvent, type Approval, type EventMessage} from "./protocol.js";

function pythonCommand(): string {
  return process.env.FOLIUM_PYTHON || (process.platform === "win32" ? "python" : "python3");
}

function App(): React.ReactElement {
  const {exit} = useApp();
  const {stdout} = useStdout();
  const [terminalRows, setTerminalRows] = useState(Math.max(stdout.rows || 24, 12));
  const [lines, setLines] = useState<string[]>([]);
  const [approval, setApproval] = useState<Approval | null>(null);
  const [ready, setReady] = useState<EventMessage | null>(null);
  const [busy, setBusy] = useState(false);
  const streamRef = useRef("");
  const requestCounter = useRef(0);
  const childRef = useRef<ChildProcess | null>(null);

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
            if (last >= 0 && next[last].startsWith("AGENT >> ")) next[last] = `AGENT >> ${streamRef.current}`;
            else next.push(`AGENT >> ${streamRef.current}`);
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
    child.on("error", (error) => {
      setLines((previous) => [...previous, `Process error: ${error.message}`]);
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
    setLines((previous) => [...previous, `YOU >> ${value}`]);
    setBusy(type === "message");
  };

  const approve = (decision: "approved" | "rejected" | "revision_requested") => {
    sendRequest({type: "approve", request_id: approval?.request_id, decision, ...(decision === "revision_requested" ? {feedback: "Please revise the change."} : {})});
    setApproval(null);
  };

  const shutdown = () => sendRequest({type: "shutdown"});
  const messageHeight = Math.max(4, terminalRows - 10);

  return (
    <Box flexDirection="column" height={terminalRows} paddingX={1}>
      <Box borderStyle="round" borderColor="cyan" paddingX={1}>
        <Text color="cyan" bold>FOLIUM</Text><Text> / RESEARCH AGENT / v0.3.0</Text>
      </Box>
      <MessageViewport lines={lines} height={messageHeight} />
      <PromptInput ready={ready} skills={ready?.skills || []} busy={busy} approval={approval} onRequest={submit} onApproval={approve} onShutdown={shutdown} />
    </Box>
  );
}

render(<App />);
