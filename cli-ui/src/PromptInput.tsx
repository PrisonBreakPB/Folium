import React, {useEffect, useMemo, useRef, useState} from "react";
import {Box, Text, useInput, useStdout} from "ink";
import FooterStatus from "./FooterStatus.js";
import {COMMANDS, type Approval, type Completion, type EventMessage} from "./protocol.js";

type Props = {
  ready: EventMessage | null;
  skills: string[];
  busy: boolean;
  approval: Approval | null;
  onRequest: (type: "message" | "command", value: string) => void;
  onApproval: (decision: "approved" | "rejected" | "revision_requested") => void;
  onShutdown: () => void;
};

function ApprovalPanel({approval}: Pick<Props, "approval">): React.ReactElement | null {
  if (!approval) return null;
  const proposal = approval.proposal;
  const files = proposal.files || [{path: proposal.path || "", diff: proposal.diff || ""}];
  const lines = [
    `Approval required: ${proposal.title || "Review change"}`,
    ...files.flatMap((file) => [
      file.path,
      ...(file.diff || "").split("\n").map((line) => line.startsWith("+") ? `+ ${line}` : line.startsWith("-") ? `- ${line}` : `  ${line}`),
    ]),
  ];
  return (
    <Box flexDirection="column" borderStyle="round" borderColor="yellow" paddingX={1}>
      {lines.slice(-8).map((line, index) => <Text key={`${index}-${line}`} color={line.startsWith("+") ? "green" : line.startsWith("-") ? "red" : undefined}>{line}</Text>)}
      <Text dimColor>[y] approve  [n] reject  [e] revise</Text>
    </Box>
  );
}

function CompletionList({items, selected}: {items: Completion[]; selected: number}): React.ReactElement | null {
  if (!items.length) return null;
  return (
    <Box flexDirection="column" borderStyle="single" borderColor="blue" paddingX={1}>
      {items.slice(0, 8).map((item, index) => <Text key={item.command} color={index === selected ? "cyan" : undefined}>{index === selected ? ">" : " "} {item.command}  <Text dimColor>{item.description}</Text></Text>)}
    </Box>
  );
}

function lineInfo(input: string, cursor: number): {line: number; column: number; starts: number[]} {
  const characters = Array.from(input);
  const starts = [0];
  characters.forEach((character, index) => {
    if (character === "\n") starts.push(index + 1);
  });
  const line = Math.max(0, starts.findIndex((start, index) => start > cursor && index > 0) - 1);
  const resolvedLine = line < 0 ? starts.length - 1 : line;
  return {line: resolvedLine, column: cursor - starts[resolvedLine], starts};
}

function moveVertical(input: string, cursor: number, direction: -1 | 1): number | null {
  const info = lineInfo(input, cursor);
  const targetLine = info.line + direction;
  if (targetLine < 0 || targetLine >= info.starts.length) return null;
  const lines = input.split("\n").map((line) => Array.from(line));
  return info.starts[targetLine] + Math.min(info.column, lines[targetLine].length);
}

export default function PromptInput({ready, skills, busy, approval, onRequest, onApproval, onShutdown}: Props): React.ReactElement {
  const {stdout} = useStdout();
  const [input, setInput] = useState("");
  const [cursor, setCursor] = useState(0);
  const [selected, setSelected] = useState(0);
  const [history, setHistory] = useState<string[]>([]);
  const [historyIndex, setHistoryIndex] = useState<number | null>(null);
  const draftRef = useRef("");
  const skillCommands = useMemo(() => skills.map((name) => ({command: `/${name}`, description: "Skill"})), [skills]);
  const matches = useMemo(() => {
    if (!input.startsWith("/") || /\s/.test(input)) return [];
    return [...COMMANDS, ...skillCommands].filter((item) => item.command.toLowerCase().startsWith(input.toLowerCase()));
  }, [input, skillCommands]);

  useEffect(() => setSelected(0), [matches]);

  const send = (type: "message" | "command", value: string) => {
    if (!value.trim() || busy) return;
    onRequest(type, value);
    setHistory((items) => [...items.filter((item) => item !== value), value].slice(-50));
    setInput("");
    setCursor(0);
    setSelected(0);
    setHistoryIndex(null);
  };

  useInput((value, key) => {
    if (approval) {
      if (value.toLowerCase() === "y") onApproval("approved");
      else if (value.toLowerCase() === "n") onApproval("rejected");
      else if (value.toLowerCase() === "e") onApproval("revision_requested");
      return;
    }
    if (key.ctrl && value === "c") {
      onShutdown();
      return;
    }
    const newlineRequested = (key.return && (key.shift || key.meta)) || (key.ctrl && (value === "j" || value === "\n"));
    if (newlineRequested) {
      const next = Array.from(input);
      next.splice(cursor, 0, "\n");
      setInput(next.join(""));
      setCursor(cursor + 1);
      setHistoryIndex(null);
      return;
    }
    if (!key.ctrl && !key.meta && key.upArrow) {
      if (matches.length) {
        setSelected((index) => (index - 1 + matches.length) % matches.length);
      } else if (input.includes("\n") && moveVertical(input, cursor, -1) !== null) {
        setCursor(moveVertical(input, cursor, -1) as number);
      } else if (history.length) {
        const nextIndex = historyIndex === null ? history.length - 1 : Math.max(0, historyIndex - 1);
        if (historyIndex === null) draftRef.current = input;
        setHistoryIndex(nextIndex);
        setInput(history[nextIndex]);
        setCursor(Array.from(history[nextIndex]).length);
      }
      return;
    }
    if (!key.ctrl && !key.meta && key.downArrow) {
      if (matches.length) {
        setSelected((index) => (index + 1) % matches.length);
      } else if (input.includes("\n") && moveVertical(input, cursor, 1) !== null) {
        setCursor(moveVertical(input, cursor, 1) as number);
      } else if (historyIndex !== null) {
        if (historyIndex < history.length - 1) {
          const nextIndex = historyIndex + 1;
          setHistoryIndex(nextIndex);
          setInput(history[nextIndex]);
          setCursor(Array.from(history[nextIndex]).length);
        } else {
          setHistoryIndex(null);
          setInput(draftRef.current);
          setCursor(Array.from(draftRef.current).length);
        }
      }
      return;
    }
    if (key.leftArrow) {
      setCursor((index) => Math.max(0, index - 1));
      return;
    }
    if (key.rightArrow) {
      setCursor((index) => Math.min(Array.from(input).length, index + 1));
      return;
    }
    const homePressed = value === "\u001b[H" || value === "\u001bOH";
    const endPressed = value === "\u001b[F" || value === "\u001bOF";
    if (homePressed || (key.ctrl && value === "a")) {
      setCursor(cursorInfo.starts[cursorInfo.line]);
      return;
    }
    if (endPressed || (key.ctrl && value === "e")) {
      const lineEnd = cursorInfo.line + 1 < cursorInfo.starts.length ? cursorInfo.starts[cursorInfo.line + 1] - 1 : Array.from(input).length;
      setCursor(lineEnd);
      return;
    }
    if (key.return) {
      if (matches.length) {
        const command = matches[selected].command;
        setInput(command);
        setCursor(command.length);
        setSelected(0);
      } else {
        send(input.startsWith("/") ? "command" : "message", input);
      }
      return;
    }
    if (key.backspace || key.delete) {
      const characters = Array.from(input);
      const offset = key.backspace ? cursor - 1 : cursor;
      if (offset >= 0 && offset < characters.length) {
        characters.splice(offset, 1);
        setInput(characters.join(""));
        if (key.backspace) setCursor(offset);
        setHistoryIndex(null);
      }
      return;
    }
    if (!key.ctrl && !key.meta && value) {
      const characters = Array.from(input);
      const inserted = Array.from(value);
      characters.splice(cursor, 0, ...inserted);
      setInput(characters.join(""));
      setCursor(cursor + inserted.length);
      setHistoryIndex(null);
    }
  });

  const characters = Array.from(input);
  const inputLines = input.split("\n");
  const cursorInfo = lineInfo(input, cursor);
  return (
    <Box flexDirection="column" marginTop={1}>
      <ApprovalPanel approval={approval} />
      <CompletionList items={matches} selected={selected} />
      <Box flexDirection="column" borderStyle="round" borderColor={busy ? "yellow" : "cyan"} paddingX={1}>
        <FooterStatus ready={ready} />
        <Box flexDirection="column" width={Math.max(20, stdout.columns || 80)}>
          {inputLines.map((line, index) => {
            const lineStart = cursorInfo.starts[index];
            const lineCursor = index === cursorInfo.line ? cursor - lineStart : -1;
            const lineCharacters = Array.from(line);
            return (
              <Box key={`${index}-${line}`}>
                <Text color="yellow" bold>{index === 0 ? (ready?.mode === "bash" ? "! " : "> ") : "  "}</Text>
                <Text>{lineCharacters.slice(0, Math.max(0, lineCursor)).join("")}</Text>
                {lineCursor >= 0 && <Text color="cyan">{busy ? "..." : "|"}</Text>}
                <Text>{lineCharacters.slice(Math.max(0, lineCursor)).join("")}</Text>
              </Box>
            );
          })}
        </Box>
      </Box>
    </Box>
  );
}
