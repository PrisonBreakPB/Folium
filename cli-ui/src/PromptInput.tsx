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

type InputRow = {text: string; start: number; end: number; logicalEnd: number};

function charWidth(character: string): number {
  return /[^\u0000-\u00ff]/.test(character) ? 2 : 1;
}

function wrapInput(input: string, width: number): InputRow[] {
  const rows: InputRow[] = [];
  let offset = 0;
  for (const logicalLine of input.split("\n")) {
    const characters = Array.from(logicalLine);
    const logicalEnd = offset + characters.length;
    if (!characters.length) rows.push({text: "", start: offset, end: offset, logicalEnd});
    else {
      let chunkStart = 0;
      while (chunkStart < characters.length) {
        let chunkEnd = chunkStart;
        let used = 0;
        while (chunkEnd < characters.length && used + charWidth(characters[chunkEnd]) <= width) {
          used += charWidth(characters[chunkEnd]);
          chunkEnd += 1;
        }
        if (chunkEnd === chunkStart) chunkEnd += 1;
        rows.push({
          text: characters.slice(chunkStart, chunkEnd).join(""),
          start: offset + chunkStart,
          end: offset + chunkEnd,
          logicalEnd,
        });
        chunkStart = chunkEnd;
      }
    }
    offset = logicalEnd + 1;
  }
  return rows;
}

function currentRowIndex(rows: InputRow[], cursor: number): number {
  for (let index = 0; index < rows.length; index += 1) {
    const row = rows[index];
    const isLastChunk = row.end === row.logicalEnd;
    if (cursor < row.end || (cursor === row.end && isLastChunk)) return index;
  }
  return Math.max(0, rows.length - 1);
}

function moveVertical(rows: InputRow[], cursor: number, direction: -1 | 1): number | null {
  const rowIndex = currentRowIndex(rows, cursor);
  const targetIndex = rowIndex + direction;
  if (targetIndex < 0 || targetIndex >= rows.length) return null;
  const column = cursor - rows[rowIndex].start;
  return Math.min(rows[targetIndex].start + column, rows[targetIndex].end);
}

function logicalLineBounds(input: string, cursor: number): {start: number; end: number} {
  const characters = Array.from(input);
  const start = characters.slice(0, cursor).lastIndexOf("\n") + 1;
  const newline = characters.slice(cursor).indexOf("\n");
  return {start, end: newline < 0 ? characters.length : cursor + newline};
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
    const contentWidth = Math.max(8, (stdout.columns || 80) - 7);
    const visualRows = wrapInput(input, contentWidth);
    if (!key.ctrl && !key.meta && key.upArrow) {
      if (matches.length) {
        setSelected((index) => (index - 1 + matches.length) % matches.length);
      } else if (visualRows.length > 1 && moveVertical(visualRows, cursor, -1) !== null) {
        setCursor(moveVertical(visualRows, cursor, -1) as number);
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
      } else if (visualRows.length > 1 && moveVertical(visualRows, cursor, 1) !== null) {
        setCursor(moveVertical(visualRows, cursor, 1) as number);
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
      setCursor(logicalLineBounds(input, cursor).start);
      return;
    }
    if (endPressed || (key.ctrl && value === "e")) {
      setCursor(logicalLineBounds(input, cursor).end);
      return;
    }
    if (key.return) {
      if (matches.length && matches[selected].command.toLowerCase() !== input.toLowerCase()) {
        const command = matches[selected].command;
        setInput(command);
        setCursor(command.length);
        setSelected(0);
      } else {
        send(input.startsWith("/") ? "command" : "message", input);
      }
      return;
    }
    // Ink exposes PowerShell's Backspace (DEL) as key.delete and hides the raw sequence.
    const backspacePressed = key.backspace || key.delete || value === "\b" || value === "\u007f" || value === "\u001b\b";
    if (backspacePressed) {
      const characters = Array.from(input);
      const offset = cursor - 1;
      if (offset >= 0 && offset < characters.length) {
        characters.splice(offset, 1);
        setInput(characters.join(""));
        setCursor(offset);
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

  const contentWidth = Math.max(8, (stdout.columns || 80) - 7);
  const inputRows = wrapInput(input, contentWidth);
  const cursorRow = currentRowIndex(inputRows, cursor);
  return (
    <Box flexDirection="column" marginTop={1}>
      <ApprovalPanel approval={approval} />
      <CompletionList items={matches} selected={selected} />
      <Box flexDirection="column" borderStyle="round" borderColor={busy ? "yellow" : "cyan"} paddingX={1}>
        <FooterStatus ready={ready} />
        <Box flexDirection="column" width={Math.max(20, stdout.columns || 80)}>
          {inputRows.map((row, index) => {
            const lineCursor = index === cursorRow ? cursor - row.start : -1;
            const lineCharacters = Array.from(row.text);
            return (
              <Box key={`${index}-${row.start}-${row.text}`}>
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
