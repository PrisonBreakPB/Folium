import React, {useEffect, useMemo, useRef, useState} from "react";
import {Box, Text, useInput} from "ink";
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

export default function PromptInput({ready, skills, busy, approval, onRequest, onApproval, onShutdown}: Props): React.ReactElement {
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
    if (!key.ctrl && !key.meta && key.upArrow) {
      if (matches.length) {
        setSelected((index) => (index - 1 + matches.length) % matches.length);
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
  return (
    <Box flexDirection="column" marginTop={1}>
      <ApprovalPanel approval={approval} />
      <CompletionList items={matches} selected={selected} />
      <Box flexDirection="column" borderStyle="round" borderColor={busy ? "yellow" : "cyan"} paddingX={1}>
        <FooterStatus ready={ready} />
        <Box>
          <Text color="yellow" bold>{ready?.mode === "bash" ? "! " : "> "}</Text>
          <Text>{characters.slice(0, cursor).join("")}</Text>
          <Text color="cyan">{busy ? "  [processing]" : "|"}</Text>
          <Text>{characters.slice(cursor).join("")}</Text>
        </Box>
      </Box>
    </Box>
  );
}
