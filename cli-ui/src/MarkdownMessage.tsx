import React from "react";
import {Text} from "ink";
import {lexer} from "marked";
import type {MessageRole, UiMessage} from "./protocol.js";

export type MarkdownRow = {
  text: string;
  kind: "normal" | "heading" | "bullet" | "code" | "quote" | "rule";
};

export type DisplayRow = MarkdownRow & {
  messageId: string;
  role: MessageRole;
  prefix: string;
};

type TokenLike = {
  type?: string;
  text?: string;
  raw?: string;
  depth?: number;
  lang?: string;
  ordered?: boolean;
  start?: number;
  items?: TokenLike[];
  tokens?: TokenLike[];
  header?: Array<{text?: string}>;
  rows?: Array<Array<{text?: string}>>;
};

function inlineText(token: TokenLike): string {
  return token.text || token.raw || "";
}

function tokenRows(tokens: TokenLike[]): MarkdownRow[] {
  const rows: MarkdownRow[] = [];
  for (const token of tokens) {
    switch (token.type) {
      case "space":
        continue;
      case "heading":
        rows.push({text: `${"#".repeat(token.depth || 1)} ${inlineText(token)}`, kind: "heading"});
        break;
      case "paragraph":
      case "text":
        rows.push(...inlineText(token).split("\n").map((text) => ({text, kind: "normal" as const})));
        break;
      case "list":
        (token.items || []).forEach((item, index) => {
          const marker = token.ordered ? `${(token.start || 1) + index}.` : "-";
          const itemText = inlineText(item).split("\n");
          rows.push({text: `${marker} ${itemText[0] || ""}`, kind: "bullet"});
          rows.push(...itemText.slice(1).map((text) => ({text: `  ${text}`, kind: "normal" as const})));
        });
        break;
      case "code":
        if (token.lang) rows.push({text: `[${token.lang}]`, kind: "code"});
        rows.push(...inlineText(token).split("\n").map((text) => ({text, kind: "code" as const})));
        break;
      case "blockquote":
        rows.push(...(token.tokens ? tokenRows(token.tokens) : inlineText(token).split("\n").map((text) => ({text, kind: "normal" as const}))).map((row) => ({...row, kind: "quote" as const, text: row.text})));
        break;
      case "hr":
        rows.push({text: "--------------------------------", kind: "rule"});
        break;
      case "br":
        rows.push({text: "", kind: "normal"});
        break;
      case "table": {
        const header = (token.header || []).map((cell) => cell.text || "");
        if (header.length) rows.push({text: `| ${header.join(" | ")} |`, kind: "normal"});
        if (header.length) rows.push({text: `| ${header.map(() => "---").join(" | ")} |`, kind: "rule"});
        for (const row of token.rows || []) rows.push({text: `| ${row.map((cell) => cell.text || "").join(" | ")} |`, kind: "normal"});
        break;
      }
      default:
        if (token.tokens) rows.push(...tokenRows(token.tokens));
        else if (inlineText(token)) rows.push({text: inlineText(token), kind: "normal"});
    }
  }
  return rows.length ? rows : [{text: "", kind: "normal"}];
}

export function markdownRows(content: string): MarkdownRow[] {
  try {
    return tokenRows(lexer(content) as unknown as TokenLike[]);
  } catch {
    return content.split("\n").map((text) => ({text, kind: "normal" as const}));
  }
}

export function displayRows(messages: UiMessage[], columns: number): DisplayRow[] {
  const rows: DisplayRow[] = [];
  for (const message of messages) {
    const messageRows = markdownRows(message.content);
    const availableWidth = Math.max(8, columns - 1);
    messageRows.forEach((row, index) => {
      const chunks = row.text ? splitWidth(row.text, availableWidth) : [""];
      chunks.forEach((text, chunkIndex) => rows.push({
        ...row,
        // Pad user rows to the column width so their background highlight spans
        // the whole line instead of stopping at the last visible character.
        text: message.role === "user" ? text.padEnd(availableWidth, " ") : text,
        messageId: message.id,
        role: message.role,
        prefix: "",
      }));
    });
    if (message.role !== "assistant" && message.role !== "user") rows.push({text: "", kind: "normal", messageId: `${message.id}-gap`, role: message.role, prefix: ""});
  }
  return rows;
}

function splitWidth(text: string, width: number): string[] {
  const characters = Array.from(text);
  const chunks: string[] = [];
  for (let index = 0; index < characters.length; index += width) chunks.push(characters.slice(index, index + width).join(""));
  return chunks.length ? chunks : [""];
}

function inlineParts(text: string): React.ReactNode[] {
  const pattern = /(\*\*[^*]+\*\*|__[^_]+__|~~[^~]+~~|`[^`]+`|\[[^\]]+\]\([^\)]+\)|\*[^*]+\*|_[^_]+_)/g;
  const parts: React.ReactNode[] = [];
  let last = 0;
  for (const match of text.matchAll(pattern)) {
    const index = match.index ?? 0;
    if (index > last) parts.push(text.slice(last, index));
    const value = match[0];
    if (value.startsWith("**") || value.startsWith("__")) parts.push(<Text key={`${index}-bold`} bold>{value.slice(2, -2)}</Text>);
    else if (value.startsWith("~~")) parts.push(<Text key={`${index}-strike`} dimColor>{value.slice(2, -2)}</Text>);
    else if (value.startsWith("`")) parts.push(<Text key={`${index}-code`} color="yellow">{value.slice(1, -1)}</Text>);
    else if (value.startsWith("[")) {
      const link = value.match(/^\[([^\]]+)\]\(([^)]+)\)$/);
      parts.push(<Text key={`${index}-link`} color="cyan" underline>{link ? `${link[1]} (${link[2]})` : value}</Text>);
    } else parts.push(<Text key={`${index}-italic`} italic>{value.slice(1, -1)}</Text>);
    last = index + value.length;
  }
  if (last < text.length) parts.push(text.slice(last));
  return parts;
}

export function MarkdownRowView({row}: {row: DisplayRow}): React.ReactElement {
  const isUser = row.role === "user";
  // User input gets a full-line dark background to stand out from agent output.
  const backgroundColor = isUser ? "#555" : undefined;
  const userText = (color?: string) => isUser ? {color: "white", backgroundColor} : {color};
  if (row.kind === "code") return <Text {...userText("gray")}>{row.prefix}{row.text}</Text>;
  if (row.kind === "heading") return <Text {...userText("cyan")} bold>{row.prefix}{inlineParts(row.text)}</Text>;
  if (row.kind === "quote") return <Text {...userText(undefined)} dimColor={!isUser}>{row.prefix}| {inlineParts(row.text)}</Text>;
  if (row.kind === "rule") return <Text {...userText(undefined)} dimColor={!isUser}>{row.prefix}{row.text}</Text>;
  const color = isUser ? "white" : row.role === "error" ? "red" : row.role === "tool" ? "magenta" : undefined;
  return <Text color={color} backgroundColor={isUser ? "#555" : undefined}>{row.prefix}{inlineParts(row.text)}</Text>;
}
