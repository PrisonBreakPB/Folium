import React, {useEffect, useMemo, useState} from "react";
import {Box, Text, useInput} from "ink";
import {displayRows, MarkdownRowView} from "./MarkdownMessage.js";
import type {UiMessage} from "./protocol.js";

type Props = {
  messages: UiMessage[];
  height: number;
  columns: number;
};

export default function MessageViewport({messages, height, columns}: Props): React.ReactElement {
  const [scrollOffset, setScrollOffset] = useState(0);
  const rows = useMemo(() => displayRows(messages, columns), [messages, columns]);
  const visibleHeight = Math.max(1, height - (scrollOffset > 0 ? 1 : 0));

  useEffect(() => {
    setScrollOffset((offset) => Math.min(offset, Math.max(0, rows.length - 1)));
  }, [rows.length]);

  useInput((_, key) => {
    const page = Math.max(1, Math.floor(visibleHeight / 2));
    if (key.pageUp || (key.ctrl && key.upArrow)) {
      setScrollOffset((offset) => Math.min(rows.length, offset + page));
    } else if (key.pageDown || (key.ctrl && key.downArrow)) {
      setScrollOffset((offset) => Math.max(0, offset - page));
    }
  });

  const end = rows.length - scrollOffset;
  const visibleRows = rows.slice(Math.max(0, end - visibleHeight), end);
  return (
    <Box flexDirection="column" height={height} overflow="hidden" marginTop={1}>
      {scrollOffset > 0 && <Text dimColor>...</Text>}
      {visibleRows.map((row, index) => <MarkdownRowView key={`${end - visibleRows.length + index}-${row.messageId}`} row={row} />)}
    </Box>
  );
}
