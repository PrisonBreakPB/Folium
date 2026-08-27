import React from "react";
import {Text} from "ink";
import type {EventMessage} from "./protocol.js";

type Props = {
  ready: EventMessage | null;
};

export default function FooterStatus({ready}: Props): React.ReactElement {
  const status = ready
    ? `Model: ${String(ready.model)}  Mode: ${String(ready.mode)}  Session: ${ready.session_id || "(unsaved)"}  Workspace: ${String(ready.workspace)}`
    : "Connecting to Python Agent...";
  return <Text dimColor wrap="truncate-end">{status}</Text>;
}
