/**
 * TypeScript SDK for the agent harness. Protocol types are generated from protocol/ — never
 * hand-write them here.
 */
export type * from "./protocol.gen.ts";

/** Must match protocol/VERSION. */
export const PROTOCOL_VERSION = 1;

import type { Command, Event } from "./protocol.gen.ts";

/** Command `type` discriminants, useful for exhaustive switches in clients. */
export type CommandType = Command["type"];
/** Event `type` discriminants, useful for exhaustive switches in clients. */
export type EventType = Event["type"];
