/**
 * @license
 * Copyright 2026 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */
import { type Config, type ToolCallRequestInfo, type ToolCall, type CompletedToolCall, type EditorType } from '@google/gemini-cli-core';
export type ScheduleFn = (request: ToolCallRequestInfo | ToolCallRequestInfo[], signal: AbortSignal) => Promise<CompletedToolCall[]>;
export type MarkToolsAsSubmittedFn = (callIds: string[]) => void;
export type CancelAllFn = (signal: AbortSignal) => void;
/**
 * The shape expected by useGeminiStream.
 * It matches the Core ToolCall structure + the UI metadata flag.
 */
export type TrackedToolCall = ToolCall & {
    responseSubmittedToGemini?: boolean;
};
/**
 * Modern tool scheduler hook using the event-driven Core Scheduler.
 *
 * This hook acts as an Adapter between the new MessageBus-driven Core
 * and the legacy callback-based UI components.
 */
export declare function useToolExecutionScheduler(onComplete: (tools: CompletedToolCall[]) => Promise<void>, config: Config, getPreferredEditor: () => EditorType | undefined): [
    TrackedToolCall[],
    ScheduleFn,
    MarkToolsAsSubmittedFn,
    React.Dispatch<React.SetStateAction<TrackedToolCall[]>>,
    CancelAllFn,
    number
];
