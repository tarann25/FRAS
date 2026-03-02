/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */
import type { Config, ToolCallRequestInfo, EditorType, CompletedToolCall, ExecutingToolCall, ScheduledToolCall, ValidatingToolCall, WaitingToolCall, CancelledToolCall } from '@google/gemini-cli-core';
export type ScheduleFn = (request: ToolCallRequestInfo | ToolCallRequestInfo[], signal: AbortSignal) => Promise<void>;
export type MarkToolsAsSubmittedFn = (callIds: string[]) => void;
export type CancelAllFn = (signal: AbortSignal) => void;
export type TrackedScheduledToolCall = ScheduledToolCall & {
    responseSubmittedToGemini?: boolean;
};
export type TrackedValidatingToolCall = ValidatingToolCall & {
    responseSubmittedToGemini?: boolean;
};
export type TrackedWaitingToolCall = WaitingToolCall & {
    responseSubmittedToGemini?: boolean;
};
export type TrackedExecutingToolCall = ExecutingToolCall & {
    responseSubmittedToGemini?: boolean;
};
export type TrackedCompletedToolCall = CompletedToolCall & {
    responseSubmittedToGemini?: boolean;
};
export type TrackedCancelledToolCall = CancelledToolCall & {
    responseSubmittedToGemini?: boolean;
};
export type TrackedToolCall = TrackedScheduledToolCall | TrackedValidatingToolCall | TrackedWaitingToolCall | TrackedExecutingToolCall | TrackedCompletedToolCall | TrackedCancelledToolCall;
/**
 * Legacy scheduler implementation based on CoreToolScheduler callbacks.
 *
 * This is currently the default implementation used by useGeminiStream.
 * It will be phased out once the event-driven scheduler migration is complete.
 */
export declare function useReactToolScheduler(onComplete: (tools: CompletedToolCall[]) => Promise<void>, config: Config, getPreferredEditor: () => EditorType | undefined): [
    TrackedToolCall[],
    ScheduleFn,
    MarkToolsAsSubmittedFn,
    React.Dispatch<React.SetStateAction<TrackedToolCall[]>>,
    CancelAllFn,
    number
];
