/**
 * @license
 * Copyright 2026 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */
import { type ToolCall, type Status as CoreStatus } from '@google/gemini-cli-core';
import { ToolCallStatus, type HistoryItemToolGroup } from '../types.js';
export declare function mapCoreStatusToDisplayStatus(coreStatus: CoreStatus): ToolCallStatus;
/**
 * Transforms `ToolCall` objects into `HistoryItemToolGroup` objects for UI
 * display. This is a pure projection layer and does not track interaction
 * state.
 */
export declare function mapToDisplay(toolOrTools: ToolCall[] | ToolCall, options?: {
    borderTop?: boolean;
    borderBottom?: boolean;
}): HistoryItemToolGroup;
