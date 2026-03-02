/**
 * @license
 * Copyright 2026 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */
import { type IndividualToolCallDisplay } from '../types.js';
export interface ConfirmingToolState {
    tool: IndividualToolCallDisplay;
    index: number;
    total: number;
}
/**
 * Selects the "Head" of the confirmation queue.
 * Returns the first tool in the pending state that requires confirmation.
 */
export declare function useConfirmingTool(): ConfirmingToolState | null;
