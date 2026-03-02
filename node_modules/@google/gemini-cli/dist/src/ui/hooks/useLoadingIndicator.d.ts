/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */
import { StreamingState } from '../types.js';
import { type RetryAttemptPayload } from '@google/gemini-cli-core';
export interface UseLoadingIndicatorProps {
    streamingState: StreamingState;
    shouldShowFocusHint: boolean;
    retryStatus: RetryAttemptPayload | null;
    customWittyPhrases?: string[];
}
export declare const useLoadingIndicator: ({ streamingState, shouldShowFocusHint, retryStatus, customWittyPhrases, }: UseLoadingIndicatorProps) => {
    elapsedTime: number;
    currentLoadingPhrase: string | undefined;
};
