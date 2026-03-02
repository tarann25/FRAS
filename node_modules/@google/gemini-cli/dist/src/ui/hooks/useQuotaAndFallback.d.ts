/**
 * @license
 * Copyright 2026 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */
import { type Config, type FallbackIntent, type ValidationIntent, type UserTierId } from '@google/gemini-cli-core';
import { type UseHistoryManagerReturn } from './useHistoryManager.js';
import { type ProQuotaDialogRequest, type ValidationDialogRequest } from '../contexts/UIStateContext.js';
interface UseQuotaAndFallbackArgs {
    config: Config;
    historyManager: UseHistoryManagerReturn;
    userTier: UserTierId | undefined;
    setModelSwitchedFromQuotaError: (value: boolean) => void;
    onShowAuthSelection: () => void;
}
export declare function useQuotaAndFallback({ config, historyManager, userTier, setModelSwitchedFromQuotaError, onShowAuthSelection, }: UseQuotaAndFallbackArgs): {
    proQuotaRequest: ProQuotaDialogRequest | null;
    handleProQuotaChoice: (choice: FallbackIntent) => void;
    validationRequest: ValidationDialogRequest | null;
    handleValidationChoice: (choice: ValidationIntent) => void;
};
export {};
