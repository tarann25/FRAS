/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */
import React from 'react';
import { ToolCallStatus } from '../../types.js';
import { type Config, type ToolResultDisplay } from '@google/gemini-cli-core';
export declare const STATUS_INDICATOR_WIDTH = 3;
/**
 * Returns true if the tool name corresponds to a shell tool.
 */
export declare function isShellTool(name: string): boolean;
/**
 * Returns true if the shell tool call is currently focusable.
 */
export declare function isThisShellFocusable(name: string, status: ToolCallStatus, config?: Config): boolean;
/**
 * Returns true if this specific shell tool call is currently focused.
 */
export declare function isThisShellFocused(name: string, status: ToolCallStatus, ptyId?: number, activeShellPtyId?: number | null, embeddedShellFocused?: boolean): boolean;
/**
 * Hook to manage focus hint state.
 */
export declare function useFocusHint(isThisShellFocusable: boolean, isThisShellFocused: boolean, resultDisplay: ToolResultDisplay | undefined): {
    shouldShowFocusHint: boolean;
};
/**
 * Component to render the focus hint.
 */
export declare const FocusHint: React.FC<{
    shouldShowFocusHint: boolean;
    isThisShellFocused: boolean;
}>;
export type TextEmphasis = 'high' | 'medium' | 'low';
type ToolStatusIndicatorProps = {
    status: ToolCallStatus;
    name: string;
};
export declare const ToolStatusIndicator: React.FC<ToolStatusIndicatorProps>;
type ToolInfoProps = {
    name: string;
    description: string;
    status: ToolCallStatus;
    emphasis: TextEmphasis;
};
export declare const ToolInfo: React.FC<ToolInfoProps>;
export declare const TrailingIndicator: React.FC;
export {};
