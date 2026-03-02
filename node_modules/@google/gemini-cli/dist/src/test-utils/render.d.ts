/**
 * @license
 * Copyright 2026 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */
import { render as inkRender } from 'ink-testing-library';
import type React from 'react';
import { LoadedSettings } from '../config/settings.js';
import { type UIState } from '../ui/contexts/UIStateContext.js';
import { type UIActions } from '../ui/contexts/UIActionsContext.js';
import { type Config } from '@google/gemini-cli-core';
import { FakePersistentState } from './persistentStateFake.js';
import { type AppState } from '../ui/contexts/AppContext.js';
export declare const persistentStateMock: FakePersistentState;
export declare const render: (tree: React.ReactElement, terminalWidth?: number) => ReturnType<typeof inkRender>;
export declare const simulateClick: (stdin: ReturnType<typeof inkRender>["stdin"], col: number, row: number, button?: 0 | 1 | 2) => Promise<void>;
export declare const mockSettings: LoadedSettings;
export declare const mockAppState: AppState;
export declare const renderWithProviders: (component: React.ReactElement, { shellFocus, settings, uiState: providedUiState, width, mouseEventsEnabled, config, useAlternateBuffer, uiActions, persistentState, appState, }?: {
    shellFocus?: boolean;
    settings?: LoadedSettings;
    uiState?: Partial<UIState>;
    width?: number;
    mouseEventsEnabled?: boolean;
    config?: Config;
    useAlternateBuffer?: boolean;
    uiActions?: Partial<UIActions>;
    persistentState?: {
        get?: typeof persistentStateMock.get;
        set?: typeof persistentStateMock.set;
    };
    appState?: AppState;
}) => ReturnType<typeof render> & {
    simulateClick: typeof simulateClick;
};
export declare function renderHook<Result, Props>(renderCallback: (props: Props) => Result, options?: {
    initialProps?: Props;
    wrapper?: React.ComponentType<{
        children: React.ReactNode;
    }>;
}): {
    result: {
        current: Result;
    };
    rerender: (props?: Props) => void;
    unmount: () => void;
};
export declare function renderHookWithProviders<Result, Props>(renderCallback: (props: Props) => Result, options?: {
    initialProps?: Props;
    wrapper?: React.ComponentType<{
        children: React.ReactNode;
    }>;
    shellFocus?: boolean;
    settings?: LoadedSettings;
    uiState?: Partial<UIState>;
    width?: number;
    mouseEventsEnabled?: boolean;
    config?: Config;
    useAlternateBuffer?: boolean;
}): {
    result: {
        current: Result;
    };
    rerender: (props?: Props) => void;
    unmount: () => void;
};
