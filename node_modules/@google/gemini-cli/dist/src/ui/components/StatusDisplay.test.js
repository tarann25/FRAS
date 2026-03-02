import { jsxs as _jsxs, jsx as _jsx } from "react/jsx-runtime";
/**
 * @license
 * Copyright 2026 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */
import { describe, it, expect, vi, afterEach } from 'vitest';
import { render } from '../../test-utils/render.js';
import { Text } from 'ink';
import { StatusDisplay } from './StatusDisplay.js';
import { UIStateContext } from '../contexts/UIStateContext.js';
import { ConfigContext } from '../contexts/ConfigContext.js';
import { SettingsContext } from '../contexts/SettingsContext.js';
import { createMockSettings } from '../../test-utils/settings.js';
// Mock child components to simplify testing
vi.mock('./ContextSummaryDisplay.js', () => ({
    ContextSummaryDisplay: (props) => (_jsxs(Text, { children: ["Mock Context Summary Display (Skills: ", props.skillCount, ", Shells:", ' ', props.backgroundProcessCount, ")"] })),
}));
vi.mock('./HookStatusDisplay.js', () => ({
    HookStatusDisplay: () => _jsx(Text, { children: "Mock Hook Status Display" }),
}));
// Create mock context providers
const createMockUIState = (overrides = {}) => ({
    ctrlCPressedOnce: false,
    transientMessage: null,
    ctrlDPressedOnce: false,
    showEscapePrompt: false,
    shortcutsHelpVisible: false,
    queueErrorMessage: null,
    activeHooks: [],
    ideContextState: null,
    geminiMdFileCount: 0,
    contextFileNames: [],
    backgroundShellCount: 0,
    buffer: { text: '' },
    history: [{ id: 1, type: 'user', text: 'test' }],
    ...overrides,
});
const createMockConfig = (overrides = {}) => ({
    getMcpClientManager: vi.fn().mockImplementation(() => ({
        getBlockedMcpServers: vi.fn(() => []),
        getMcpServers: vi.fn(() => ({})),
    })),
    getSkillManager: vi.fn().mockImplementation(() => ({
        getSkills: vi.fn(() => ['skill1', 'skill2']),
        getDisplayableSkills: vi.fn(() => ['skill1', 'skill2']),
    })),
    ...overrides,
});
const renderStatusDisplay = (props = { hideContextSummary: false }, uiState = createMockUIState(), settings = createMockSettings(), config = createMockConfig()) => render(_jsx(ConfigContext.Provider, { value: config, children: _jsx(SettingsContext.Provider, { value: settings, children: _jsx(UIStateContext.Provider, { value: uiState, children: _jsx(StatusDisplay, { ...props }) }) }) }));
describe('StatusDisplay', () => {
    const originalEnv = process.env;
    afterEach(() => {
        process.env = { ...originalEnv };
        delete process.env['GEMINI_SYSTEM_MD'];
        vi.restoreAllMocks();
    });
    it('renders nothing by default if context summary is hidden via props', () => {
        const { lastFrame } = renderStatusDisplay({ hideContextSummary: true });
        expect(lastFrame()).toBe('');
    });
    it('renders ContextSummaryDisplay by default', () => {
        const { lastFrame } = renderStatusDisplay();
        expect(lastFrame()).toMatchSnapshot();
    });
    it('renders system md indicator if env var is set', () => {
        process.env['GEMINI_SYSTEM_MD'] = 'true';
        const { lastFrame } = renderStatusDisplay();
        expect(lastFrame()).toMatchSnapshot();
    });
    it('renders HookStatusDisplay when hooks are active', () => {
        const uiState = createMockUIState({
            activeHooks: [{ name: 'hook', eventName: 'event' }],
        });
        const { lastFrame } = renderStatusDisplay({ hideContextSummary: false }, uiState);
        expect(lastFrame()).toMatchSnapshot();
    });
    it('does NOT render HookStatusDisplay if notifications are disabled in settings', () => {
        const uiState = createMockUIState({
            activeHooks: [{ name: 'hook', eventName: 'event' }],
        });
        const settings = createMockSettings({
            hooksConfig: { notifications: false },
        });
        const { lastFrame } = renderStatusDisplay({ hideContextSummary: false }, uiState, settings);
        expect(lastFrame()).toMatchSnapshot();
    });
    it('hides ContextSummaryDisplay if configured in settings', () => {
        const settings = createMockSettings({
            ui: { hideContextSummary: true },
        });
        const { lastFrame } = renderStatusDisplay({ hideContextSummary: false }, undefined, settings);
        expect(lastFrame()).toBe('');
    });
    it('passes backgroundShellCount to ContextSummaryDisplay', () => {
        const uiState = createMockUIState({
            backgroundShellCount: 3,
        });
        const { lastFrame } = renderStatusDisplay({ hideContextSummary: false }, uiState);
        expect(lastFrame()).toContain('Shells: 3');
    });
});
//# sourceMappingURL=StatusDisplay.test.js.map