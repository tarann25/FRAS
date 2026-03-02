import { jsx as _jsx } from "react/jsx-runtime";
/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */
import { describe, it, expect, vi, beforeAll } from 'vitest';
import { ToolConfirmationMessage } from './ToolConfirmationMessage.js';
import { initializeShellParsers } from '@google/gemini-cli-core';
import { renderWithProviders } from '../../../test-utils/render.js';
describe('ToolConfirmationMessage Redirection', () => {
    beforeAll(async () => {
        await initializeShellParsers();
    });
    const mockConfig = {
        isTrustedFolder: () => true,
        getIdeMode: () => false,
    };
    it('should display redirection warning and tip for redirected commands', () => {
        const confirmationDetails = {
            type: 'exec',
            title: 'Confirm Shell Command',
            command: 'echo "hello" > test.txt',
            rootCommand: 'echo, redirection (>)',
            rootCommands: ['echo'],
            onConfirm: vi.fn(),
        };
        const { lastFrame } = renderWithProviders(_jsx(ToolConfirmationMessage, { callId: "test-call-id", confirmationDetails: confirmationDetails, config: mockConfig, availableTerminalHeight: 30, terminalWidth: 100 }));
        const output = lastFrame();
        expect(output).toMatchSnapshot();
    });
});
//# sourceMappingURL=RedirectionConfirmation.test.js.map