import { jsx as _jsx } from "react/jsx-runtime";
/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */
import { render } from '../../test-utils/render.js';
import { ShellInputPrompt } from './ShellInputPrompt.js';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { ShellExecutionService } from '@google/gemini-cli-core';
import { useUIActions } from '../contexts/UIActionsContext.js';
// Mock useUIActions
vi.mock('../contexts/UIActionsContext.js', () => ({
    useUIActions: vi.fn(),
}));
// Mock useKeypress
const mockUseKeypress = vi.fn();
vi.mock('../hooks/useKeypress.js', () => ({
    useKeypress: (handler, options) => mockUseKeypress(handler, options),
}));
// Mock ShellExecutionService
vi.mock('@google/gemini-cli-core', async () => {
    const actual = await vi.importActual('@google/gemini-cli-core');
    return {
        ...actual,
        ShellExecutionService: {
            writeToPty: vi.fn(),
            scrollPty: vi.fn(),
        },
    };
});
describe('ShellInputPrompt', () => {
    const mockWriteToPty = vi.mocked(ShellExecutionService.writeToPty);
    const mockScrollPty = vi.mocked(ShellExecutionService.scrollPty);
    const mockHandleWarning = vi.fn();
    beforeEach(() => {
        vi.clearAllMocks();
        vi.mocked(useUIActions).mockReturnValue({
            handleWarning: mockHandleWarning,
        });
    });
    it('renders nothing', () => {
        const { lastFrame } = render(_jsx(ShellInputPrompt, { activeShellPtyId: 1, focus: true }));
        expect(lastFrame()).toBe('');
    });
    it('sends tab to pty', () => {
        render(_jsx(ShellInputPrompt, { activeShellPtyId: 1, focus: true }));
        const handler = mockUseKeypress.mock.calls[0][0];
        handler({
            name: 'tab',
            shift: false,
            alt: false,
            ctrl: false,
            cmd: false,
            sequence: '\t',
        });
        expect(mockWriteToPty).toHaveBeenCalledWith(1, '\t');
    });
    it.each([
        ['a', 'a'],
        ['b', 'b'],
    ])('handles keypress input: %s', (name, sequence) => {
        render(_jsx(ShellInputPrompt, { activeShellPtyId: 1, focus: true }));
        // Get the registered handler
        const handler = mockUseKeypress.mock.calls[0][0];
        // Simulate keypress
        handler({
            name,
            shift: false,
            alt: false,
            ctrl: false,
            cmd: false,
            sequence,
        });
        expect(mockWriteToPty).toHaveBeenCalledWith(1, sequence);
    });
    it.each([
        ['up', -1],
        ['down', 1],
    ])('handles scroll %s (Command.SCROLL_%s)', (key, direction) => {
        render(_jsx(ShellInputPrompt, { activeShellPtyId: 1, focus: true }));
        const handler = mockUseKeypress.mock.calls[0][0];
        handler({ name: key, shift: true, alt: false, ctrl: false, cmd: false });
        expect(mockScrollPty).toHaveBeenCalledWith(1, direction);
    });
    it.each([
        ['pageup', -15],
        ['pagedown', 15],
    ])('handles page scroll %s (Command.PAGE_%s) with default size', (key, expectedScroll) => {
        render(_jsx(ShellInputPrompt, { activeShellPtyId: 1, focus: true }));
        const handler = mockUseKeypress.mock.calls[0][0];
        handler({ name: key, shift: false, alt: false, ctrl: false, cmd: false });
        expect(mockScrollPty).toHaveBeenCalledWith(1, expectedScroll);
    });
    it('respects scrollPageSize prop', () => {
        render(_jsx(ShellInputPrompt, { activeShellPtyId: 1, focus: true, scrollPageSize: 10 }));
        const handler = mockUseKeypress.mock.calls[0][0];
        // PageDown
        handler({
            name: 'pagedown',
            shift: false,
            alt: false,
            ctrl: false,
            cmd: false,
        });
        expect(mockScrollPty).toHaveBeenCalledWith(1, 10);
        // PageUp
        handler({
            name: 'pageup',
            shift: false,
            alt: false,
            ctrl: false,
            cmd: false,
        });
        expect(mockScrollPty).toHaveBeenCalledWith(1, -10);
    });
    it('does not handle input when not focused', () => {
        render(_jsx(ShellInputPrompt, { activeShellPtyId: 1, focus: false }));
        const handler = mockUseKeypress.mock.calls[0][0];
        handler({
            name: 'a',
            shift: false,
            alt: false,
            ctrl: false,
            cmd: false,
            sequence: 'a',
        });
        expect(mockWriteToPty).not.toHaveBeenCalled();
    });
    it('does not handle input when no active shell', () => {
        render(_jsx(ShellInputPrompt, { activeShellPtyId: null, focus: true }));
        const handler = mockUseKeypress.mock.calls[0][0];
        handler({
            name: 'a',
            shift: false,
            alt: false,
            ctrl: false,
            cmd: false,
            sequence: 'a',
        });
        expect(mockWriteToPty).not.toHaveBeenCalled();
    });
    it('ignores Command.UNFOCUS_SHELL (Shift+Tab) to allow focus navigation', () => {
        render(_jsx(ShellInputPrompt, { activeShellPtyId: 1, focus: true }));
        const handler = mockUseKeypress.mock.calls[0][0];
        const result = handler({
            name: 'tab',
            shift: true,
            alt: false,
            ctrl: false,
            cmd: false,
        });
        expect(result).toBe(false);
        expect(mockWriteToPty).not.toHaveBeenCalled();
    });
});
//# sourceMappingURL=ShellInputPrompt.test.js.map