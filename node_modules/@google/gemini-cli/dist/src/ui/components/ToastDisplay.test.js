import { jsx as _jsx } from "react/jsx-runtime";
/**
 * @license
 * Copyright 2026 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */
import { describe, it, expect, vi, afterEach } from 'vitest';
import { renderWithProviders } from '../../test-utils/render.js';
import { ToastDisplay, shouldShowToast } from './ToastDisplay.js';
import { TransientMessageType } from '../../utils/events.js';
import {} from '../contexts/UIStateContext.js';
import {} from './shared/text-buffer.js';
import {} from '../types.js';
const renderToastDisplay = (uiState = {}) => renderWithProviders(_jsx(ToastDisplay, {}), {
    uiState: {
        buffer: { text: '' },
        history: [],
        ...uiState,
    },
});
describe('ToastDisplay', () => {
    afterEach(() => {
        vi.restoreAllMocks();
    });
    describe('shouldShowToast', () => {
        const baseState = {
            ctrlCPressedOnce: false,
            transientMessage: null,
            ctrlDPressedOnce: false,
            showEscapePrompt: false,
            buffer: { text: '' },
            history: [],
            queueErrorMessage: null,
        };
        it('returns false for default state', () => {
            expect(shouldShowToast(baseState)).toBe(false);
        });
        it('returns true when ctrlCPressedOnce is true', () => {
            expect(shouldShowToast({ ...baseState, ctrlCPressedOnce: true })).toBe(true);
        });
        it('returns true when transientMessage is present', () => {
            expect(shouldShowToast({
                ...baseState,
                transientMessage: { text: 'test', type: TransientMessageType.Hint },
            })).toBe(true);
        });
        it('returns true when ctrlDPressedOnce is true', () => {
            expect(shouldShowToast({ ...baseState, ctrlDPressedOnce: true })).toBe(true);
        });
        it('returns true when showEscapePrompt is true and buffer is NOT empty', () => {
            expect(shouldShowToast({
                ...baseState,
                showEscapePrompt: true,
                buffer: { text: 'some text' },
            })).toBe(true);
        });
        it('returns true when showEscapePrompt is true and history is NOT empty', () => {
            expect(shouldShowToast({
                ...baseState,
                showEscapePrompt: true,
                history: [{ id: '1' }],
            })).toBe(true);
        });
        it('returns false when showEscapePrompt is true but buffer and history are empty', () => {
            expect(shouldShowToast({
                ...baseState,
                showEscapePrompt: true,
            })).toBe(false);
        });
        it('returns true when queueErrorMessage is present', () => {
            expect(shouldShowToast({
                ...baseState,
                queueErrorMessage: 'error',
            })).toBe(true);
        });
    });
    it('renders nothing by default', () => {
        const { lastFrame } = renderToastDisplay();
        expect(lastFrame()).toBe('');
    });
    it('renders Ctrl+C prompt', () => {
        const { lastFrame } = renderToastDisplay({
            ctrlCPressedOnce: true,
        });
        expect(lastFrame()).toMatchSnapshot();
    });
    it('renders warning message', () => {
        const { lastFrame } = renderToastDisplay({
            transientMessage: {
                text: 'This is a warning',
                type: TransientMessageType.Warning,
            },
        });
        expect(lastFrame()).toMatchSnapshot();
    });
    it('renders hint message', () => {
        const { lastFrame } = renderToastDisplay({
            transientMessage: {
                text: 'This is a hint',
                type: TransientMessageType.Hint,
            },
        });
        expect(lastFrame()).toMatchSnapshot();
    });
    it('renders Ctrl+D prompt', () => {
        const { lastFrame } = renderToastDisplay({
            ctrlDPressedOnce: true,
        });
        expect(lastFrame()).toMatchSnapshot();
    });
    it('renders Escape prompt when buffer is empty', () => {
        const { lastFrame } = renderToastDisplay({
            showEscapePrompt: true,
            history: [{ id: 1, type: 'user', text: 'test' }],
        });
        expect(lastFrame()).toMatchSnapshot();
    });
    it('renders Escape prompt when buffer is NOT empty', () => {
        const { lastFrame } = renderToastDisplay({
            showEscapePrompt: true,
            buffer: { text: 'some text' },
        });
        expect(lastFrame()).toMatchSnapshot();
    });
    it('renders Queue Error Message', () => {
        const { lastFrame } = renderToastDisplay({
            queueErrorMessage: 'Queue Error',
        });
        expect(lastFrame()).toMatchSnapshot();
    });
});
//# sourceMappingURL=ToastDisplay.test.js.map