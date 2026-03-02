/**
 * @license
 * Copyright 2026 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */
import { MessageBusType, ToolConfirmationOutcome, Scheduler, ROOT_SCHEDULER_ID, } from '@google/gemini-cli-core';
import { useCallback, useState, useMemo, useEffect, useRef } from 'react';
/**
 * Modern tool scheduler hook using the event-driven Core Scheduler.
 *
 * This hook acts as an Adapter between the new MessageBus-driven Core
 * and the legacy callback-based UI components.
 */
export function useToolExecutionScheduler(onComplete, config, getPreferredEditor) {
    // State stores tool calls organized by their originating schedulerId
    const [toolCallsMap, setToolCallsMap] = useState({});
    const [lastToolOutputTime, setLastToolOutputTime] = useState(0);
    const messageBus = useMemo(() => config.getMessageBus(), [config]);
    const onCompleteRef = useRef(onComplete);
    useEffect(() => {
        onCompleteRef.current = onComplete;
    }, [onComplete]);
    const getPreferredEditorRef = useRef(getPreferredEditor);
    useEffect(() => {
        getPreferredEditorRef.current = getPreferredEditor;
    }, [getPreferredEditor]);
    const scheduler = useMemo(() => new Scheduler({
        config,
        messageBus,
        getPreferredEditor: () => getPreferredEditorRef.current(),
        schedulerId: ROOT_SCHEDULER_ID,
    }), [config, messageBus]);
    const internalAdaptToolCalls = useCallback((coreCalls, prevTracked) => adaptToolCalls(coreCalls, prevTracked, messageBus), [messageBus]);
    useEffect(() => {
        const handler = (event) => {
            // Update output timer for UI spinners (Side Effect)
            if (event.toolCalls.some((tc) => tc.status === 'executing')) {
                setLastToolOutputTime(Date.now());
            }
            setToolCallsMap((prev) => {
                const adapted = internalAdaptToolCalls(event.toolCalls, prev[event.schedulerId] ?? []);
                return {
                    ...prev,
                    [event.schedulerId]: adapted,
                };
            });
        };
        messageBus.subscribe(MessageBusType.TOOL_CALLS_UPDATE, handler);
        return () => {
            messageBus.unsubscribe(MessageBusType.TOOL_CALLS_UPDATE, handler);
        };
    }, [messageBus, internalAdaptToolCalls]);
    const schedule = useCallback(async (request, signal) => {
        // Clear state for new run
        setToolCallsMap({});
        // 1. Await Core Scheduler directly
        const results = await scheduler.schedule(request, signal);
        // 2. Trigger legacy reinjection logic (useGeminiStream loop)
        // Since this hook instance owns the "root" scheduler, we always trigger
        // onComplete when it finishes its batch.
        await onCompleteRef.current(results);
        return results;
    }, [scheduler]);
    const cancelAll = useCallback((_signal) => {
        scheduler.cancelAll();
    }, [scheduler]);
    const markToolsAsSubmitted = useCallback((callIdsToMark) => {
        setToolCallsMap((prevMap) => {
            const nextMap = { ...prevMap };
            for (const [sid, calls] of Object.entries(nextMap)) {
                nextMap[sid] = calls.map((tc) => callIdsToMark.includes(tc.request.callId)
                    ? { ...tc, responseSubmittedToGemini: true }
                    : tc);
            }
            return nextMap;
        });
    }, []);
    // Flatten the map for the UI components that expect a single list of tools.
    const toolCalls = useMemo(() => Object.values(toolCallsMap).flat(), [toolCallsMap]);
    // Provide a setter that maintains compatibility with legacy [].
    const setToolCallsForDisplay = useCallback((action) => {
        setToolCallsMap((prev) => {
            const currentFlattened = Object.values(prev).flat();
            const nextFlattened = typeof action === 'function' ? action(currentFlattened) : action;
            if (nextFlattened.length === 0) {
                return {};
            }
            // Re-group by schedulerId to preserve multi-scheduler state
            const nextMap = {};
            for (const call of nextFlattened) {
                // All tool calls should have a schedulerId from the core.
                // Default to ROOT_SCHEDULER_ID as a safeguard.
                const sid = call.schedulerId ?? ROOT_SCHEDULER_ID;
                if (!nextMap[sid]) {
                    nextMap[sid] = [];
                }
                nextMap[sid].push(call);
            }
            return nextMap;
        });
    }, []);
    return [
        toolCalls,
        schedule,
        markToolsAsSubmitted,
        setToolCallsForDisplay,
        cancelAll,
        lastToolOutputTime,
    ];
}
/**
 * ADAPTER: Merges UI metadata (submitted flag) and injects legacy callbacks.
 */
function adaptToolCalls(coreCalls, prevTracked, messageBus) {
    const prevMap = new Map(prevTracked.map((t) => [t.request.callId, t]));
    return coreCalls.map((coreCall) => {
        const prev = prevMap.get(coreCall.request.callId);
        const responseSubmittedToGemini = prev?.responseSubmittedToGemini ?? false;
        // Inject onConfirm adapter for tools awaiting approval.
        // The Core provides data-only (serializable) confirmationDetails. We must
        // inject the legacy callback function that proxies responses back to the
        // MessageBus.
        if (coreCall.status === 'awaiting_approval' && coreCall.correlationId) {
            const correlationId = coreCall.correlationId;
            return {
                ...coreCall,
                confirmationDetails: {
                    ...coreCall.confirmationDetails,
                    onConfirm: async (outcome, payload) => {
                        await messageBus.publish({
                            type: MessageBusType.TOOL_CONFIRMATION_RESPONSE,
                            correlationId,
                            confirmed: outcome !== ToolConfirmationOutcome.Cancel,
                            requiresUserConfirmation: false,
                            outcome,
                            payload,
                        });
                    },
                },
                responseSubmittedToGemini,
            };
        }
        return {
            ...coreCall,
            responseSubmittedToGemini,
        };
    });
}
//# sourceMappingURL=useToolExecutionScheduler.js.map