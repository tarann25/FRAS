import { jsx as _jsx, jsxs as _jsxs, Fragment as _Fragment } from "react/jsx-runtime";
import { useMemo, useCallback } from 'react';
import { Box, Text } from 'ink';
import { DiffRenderer } from './DiffRenderer.js';
import { RenderInline } from '../../utils/InlineMarkdownRenderer.js';
import { ToolConfirmationOutcome, hasRedirection, debugLogger, } from '@google/gemini-cli-core';
import { useToolActions } from '../../contexts/ToolActionsContext.js';
import { RadioButtonSelect } from '../shared/RadioButtonSelect.js';
import { MaxSizedBox, MINIMUM_MAX_HEIGHT } from '../shared/MaxSizedBox.js';
import { sanitizeForDisplay } from '../../utils/textUtils.js';
import { useKeypress } from '../../hooks/useKeypress.js';
import { theme } from '../../semantic-colors.js';
import { useSettings } from '../../contexts/SettingsContext.js';
import { keyMatchers, Command } from '../../keyMatchers.js';
import { REDIRECTION_WARNING_NOTE_LABEL, REDIRECTION_WARNING_NOTE_TEXT, REDIRECTION_WARNING_TIP_LABEL, REDIRECTION_WARNING_TIP_TEXT, } from '../../textConstants.js';
import { AskUserDialog } from '../AskUserDialog.js';
import { ExitPlanModeDialog } from '../ExitPlanModeDialog.js';
export const ToolConfirmationMessage = ({ callId, confirmationDetails, config, isFocused = true, availableTerminalHeight, terminalWidth, }) => {
    const { confirm, isDiffingEnabled } = useToolActions();
    const settings = useSettings();
    const allowPermanentApproval = settings.merged.security.enablePermanentToolApproval;
    const handlesOwnUI = confirmationDetails.type === 'ask_user' ||
        confirmationDetails.type === 'exit_plan_mode';
    const isTrustedFolder = config.isTrustedFolder();
    const handleConfirm = useCallback((outcome, payload) => {
        void confirm(callId, outcome, payload).catch((error) => {
            debugLogger.error(`Failed to handle tool confirmation for ${callId}:`, error);
        });
    }, [confirm, callId]);
    useKeypress((key) => {
        if (!isFocused)
            return false;
        if (keyMatchers[Command.ESCAPE](key)) {
            handleConfirm(ToolConfirmationOutcome.Cancel);
            return true;
        }
        if (keyMatchers[Command.QUIT](key)) {
            // Return false to let ctrl-C bubble up to AppContainer for exit flow.
            // AppContainer will call cancelOngoingRequest which will cancel the tool.
            return false;
        }
        return false;
    }, { isActive: isFocused });
    const handleSelect = useCallback((item) => handleConfirm(item), [handleConfirm]);
    const getOptions = useCallback(() => {
        const options = [];
        if (confirmationDetails.type === 'edit') {
            if (!confirmationDetails.isModifying) {
                options.push({
                    label: 'Allow once',
                    value: ToolConfirmationOutcome.ProceedOnce,
                    key: 'Allow once',
                });
                if (isTrustedFolder) {
                    options.push({
                        label: 'Allow for this session',
                        value: ToolConfirmationOutcome.ProceedAlways,
                        key: 'Allow for this session',
                    });
                    if (allowPermanentApproval) {
                        options.push({
                            label: 'Allow for all future sessions',
                            value: ToolConfirmationOutcome.ProceedAlwaysAndSave,
                            key: 'Allow for all future sessions',
                        });
                    }
                }
                // We hide "Modify with external editor" if IDE mode is active AND
                // the IDE is actually capable of showing a diff (connected).
                if (!config.getIdeMode() || !isDiffingEnabled) {
                    options.push({
                        label: 'Modify with external editor',
                        value: ToolConfirmationOutcome.ModifyWithEditor,
                        key: 'Modify with external editor',
                    });
                }
                options.push({
                    label: 'No, suggest changes (esc)',
                    value: ToolConfirmationOutcome.Cancel,
                    key: 'No, suggest changes (esc)',
                });
            }
        }
        else if (confirmationDetails.type === 'exec') {
            options.push({
                label: 'Allow once',
                value: ToolConfirmationOutcome.ProceedOnce,
                key: 'Allow once',
            });
            if (isTrustedFolder) {
                options.push({
                    label: `Allow for this session`,
                    value: ToolConfirmationOutcome.ProceedAlways,
                    key: `Allow for this session`,
                });
                if (allowPermanentApproval) {
                    options.push({
                        label: `Allow for all future sessions`,
                        value: ToolConfirmationOutcome.ProceedAlwaysAndSave,
                        key: `Allow for all future sessions`,
                    });
                }
            }
            options.push({
                label: 'No, suggest changes (esc)',
                value: ToolConfirmationOutcome.Cancel,
                key: 'No, suggest changes (esc)',
            });
        }
        else if (confirmationDetails.type === 'info') {
            options.push({
                label: 'Allow once',
                value: ToolConfirmationOutcome.ProceedOnce,
                key: 'Allow once',
            });
            if (isTrustedFolder) {
                options.push({
                    label: 'Allow for this session',
                    value: ToolConfirmationOutcome.ProceedAlways,
                    key: 'Allow for this session',
                });
                if (allowPermanentApproval) {
                    options.push({
                        label: 'Allow for all future sessions',
                        value: ToolConfirmationOutcome.ProceedAlwaysAndSave,
                        key: 'Allow for all future sessions',
                    });
                }
            }
            options.push({
                label: 'No, suggest changes (esc)',
                value: ToolConfirmationOutcome.Cancel,
                key: 'No, suggest changes (esc)',
            });
        }
        else if (confirmationDetails.type === 'mcp') {
            // mcp tool confirmation
            options.push({
                label: 'Allow once',
                value: ToolConfirmationOutcome.ProceedOnce,
                key: 'Allow once',
            });
            if (isTrustedFolder) {
                options.push({
                    label: 'Allow tool for this session',
                    value: ToolConfirmationOutcome.ProceedAlwaysTool,
                    key: 'Allow tool for this session',
                });
                options.push({
                    label: 'Allow all server tools for this session',
                    value: ToolConfirmationOutcome.ProceedAlwaysServer,
                    key: 'Allow all server tools for this session',
                });
                if (allowPermanentApproval) {
                    options.push({
                        label: 'Allow tool for all future sessions',
                        value: ToolConfirmationOutcome.ProceedAlwaysAndSave,
                        key: 'Allow tool for all future sessions',
                    });
                }
            }
            options.push({
                label: 'No, suggest changes (esc)',
                value: ToolConfirmationOutcome.Cancel,
                key: 'No, suggest changes (esc)',
            });
        }
        return options;
    }, [
        confirmationDetails,
        isTrustedFolder,
        allowPermanentApproval,
        config,
        isDiffingEnabled,
    ]);
    const availableBodyContentHeight = useCallback(() => {
        if (availableTerminalHeight === undefined) {
            return undefined;
        }
        // Calculate the vertical space (in lines) consumed by UI elements
        // surrounding the main body content.
        const PADDING_OUTER_Y = 2; // Main container has `padding={1}` (top & bottom).
        const MARGIN_BODY_BOTTOM = 1; // margin on the body container.
        const HEIGHT_QUESTION = 1; // The question text is one line.
        const MARGIN_QUESTION_BOTTOM = 1; // Margin on the question container.
        const optionsCount = getOptions().length;
        const surroundingElementsHeight = PADDING_OUTER_Y +
            MARGIN_BODY_BOTTOM +
            HEIGHT_QUESTION +
            MARGIN_QUESTION_BOTTOM +
            optionsCount +
            1; // Reserve one line for 'ShowMoreLines' hint
        return Math.max(availableTerminalHeight - surroundingElementsHeight, 1);
    }, [availableTerminalHeight, getOptions]);
    const { question, bodyContent, options } = useMemo(() => {
        let bodyContent = null;
        let question = '';
        const options = getOptions();
        if (confirmationDetails.type === 'ask_user') {
            bodyContent = (_jsx(AskUserDialog, { questions: confirmationDetails.questions, onSubmit: (answers) => {
                    handleConfirm(ToolConfirmationOutcome.ProceedOnce, { answers });
                }, onCancel: () => {
                    handleConfirm(ToolConfirmationOutcome.Cancel);
                }, width: terminalWidth, availableHeight: availableBodyContentHeight() }));
            return { question: '', bodyContent, options: [] };
        }
        if (confirmationDetails.type === 'exit_plan_mode') {
            bodyContent = (_jsx(ExitPlanModeDialog, { planPath: confirmationDetails.planPath, onApprove: (approvalMode) => {
                    handleConfirm(ToolConfirmationOutcome.ProceedOnce, {
                        approved: true,
                        approvalMode,
                    });
                }, onFeedback: (feedback) => {
                    handleConfirm(ToolConfirmationOutcome.ProceedOnce, {
                        approved: false,
                        feedback,
                    });
                }, onCancel: () => {
                    handleConfirm(ToolConfirmationOutcome.Cancel);
                }, width: terminalWidth, availableHeight: availableBodyContentHeight() }));
            return { question: '', bodyContent, options: [] };
        }
        if (confirmationDetails.type === 'edit') {
            if (!confirmationDetails.isModifying) {
                question = `Apply this change?`;
            }
        }
        else if (confirmationDetails.type === 'exec') {
            const executionProps = confirmationDetails;
            if (executionProps.commands && executionProps.commands.length > 1) {
                question = `Allow execution of ${executionProps.commands.length} commands?`;
            }
            else {
                question = `Allow execution of: '${sanitizeForDisplay(executionProps.rootCommand)}'?`;
            }
        }
        else if (confirmationDetails.type === 'info') {
            question = `Do you want to proceed?`;
        }
        else if (confirmationDetails.type === 'mcp') {
            // mcp tool confirmation
            const mcpProps = confirmationDetails;
            question = `Allow execution of MCP tool "${mcpProps.toolName}" from server "${mcpProps.serverName}"?`;
        }
        if (confirmationDetails.type === 'edit') {
            if (!confirmationDetails.isModifying) {
                bodyContent = (_jsx(DiffRenderer, { diffContent: confirmationDetails.fileDiff, filename: confirmationDetails.fileName, availableTerminalHeight: availableBodyContentHeight(), terminalWidth: terminalWidth }));
            }
        }
        else if (confirmationDetails.type === 'exec') {
            const executionProps = confirmationDetails;
            const commandsToDisplay = executionProps.commands && executionProps.commands.length > 1
                ? executionProps.commands
                : [executionProps.command];
            const containsRedirection = commandsToDisplay.some((cmd) => hasRedirection(cmd));
            let bodyContentHeight = availableBodyContentHeight();
            let warnings = null;
            if (bodyContentHeight !== undefined) {
                bodyContentHeight -= 2; // Account for padding;
            }
            if (containsRedirection) {
                // Calculate lines needed for Note and Tip
                const safeWidth = Math.max(terminalWidth, 1);
                const noteLength = REDIRECTION_WARNING_NOTE_LABEL.length +
                    REDIRECTION_WARNING_NOTE_TEXT.length;
                const tipLength = REDIRECTION_WARNING_TIP_LABEL.length +
                    REDIRECTION_WARNING_TIP_TEXT.length;
                const noteLines = Math.ceil(noteLength / safeWidth);
                const tipLines = Math.ceil(tipLength / safeWidth);
                const spacerLines = 1;
                const warningHeight = noteLines + tipLines + spacerLines;
                if (bodyContentHeight !== undefined) {
                    bodyContentHeight = Math.max(bodyContentHeight - warningHeight, MINIMUM_MAX_HEIGHT);
                }
                warnings = (_jsxs(_Fragment, { children: [_jsx(Box, { height: 1 }), _jsx(Box, { children: _jsxs(Text, { color: theme.text.primary, children: [_jsx(Text, { bold: true, children: REDIRECTION_WARNING_NOTE_LABEL }), REDIRECTION_WARNING_NOTE_TEXT] }) }), _jsx(Box, { children: _jsxs(Text, { color: theme.border.default, children: [_jsx(Text, { bold: true, children: REDIRECTION_WARNING_TIP_LABEL }), REDIRECTION_WARNING_TIP_TEXT] }) })] }));
            }
            bodyContent = (_jsxs(Box, { flexDirection: "column", children: [_jsx(MaxSizedBox, { maxHeight: bodyContentHeight, maxWidth: Math.max(terminalWidth, 1), children: _jsx(Box, { flexDirection: "column", children: commandsToDisplay.map((cmd, idx) => (_jsx(Text, { color: theme.text.link, children: sanitizeForDisplay(cmd) }, idx))) }) }), warnings] }));
        }
        else if (confirmationDetails.type === 'info') {
            const infoProps = confirmationDetails;
            const displayUrls = infoProps.urls &&
                !(infoProps.urls.length === 1 && infoProps.urls[0] === infoProps.prompt);
            bodyContent = (_jsxs(Box, { flexDirection: "column", children: [_jsx(Text, { color: theme.text.link, children: _jsx(RenderInline, { text: infoProps.prompt, defaultColor: theme.text.link }) }), displayUrls && infoProps.urls && infoProps.urls.length > 0 && (_jsxs(Box, { flexDirection: "column", marginTop: 1, children: [_jsx(Text, { color: theme.text.primary, children: "URLs to fetch:" }), infoProps.urls.map((url) => (_jsxs(Text, { children: [' ', "- ", _jsx(RenderInline, { text: url })] }, url)))] }))] }));
        }
        else if (confirmationDetails.type === 'mcp') {
            // mcp tool confirmation
            const mcpProps = confirmationDetails;
            bodyContent = (_jsxs(Box, { flexDirection: "column", children: [_jsxs(Text, { color: theme.text.link, children: ["MCP Server: ", mcpProps.serverName] }), _jsxs(Text, { color: theme.text.link, children: ["Tool: ", mcpProps.toolName] })] }));
        }
        return { question, bodyContent, options };
    }, [
        confirmationDetails,
        getOptions,
        availableBodyContentHeight,
        terminalWidth,
        handleConfirm,
    ]);
    if (confirmationDetails.type === 'edit') {
        if (confirmationDetails.isModifying) {
            return (_jsxs(Box, { width: terminalWidth, borderStyle: "round", borderColor: theme.border.default, justifyContent: "space-around", paddingTop: 1, paddingBottom: 1, overflow: "hidden", children: [_jsx(Text, { color: theme.text.primary, children: "Modify in progress: " }), _jsx(Text, { color: theme.status.success, children: "Save and close external editor to continue" })] }));
        }
    }
    return (_jsx(Box, { flexDirection: "column", paddingTop: 0, paddingBottom: handlesOwnUI ? 0 : 1, children: handlesOwnUI ? (bodyContent) : (_jsxs(_Fragment, { children: [_jsx(Box, { flexGrow: 1, flexShrink: 1, overflow: "hidden", children: _jsx(MaxSizedBox, { maxHeight: availableBodyContentHeight(), maxWidth: terminalWidth, overflowDirection: "top", children: bodyContent }) }), _jsx(Box, { marginBottom: 1, flexShrink: 0, children: _jsx(Text, { color: theme.text.primary, children: question }) }), _jsx(Box, { flexShrink: 0, children: _jsx(RadioButtonSelect, { items: options, onSelect: handleSelect, isFocused: isFocused }) })] })) }));
};
//# sourceMappingURL=ToolConfirmationMessage.js.map