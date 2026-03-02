import { jsx as _jsx, jsxs as _jsxs, Fragment as _Fragment } from "react/jsx-runtime";
import { Box } from 'ink';
import { StickyHeader } from '../StickyHeader.js';
import { ToolResultDisplay } from './ToolResultDisplay.js';
import { ToolStatusIndicator, ToolInfo, TrailingIndicator, STATUS_INDICATOR_WIDTH, isThisShellFocusable as checkIsShellFocusable, isThisShellFocused as checkIsShellFocused, useFocusHint, FocusHint, } from './ToolShared.js';
import {} from '@google/gemini-cli-core';
import { ShellInputPrompt } from '../ShellInputPrompt.js';
export const ToolMessage = ({ name, description, resultDisplay, status, availableTerminalHeight, terminalWidth, emphasis = 'medium', renderOutputAsMarkdown = true, isFirst, borderColor, borderDimColor, activeShellPtyId, embeddedShellFocused, ptyId, config, }) => {
    const isThisShellFocused = checkIsShellFocused(name, status, ptyId, activeShellPtyId, embeddedShellFocused);
    const isThisShellFocusable = checkIsShellFocusable(name, status, config);
    const { shouldShowFocusHint } = useFocusHint(isThisShellFocusable, isThisShellFocused, resultDisplay);
    return (
    // It is crucial we don't replace this <> with a Box because otherwise the
    // sticky header inside it would be sticky to that box rather than to the
    // parent component of this ToolMessage.
    _jsxs(_Fragment, { children: [_jsxs(StickyHeader, { width: terminalWidth, isFirst: isFirst, borderColor: borderColor, borderDimColor: borderDimColor, children: [_jsx(ToolStatusIndicator, { status: status, name: name }), _jsx(ToolInfo, { name: name, status: status, description: description, emphasis: emphasis }), _jsx(FocusHint, { shouldShowFocusHint: shouldShowFocusHint, isThisShellFocused: isThisShellFocused }), emphasis === 'high' && _jsx(TrailingIndicator, {})] }), _jsxs(Box, { width: terminalWidth, borderStyle: "round", borderColor: borderColor, borderDimColor: borderDimColor, borderTop: false, borderBottom: false, borderLeft: true, borderRight: true, paddingX: 1, flexDirection: "column", children: [_jsx(ToolResultDisplay, { resultDisplay: resultDisplay, availableTerminalHeight: availableTerminalHeight, terminalWidth: terminalWidth, renderOutputAsMarkdown: renderOutputAsMarkdown, hasFocus: isThisShellFocused }), isThisShellFocused && config && (_jsx(Box, { paddingLeft: STATUS_INDICATOR_WIDTH, marginTop: 1, children: _jsx(ShellInputPrompt, { activeShellPtyId: activeShellPtyId ?? null, focus: embeddedShellFocused }) }))] })] }));
};
//# sourceMappingURL=ToolMessage.js.map