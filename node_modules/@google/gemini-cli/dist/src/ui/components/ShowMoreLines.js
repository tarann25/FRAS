import { jsx as _jsx } from "react/jsx-runtime";
/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */
import { Box, Text } from 'ink';
import { useOverflowState } from '../contexts/OverflowContext.js';
import { useStreamingContext } from '../contexts/StreamingContext.js';
import { StreamingState } from '../types.js';
import { theme } from '../semantic-colors.js';
export const ShowMoreLines = ({ constrainHeight }) => {
    const overflowState = useOverflowState();
    const streamingState = useStreamingContext();
    if (overflowState === undefined ||
        overflowState.overflowingIds.size === 0 ||
        !constrainHeight ||
        !(streamingState === StreamingState.Idle ||
            streamingState === StreamingState.WaitingForConfirmation)) {
        return null;
    }
    return (_jsx(Box, { paddingX: 1, children: _jsx(Text, { color: theme.text.secondary, wrap: "truncate", children: "Press ctrl-o to show more lines" }) }));
};
//# sourceMappingURL=ShowMoreLines.js.map