/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */
import { lightTheme, darkTheme } from './theme.js';
export const lightSemanticColors = {
    text: {
        primary: lightTheme.Foreground,
        secondary: lightTheme.Gray,
        link: lightTheme.AccentBlue,
        accent: lightTheme.AccentPurple,
        response: lightTheme.Foreground,
    },
    background: {
        primary: lightTheme.Background,
        diff: {
            added: lightTheme.DiffAdded,
            removed: lightTheme.DiffRemoved,
        },
    },
    border: {
        default: lightTheme.Gray,
        focused: lightTheme.AccentBlue,
    },
    ui: {
        comment: lightTheme.Comment,
        symbol: lightTheme.Gray,
        dark: lightTheme.DarkGray,
        gradient: lightTheme.GradientColors,
    },
    status: {
        error: lightTheme.AccentRed,
        success: lightTheme.AccentGreen,
        warning: lightTheme.AccentYellow,
    },
};
export const darkSemanticColors = {
    text: {
        primary: darkTheme.Foreground,
        secondary: darkTheme.Gray,
        link: darkTheme.AccentBlue,
        accent: darkTheme.AccentPurple,
        response: darkTheme.Foreground,
    },
    background: {
        primary: darkTheme.Background,
        diff: {
            added: darkTheme.DiffAdded,
            removed: darkTheme.DiffRemoved,
        },
    },
    border: {
        default: darkTheme.Gray,
        focused: darkTheme.AccentBlue,
    },
    ui: {
        comment: darkTheme.Comment,
        symbol: darkTheme.Gray,
        dark: darkTheme.DarkGray,
        gradient: darkTheme.GradientColors,
    },
    status: {
        error: darkTheme.AccentRed,
        success: darkTheme.AccentGreen,
        warning: darkTheme.AccentYellow,
    },
};
//# sourceMappingURL=semantic-tokens.js.map