import { jsx as _jsx } from "react/jsx-runtime";
import { useState, useEffect, useMemo } from 'react';
import { Text, useIsScreenReaderEnabled } from 'ink';
import { CliSpinner } from './CliSpinner.js';
import { useStreamingContext } from '../contexts/StreamingContext.js';
import { StreamingState } from '../types.js';
import { SCREEN_READER_LOADING, SCREEN_READER_RESPONDING, } from '../textConstants.js';
import { theme } from '../semantic-colors.js';
import { Colors } from '../colors.js';
import tinygradient from 'tinygradient';
const COLOR_CYCLE_DURATION_MS = 4000;
export const GeminiRespondingSpinner = ({ nonRespondingDisplay, spinnerType = 'dots' }) => {
    const streamingState = useStreamingContext();
    const isScreenReaderEnabled = useIsScreenReaderEnabled();
    if (streamingState === StreamingState.Responding) {
        return (_jsx(GeminiSpinner, { spinnerType: spinnerType, altText: SCREEN_READER_RESPONDING }));
    }
    if (nonRespondingDisplay) {
        return isScreenReaderEnabled ? (_jsx(Text, { children: SCREEN_READER_LOADING })) : (_jsx(Text, { color: theme.text.primary, children: nonRespondingDisplay }));
    }
    return null;
};
export const GeminiSpinner = ({ spinnerType = 'dots', altText, }) => {
    const isScreenReaderEnabled = useIsScreenReaderEnabled();
    const [time, setTime] = useState(0);
    const googleGradient = useMemo(() => {
        const brandColors = [
            Colors.AccentPurple,
            Colors.AccentBlue,
            Colors.AccentCyan,
            Colors.AccentGreen,
            Colors.AccentYellow,
            Colors.AccentRed,
        ];
        return tinygradient([...brandColors, brandColors[0]]);
    }, []);
    useEffect(() => {
        if (isScreenReaderEnabled) {
            return;
        }
        const interval = setInterval(() => {
            setTime((prevTime) => prevTime + 30);
        }, 30); // ~33fps for smooth color transitions
        return () => clearInterval(interval);
    }, [isScreenReaderEnabled]);
    const progress = (time % COLOR_CYCLE_DURATION_MS) / COLOR_CYCLE_DURATION_MS;
    const currentColor = googleGradient.rgbAt(progress).toHexString();
    return isScreenReaderEnabled ? (_jsx(Text, { children: altText })) : (_jsx(Text, { color: currentColor, children: _jsx(CliSpinner, { type: spinnerType }) }));
};
//# sourceMappingURL=GeminiRespondingSpinner.js.map