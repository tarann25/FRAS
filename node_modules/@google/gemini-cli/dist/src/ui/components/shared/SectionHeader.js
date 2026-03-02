import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { Box, Text } from 'ink';
import { theme } from '../../semantic-colors.js';
export const SectionHeader = ({ title }) => (_jsxs(Box, { width: "100%", flexDirection: "row", overflow: "hidden", children: [_jsx(Text, { color: theme.text.secondary, wrap: "truncate-end", children: `── ${title}` }), _jsx(Box, { flexGrow: 1, flexShrink: 0, minWidth: 2, marginLeft: 1, borderStyle: "single", borderTop: true, borderBottom: false, borderLeft: false, borderRight: false, borderColor: theme.text.secondary })] }));
//# sourceMappingURL=SectionHeader.js.map