/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */
import { type Config } from '@google/gemini-cli-core';
import type { Settings } from '../config/settings.js';
export declare const DEFAULT_MIN_RETENTION: string;
/**
 * Result of session cleanup operation
 */
export interface CleanupResult {
    disabled: boolean;
    scanned: number;
    deleted: number;
    skipped: number;
    failed: number;
}
/**
 * Main entry point for session cleanup during CLI startup
 */
export declare function cleanupExpiredSessions(config: Config, settings: Settings): Promise<CleanupResult>;
/**
 * Result of tool output cleanup operation
 */
export interface ToolOutputCleanupResult {
    disabled: boolean;
    scanned: number;
    deleted: number;
    failed: number;
}
/**
 * Cleans up tool output files based on age and count limits.
 * Uses the same retention settings as session cleanup.
 */
export declare function cleanupToolOutputFiles(settings: Settings, debugMode?: boolean, projectTempDir?: string): Promise<ToolOutputCleanupResult>;
