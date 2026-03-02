/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */
import type { Settings } from '../config/settingsSchema.js';
export declare function getUserStartupWarnings(settings: Settings, workspaceRoot?: string): Promise<string[]>;
