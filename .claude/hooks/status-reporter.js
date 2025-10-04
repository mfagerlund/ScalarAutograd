#!/usr/bin/env node

/**
 * Status Reporter Hook
 * Compares current status files with snapshots and reports deltas to Claude
 */

const fs = require('fs');
const path = require('path');

const STATUS_DIR = path.join(__dirname, '..', 'status');
const SNAPSHOT_DIR = path.join(STATUS_DIR, '.snapshot');

function readJsonFile(filePath) {
  try {
    return JSON.parse(fs.readFileSync(filePath, 'utf8'));
  } catch (e) {
    return null;
  }
}

function getStatusFiles() {
  if (!fs.existsSync(STATUS_DIR)) return [];
  return fs.readdirSync(STATUS_DIR)
    .filter(f => f.endsWith('.json') && !f.startsWith('.'))
    .map(f => path.join(STATUS_DIR, f));
}

function formatDelta(current, snapshot) {
  const category = current.category;
  const parts = [];

  // State transition
  if (snapshot && snapshot.state !== current.state) {
    const stateSymbol = (state) => state === 'success' || state === 'passing' ? '✓' : '✗';
    parts.push(`${stateSymbol(snapshot.state)}→${stateSymbol(current.state)}`);
  }

  // Metrics delta
  if (current.metrics && snapshot?.metrics) {
    const curr = current.metrics;
    const snap = snapshot.metrics;

    // For tests: show +passed -failed
    if (curr.passing !== undefined && curr.failing !== undefined) {
      const passedDelta = curr.passing - (snap.passing || 0);
      const failedDelta = curr.failing - (snap.failing || 0);

      if (passedDelta !== 0 || failedDelta !== 0) {
        let delta = '';
        if (passedDelta > 0) delta += `+${passedDelta}`;
        if (failedDelta > 0) delta += `-${failedDelta}`;
        if (failedDelta < 0) delta += `+${Math.abs(failedDelta)}`; // Fixed tests
        if (passedDelta < 0) delta += `-${Math.abs(passedDelta)}`; // Regressed tests

        if (curr.failing > 0) {
          delta += ` (${curr.failing} failing)`;
        }
        parts.push(delta.trim());
      } else if (!snapshot && curr.failing > 0) {
        parts.push(`(${curr.failing} failing)`);
      }
    }

    // For percentage-based metrics
    Object.keys(curr).forEach(key => {
      if (key.endsWith('Percent') || key.endsWith('%')) {
        const snapVal = snap[key];
        if (snapVal !== undefined && snapVal !== curr[key]) {
          parts.push(`${snapVal}%→${curr[key]}%`);
        }
      }
    });
  }

  // Message (error details, etc.)
  if (current.message && current.state !== 'success') {
    // Truncate long messages
    const msg = current.message.length > 60
      ? current.message.substring(0, 57) + '...'
      : current.message;
    parts.push(`(${msg})`);
  } else if (current.message && !snapshot?.message) {
    // New info message
    parts.push(`(${current.message})`);
  }

  // If nothing changed, return null
  if (parts.length === 0 && snapshot) {
    return null;
  }

  // Special case: first run with no issues
  if (!snapshot && parts.length === 0 && current.state === 'success') {
    return null;
  }

  return `${category}: ${parts.join(' ')}`;
}

function saveSnapshot(statusFile) {
  const fileName = path.basename(statusFile);
  const snapshotPath = path.join(SNAPSHOT_DIR, fileName);

  if (!fs.existsSync(SNAPSHOT_DIR)) {
    fs.mkdirSync(SNAPSHOT_DIR, { recursive: true });
  }

  fs.copyFileSync(statusFile, snapshotPath);
}

function main() {
  const statusFiles = getStatusFiles();
  const deltas = [];

  for (const statusFile of statusFiles) {
    const current = readJsonFile(statusFile);
    if (!current) continue;

    const fileName = path.basename(statusFile);
    const snapshotPath = path.join(SNAPSHOT_DIR, fileName);
    const snapshot = readJsonFile(snapshotPath);

    const delta = formatDelta(current, snapshot);
    if (delta) {
      deltas.push(delta);
    }

    // Save current as new snapshot
    saveSnapshot(statusFile);
  }

  // Output deltas as JSON for Claude
  if (deltas.length > 0) {
    const output = {
      type: 'context_injection',
      content: deltas.join('\n')
    };
    console.log(JSON.stringify(output));
  }
}

main();
