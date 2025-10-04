#!/usr/bin/env node

/**
 * Status Query Tool
 * Reports current status (not deltas) - for explicit queries
 */

const fs = require('fs');
const path = require('path');

const STATUS_DIR = path.join(__dirname, '..', 'status');

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

function formatStatus(status) {
  const category = status.category;
  const parts = [];

  // State symbol
  const stateSymbol = (state) => {
    if (state === 'success' || state === 'passing') return '✓';
    if (state === 'failing' || state === 'fail') return '✗';
    if (state === 'warning') return '⚠';
    if (state === 'error') return '⚠';
    return '?';
  };

  parts.push(stateSymbol(status.state));

  // Metrics
  if (status.metrics) {
    const m = status.metrics;

    // Tests
    if (m.passing !== undefined && m.failing !== undefined) {
      if (m.failing > 0) {
        parts.push(`${m.failing}/${m.total} failing`);
      } else {
        parts.push(`${m.total}/${m.total} passing`);
      }
    }

    // Percentages
    Object.keys(m).forEach(key => {
      if (key.endsWith('Percent') || key.endsWith('%')) {
        parts.push(`${m[key]}%`);
      }
    });
  }

  // Message
  if (status.message) {
    const msg = status.message.length > 60
      ? status.message.substring(0, 57) + '...'
      : status.message;
    parts.push(`(${msg})`);
  }

  // Age
  if (status.lastUpdated) {
    const age = Math.floor((Date.now() - new Date(status.lastUpdated)) / 1000);
    if (age > 120) {
      parts.push(`[${age}s ago]`);
    }
  }

  return `${category}: ${parts.join(' ')}`;
}

function main() {
  const statusFiles = getStatusFiles();

  if (statusFiles.length === 0) {
    console.log(JSON.stringify({
      type: 'context_injection',
      content: 'Status: No watchers running'
    }));
    return;
  }

  const lines = [];

  for (const statusFile of statusFiles) {
    const status = readJsonFile(statusFile);
    if (!status) continue;

    lines.push(formatStatus(status));
  }

  if (lines.length > 0) {
    const output = {
      type: 'context_injection',
      content: lines.join('\n')
    };
    console.log(JSON.stringify(output));
  }
}

main();
