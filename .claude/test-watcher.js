#!/usr/bin/env node

/**
 * Test Watcher
 * Runs tests periodically and updates status file
 */

const { spawn } = require('child_process');
const fs = require('fs');
const path = require('path');

const STATUS_FILE = path.join(__dirname, 'status', 'tests.json');
const CHECK_INTERVAL = 30000; // 30 seconds
const DEBOUNCE_DELAY = 5000; // Wait 5s after file change before running

let lastRunTime = 0;
let debounceTimer = null;
let isRunning = false;

const colors = {
  reset: '\x1b[0m',
  dim: '\x1b[90m',
  green: '\x1b[32m',
  red: '\x1b[31m',
};

function prefix() {
  return `${colors.dim}[test-watcher]${colors.reset}`;
}

function updateStatus(state, metrics, message = null) {
  const status = {
    category: 'Tests',
    state,
    metrics,
    message,
    lastUpdated: new Date().toISOString()
  };

  const dir = path.dirname(STATUS_FILE);
  if (!fs.existsSync(dir)) {
    fs.mkdirSync(dir, { recursive: true });
  }

  fs.writeFileSync(STATUS_FILE, JSON.stringify(status, null, 2));
}

function parseTestOutput(output) {
  const lines = output.split('\n');
  let passing = 0;
  let failing = 0;
  let total = 0;
  let firstError = null;

  // Parse Vitest v3 output
  // Look for summary at end like:
  // " Test Files  1 failed | 5 passed (6)"
  // " Tests  2 failed | 143 passed (145)"

  for (const line of lines) {
    // Strip ANSI codes for parsing
    const clean = line.replace(/\x1B\[[0-9;]*[a-zA-Z]/g, '');

    // Match "Test Files X failed | Y passed (Z)"
    const filesMatch = clean.match(/Test Files\s+(\d+)\s+failed\s*\|\s*(\d+)\s+passed\s*\((\d+)\)/);
    if (filesMatch) {
      // Just note there are failures, we'll get test count below
      continue;
    }

    // Match "Tests X failed | Y passed (Z)"
    const testFailMatch = clean.match(/Tests\s+(\d+)\s+failed\s*\|\s*(\d+)\s+passed\s*\((\d+)\)/);
    if (testFailMatch) {
      failing = parseInt(testFailMatch[1]);
      passing = parseInt(testFailMatch[2]);
      total = parseInt(testFailMatch[3]);
      continue;
    }

    // Match "Tests X passed (Y)" - all passing
    const testPassMatch = clean.match(/Tests\s+(\d+)\s+passed\s*\((\d+)\)/);
    if (testPassMatch) {
      passing = parseInt(testPassMatch[1]);
      total = parseInt(testPassMatch[2]);
      failing = 0;
      continue;
    }

    // Capture first error/failure
    if (!firstError && (clean.includes('FAIL') || clean.includes('✗')) && clean.trim().length > 0) {
      firstError = clean.trim().substring(0, 100);
    }
    if (!firstError && clean.includes('Error:')) {
      firstError = clean.trim().substring(0, 100);
    }
  }

  return {
    passing,
    failing,
    total,
    firstError
  };
}

function runTests() {
  if (isRunning) {
    console.log(`${prefix()} ${colors.dim}Tests already running, skipping...${colors.reset}`);
    return;
  }

  isRunning = true;
  lastRunTime = Date.now();

  console.log(`${prefix()} ${colors.dim}Running tests...${colors.reset}`);

  const child = spawn('npm', ['test', '--', '--run', '--reporter=default'], {
    shell: true,
    cwd: path.join(__dirname, '..'),
    timeout: 120000 // 2 minute timeout
  });

  let stdout = '';
  let stderr = '';

  child.stdout.on('data', (data) => {
    stdout += data.toString();
  });

  child.stderr.on('data', (data) => {
    stderr += data.toString();
  });

  child.on('close', (code) => {
    isRunning = false;
    const output = stdout + stderr;
    const parsed = parseTestOutput(output);

    if (code === 0 && parsed.failing === 0) {
      updateStatus('passing', {
        passing: parsed.passing || parsed.total,
        failing: 0,
        total: parsed.total
      });
      console.log(`${prefix()} ${colors.green}✓ All tests passing (${parsed.total})${colors.reset}`);
    } else {
      updateStatus('failing', {
        passing: parsed.passing,
        failing: parsed.failing,
        total: parsed.total
      }, parsed.firstError);
      console.log(`${prefix()} ${colors.red}✗ Tests failing: ${parsed.failing}/${parsed.total}${colors.reset}`);
    }
  });

  child.on('error', (err) => {
    isRunning = false;
    updateStatus('error', { passing: 0, failing: 0, total: 0 }, err.message);
    console.error(`${prefix()} ${colors.red}Error running tests: ${err.message}${colors.reset}`);
  });
}

function watchFiles() {
  const srcDir = path.join(__dirname, '..', 'src');
  const testDir = path.join(__dirname, '..', 'test');

  const watcher = (eventType, filename) => {
    if (!filename || !filename.endsWith('.ts')) return;

    console.log(`${prefix()} ${colors.dim}File changed: ${filename}${colors.reset}`);

    // Debounce: wait for changes to settle
    if (debounceTimer) {
      clearTimeout(debounceTimer);
    }

    debounceTimer = setTimeout(() => {
      runTests();
    }, DEBOUNCE_DELAY);
  };

  if (fs.existsSync(srcDir)) {
    fs.watch(srcDir, { recursive: true }, watcher);
    console.log(`${prefix()} ${colors.dim}Watching src/${colors.reset}`);
  }

  if (fs.existsSync(testDir)) {
    fs.watch(testDir, { recursive: true }, watcher);
    console.log(`${prefix()} ${colors.dim}Watching test/${colors.reset}`);
  }
}

function main() {
  console.log(`${prefix()} ${colors.dim}Starting test watcher...${colors.reset}`);

  // Run tests immediately on startup
  runTests();

  // Watch for file changes
  watchFiles();

  // Also run periodically (safety net)
  setInterval(() => {
    if (Date.now() - lastRunTime > CHECK_INTERVAL) {
      runTests();
    }
  }, CHECK_INTERVAL);
}

main();
