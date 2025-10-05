#!/usr/bin/env node

/**
 * Demo TypeCheck Watcher
 * Monitors TypeScript type-checking in demo projects (separate from main build)
 */

const { spawn } = require('child_process');
const fs = require('fs');
const path = require('path');

const STATUS_FILE = path.join(__dirname, 'status', 'demo-typecheck.json');
const CHECK_INTERVAL = 45000; // 45 seconds
const DEBOUNCE_DELAY = 5000; // Wait 5s after file change before checking

let lastRunTime = 0;
let debounceTimer = null;
let isRunning = false;

// Demos to check
const DEMOS = [
  'demos/developable-sphere',
  'demos/autograd-playground',
  'demos/sketch-demo'
];

function updateStatus(state, message = null, metrics = null) {
  const status = {
    category: 'Demo TS',
    state,
    message,
    metrics,
    lastUpdated: new Date().toISOString()
  };

  const dir = path.dirname(STATUS_FILE);
  if (!fs.existsSync(dir)) {
    fs.mkdirSync(dir, { recursive: true });
  }

  fs.writeFileSync(STATUS_FILE, JSON.stringify(status, null, 2));
}

function parseTypeErrors(output) {
  const lines = output.split('\n');
  const errorPattern = /error TS\d+:/i;
  let errorCount = 0;
  let firstError = null;

  for (const line of lines) {
    if (errorPattern.test(line)) {
      errorCount++;
      if (!firstError) {
        firstError = line.trim().substring(0, 80);
      }
    }
  }

  // Also look for summary line: "Found X errors"
  const summaryMatch = output.match(/Found (\d+) errors?/);
  if (summaryMatch) {
    errorCount = Math.max(errorCount, parseInt(summaryMatch[1]));
  }

  return { errorCount, firstError };
}

async function checkDemo(demoPath) {
  return new Promise((resolve) => {
    const absolutePath = path.join(__dirname, '..', demoPath);

    if (!fs.existsSync(absolutePath)) {
      resolve({ demo: demoPath, success: true, errors: 0 });
      return;
    }

    const child = spawn('npx', ['tsc', '--noEmit'], {
      shell: true,
      cwd: absolutePath
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
      const output = stdout + stderr;
      const { errorCount, firstError } = parseTypeErrors(output);

      resolve({
        demo: demoPath,
        success: code === 0 && errorCount === 0,
        errors: errorCount,
        firstError
      });
    });

    child.on('error', (err) => {
      resolve({
        demo: demoPath,
        success: false,
        errors: 1,
        firstError: err.message
      });
    });
  });
}

async function runTypeCheck() {
  if (isRunning) {
    console.log('[demo-typecheck-watcher] Type-check already running, skipping...');
    return;
  }

  isRunning = true;
  lastRunTime = Date.now();

  console.log('[demo-typecheck-watcher] Running demo type-checks...');

  const results = await Promise.all(DEMOS.map(checkDemo));

  isRunning = false;

  const totalErrors = results.reduce((sum, r) => sum + r.errors, 0);
  const failedDemos = results.filter(r => !r.success);
  const firstError = failedDemos[0]?.firstError;

  if (totalErrors === 0) {
    updateStatus('success', null, {
      totalErrors: 0,
      demosChecked: results.length
    });
    console.log('[demo-typecheck-watcher] ✓ All demos type-check');
  } else {
    const failedNames = failedDemos.map(r => path.basename(r.demo)).join(', ');
    updateStatus('fail', firstError, {
      totalErrors,
      failedDemos: failedDemos.length,
      demosChecked: results.length
    });
    console.log(`[demo-typecheck-watcher] ✗ ${totalErrors} type errors in: ${failedNames}`);
  }
}

function watchFiles() {
  for (const demoPath of DEMOS) {
    const srcDir = path.join(__dirname, '..', demoPath, 'src');

    if (!fs.existsSync(srcDir)) continue;

    fs.watch(srcDir, { recursive: true }, (eventType, filename) => {
      if (!filename || (!filename.endsWith('.ts') && !filename.endsWith('.tsx'))) return;

      console.log(`[demo-typecheck-watcher] File changed: ${demoPath}/${filename}`);

      // Debounce: wait for changes to settle
      if (debounceTimer) {
        clearTimeout(debounceTimer);
      }

      debounceTimer = setTimeout(() => {
        runTypeCheck();
      }, DEBOUNCE_DELAY);
    });

    console.log(`[demo-typecheck-watcher] Watching ${demoPath}/src/`);
  }
}

function main() {
  console.log('[demo-typecheck-watcher] Starting demo type-check watcher...');

  // Run type-check immediately on startup
  runTypeCheck();

  // Watch for file changes
  watchFiles();

  // Also run periodically (safety net)
  setInterval(() => {
    if (Date.now() - lastRunTime > CHECK_INTERVAL) {
      runTypeCheck();
    }
  }, CHECK_INTERVAL);
}

main();
