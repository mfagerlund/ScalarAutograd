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
let lastReportedFile = null;

const colors = {
  reset: '\x1b[0m',
  dim: '\x1b[90m',
  green: '\x1b[32m',
  red: '\x1b[31m',
};

function prefix() {
  return `${colors.dim}[demo-typecheck-watcher]${colors.reset}`;
}

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
    console.log(`${prefix()} ${colors.dim}Type-check already running, skipping...${colors.reset}`);
    return;
  }

  isRunning = true;
  lastRunTime = Date.now();

  console.log(`${prefix()} ${colors.dim}Running demo type-checks...${colors.reset}`);

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
    console.log(`${prefix()} ${colors.green}✓ All demos type-check${colors.reset}`);
  } else {
    const failedNames = failedDemos.map(r => path.basename(r.demo)).join(', ');
    updateStatus('fail', firstError, {
      totalErrors,
      failedDemos: failedDemos.length,
      demosChecked: results.length
    });
    console.log(`${prefix()} ${colors.red}✗ ${totalErrors} type error${totalErrors === 1 ? '' : 's'} in: ${failedNames}${colors.reset}`);
  }
}

function watchFiles() {
  for (const demoPath of DEMOS) {
    const srcDir = path.join(__dirname, '..', demoPath, 'src');

    if (!fs.existsSync(srcDir)) continue;

    fs.watch(srcDir, { recursive: true }, (eventType, filename) => {
      if (!filename || (!filename.endsWith('.ts') && !filename.endsWith('.tsx'))) return;

      const fileKey = `${demoPath}/${filename}`;
      if (fileKey !== lastReportedFile) {
        console.log(`${prefix()} ${colors.dim}File changed: ${fileKey}${colors.reset}`);
        lastReportedFile = fileKey;
      }

      // Debounce: wait for changes to settle
      if (debounceTimer) {
        clearTimeout(debounceTimer);
      }

      debounceTimer = setTimeout(() => {
        runTypeCheck();
      }, DEBOUNCE_DELAY);
    });

    console.log(`${prefix()} ${colors.dim}Watching ${demoPath}/src/${colors.reset}`);
  }
}

function main() {
  console.log(`${prefix()} ${colors.dim}Starting demo type-check watcher...${colors.reset}`);

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
