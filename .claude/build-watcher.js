#!/usr/bin/env node

/**
 * Build Watcher
 * Runs builds periodically and updates status file
 */

const { spawn } = require('child_process');
const fs = require('fs');
const path = require('path');

const STATUS_FILE = path.join(__dirname, 'status', 'build.json');
const CHECK_INTERVAL = 45000; // 45 seconds
const DEBOUNCE_DELAY = 5000; // Wait 5s after file change before building

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
  return `${colors.dim}[build-watcher]${colors.reset}`;
}

function updateStatus(state, message = null) {
  const status = {
    category: 'Build',
    state,
    message,
    lastUpdated: new Date().toISOString()
  };

  const dir = path.dirname(STATUS_FILE);
  if (!fs.existsSync(dir)) {
    fs.mkdirSync(dir, { recursive: true });
  }

  fs.writeFileSync(STATUS_FILE, JSON.stringify(status, null, 2));
}

function parseFirstError(output) {
  const lines = output.split('\n');

  for (const line of lines) {
    // TypeScript errors
    if (line.includes('error TS')) {
      return line.trim().substring(0, 100);
    }
    // Generic errors
    if (line.toLowerCase().includes('error:')) {
      return line.trim().substring(0, 100);
    }
    // Failed messages
    if (line.includes('FAILED') || line.includes('Failed')) {
      return line.trim().substring(0, 100);
    }
  }

  return null;
}

function runBuild() {
  if (isRunning) {
    console.log(`${prefix()} ${colors.dim}Build already running, skipping...${colors.reset}`);
    return;
  }

  isRunning = true;
  lastRunTime = Date.now();

  console.log(`${prefix()} ${colors.dim}Running build...${colors.reset}`);

  const child = spawn('npm', ['run', 'build'], {
    shell: true,
    cwd: path.join(__dirname, '..')
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

    if (code === 0) {
      updateStatus('success');
      console.log(`${prefix()} ${colors.green}✓ Build succeeded${colors.reset}`);
    } else {
      const firstError = parseFirstError(output);
      updateStatus('fail', firstError || 'Build failed');
      console.log(`${prefix()} ${colors.red}✗ Build failed${colors.reset}`);
    }
  });

  child.on('error', (err) => {
    isRunning = false;
    updateStatus('error', err.message);
    console.error(`${prefix()} ${colors.red}Error running build: ${err.message}${colors.reset}`);
  });
}

function watchFiles() {
  const srcDir = path.join(__dirname, '..', 'src');

  const watcher = (eventType, filename) => {
    if (!filename || !filename.endsWith('.ts')) return;

    console.log(`${prefix()} ${colors.dim}File changed: ${filename}${colors.reset}`);

    // Debounce: wait for changes to settle
    if (debounceTimer) {
      clearTimeout(debounceTimer);
    }

    debounceTimer = setTimeout(() => {
      runBuild();
    }, DEBOUNCE_DELAY);
  };

  if (fs.existsSync(srcDir)) {
    fs.watch(srcDir, { recursive: true }, watcher);
    console.log(`${prefix()} ${colors.dim}Watching src/${colors.reset}`);
  }
}

function main() {
  console.log(`${prefix()} ${colors.dim}Starting build watcher...${colors.reset}`);

  // Run build immediately on startup
  runBuild();

  // Watch for file changes
  watchFiles();

  // Also run periodically (safety net)
  setInterval(() => {
    if (Date.now() - lastRunTime > CHECK_INTERVAL) {
      runBuild();
    }
  }, CHECK_INTERVAL);
}

main();
