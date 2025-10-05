#!/usr/bin/env node

/**
 * Vite Watcher
 * Monitors Vite dev server to detect when it stops updating
 *
 * This watcher looks for common signs that Vite has stopped working:
 * - Server process not responding
 * - HMR not updating after file changes
 * - Port conflicts
 */

const { spawn } = require('child_process');
const fs = require('fs');
const path = require('path');
const http = require('http');

const STATUS_FILE = path.join(__dirname, 'status', 'vite.json');
const CHECK_INTERVAL = 20000; // Check every 20 seconds
const VITE_PORT = 5173; // Default Vite port

let lastHealthCheck = Date.now();
let consecutiveFailures = 0;

const colors = {
  reset: '\x1b[0m',
  dim: '\x1b[90m',
  green: '\x1b[32m',
  red: '\x1b[31m',
};

function prefix() {
  return `${colors.dim}[vite-watcher]${colors.reset}`;
}

function updateStatus(state, message = null) {
  const status = {
    category: 'Vite',
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

function checkViteHealth() {
  return new Promise((resolve) => {
    const req = http.request(
      {
        hostname: 'localhost',
        port: VITE_PORT,
        path: '/',
        method: 'GET',
        timeout: 3000
      },
      (res) => {
        resolve({ healthy: res.statusCode === 200, error: null });
      }
    );

    req.on('error', (err) => {
      resolve({ healthy: false, error: err.message });
    });

    req.on('timeout', () => {
      req.destroy();
      resolve({ healthy: false, error: 'Timeout' });
    });

    req.end();
  });
}

async function performHealthCheck() {
  const result = await checkViteHealth();
  lastHealthCheck = Date.now();

  if (result.healthy) {
    consecutiveFailures = 0;
    updateStatus('success');
    console.log(`${prefix()} ${colors.green}✓ Vite server healthy${colors.reset}`);
  } else {
    consecutiveFailures++;

    if (consecutiveFailures >= 2) {
      // Only report after 2 consecutive failures to avoid false positives
      if (result.error.includes('ECONNREFUSED')) {
        updateStatus('fail', 'Server not running');
        console.log(`${prefix()} ${colors.red}✗ Vite server not running${colors.reset}`);
      } else if (result.error === 'Timeout') {
        updateStatus('fail', 'Server not responding');
        console.log(`${prefix()} ${colors.red}✗ Vite server not responding${colors.reset}`);
      } else {
        updateStatus('fail', result.error);
        console.log(`${prefix()} ${colors.red}✗ Vite server unhealthy: ${result.error}${colors.reset}`);
      }
    }
  }
}

function watchViteLog() {
  // Try to detect if Vite is running by looking for common demo directories
  const demoDirs = [
    path.join(__dirname, '..', 'demos', 'developable-sphere'),
    path.join(__dirname, '..', 'demos', 'autograd-playground'),
    path.join(__dirname, '..', 'demos', 'sketch-demo')
  ];

  // Watch for changes in demo src directories to detect if HMR should be working
  for (const demoDir of demoDirs) {
    const srcDir = path.join(demoDir, 'src');
    if (fs.existsSync(srcDir)) {
      fs.watch(srcDir, { recursive: true }, (eventType, filename) => {
        if (!filename || !filename.endsWith('.tsx') && !filename.endsWith('.ts') && !filename.endsWith('.css')) {
          return;
        }
        console.log(`${prefix()} ${colors.dim}File changed in demo: ${filename}, checking Vite health...${colors.reset}`);
        // Check health shortly after file changes to detect HMR issues
        setTimeout(() => performHealthCheck(), 2000);
      });
    }
  }
}

function main() {
  console.log(`${prefix()} ${colors.dim}Starting Vite health monitor...${colors.reset}`);
  console.log(`${prefix()} ${colors.dim}Monitoring port: ${VITE_PORT}${colors.reset}`);

  // Initial health check
  performHealthCheck();

  // Watch demo directories
  watchViteLog();

  // Periodic health checks
  setInterval(() => {
    performHealthCheck();
  }, CHECK_INTERVAL);
}

main();
