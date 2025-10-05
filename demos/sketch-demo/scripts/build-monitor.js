/**
 * TypeScript Build Monitor with Toast Notifications
 * Watches for file changes and notifies on build success/failure
 */

import notifier from 'node-notifier';
import { exec } from 'child_process';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { dirname } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

let isChecking = false;
let lastStatus = null; // Track last status to avoid duplicate notifications

function checkTypeScript() {
  if (isChecking) return;

  isChecking = true;

  exec('tsc --noEmit --project tsconfig.app.json', { cwd: path.join(__dirname, '..') }, (error, stdout, stderr) => {
    isChecking = false;

    const hasErrors = error !== null;

    // Only notify if status changed
    if (hasErrors && lastStatus !== 'error') {
      const errorLines = (stdout + stderr).split('\n').filter(line => line.includes('error TS'));
      const errorCount = errorLines.length;

      notifier.notify({
        title: '❌ TypeScript Build Failed',
        message: `${errorCount} error${errorCount !== 1 ? 's' : ''} found`,
        sound: 'Basso', // System sound
        timeout: 5,
        appID: 'Sketch Demo'
      });

      console.log('\x1b[31m%s\x1b[0m', `❌ Build failed with ${errorCount} errors`);
      lastStatus = 'error';
    } else if (!hasErrors && lastStatus !== 'success') {
      notifier.notify({
        title: '✅ TypeScript Build OK',
        message: 'All types are valid',
        sound: 'Glass', // System sound
        timeout: 3,
        appID: 'Sketch Demo'
      });

      console.log('\x1b[32m%s\x1b[0m', '✅ Build successful');
      lastStatus = 'success';
    }
  });
}

// Initial check
console.log('🔍 Starting TypeScript build monitor...');
checkTypeScript();

// Watch for changes using fs.watch (more reliable on Windows)
const watchPath = path.join(__dirname, '..', 'src');
console.log(`👀 Watching: ${watchPath}`);

let debounceTimer = null;

function handleFileChange(eventType, filename) {
  if (filename && (filename.endsWith('.ts') || filename.endsWith('.tsx'))) {
    console.log(`📝 File ${eventType}: ${filename}`);

    // Debounce to avoid multiple checks for same change
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(() => {
      checkTypeScript();
    }, 500);
  }
}

fs.watch(watchPath, { recursive: true }, handleFileChange);
console.log('👀 Watching src/ for TypeScript changes...\n');
