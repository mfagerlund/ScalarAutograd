# Claude Status Monitoring System

This system monitors build/test status and notifies Claude Code during development via discreet status messages.

## Architecture

**Watchers** (run these manually in separate terminals):
- `.claude/test-watcher.js` - Monitors test status
- `.claude/build-watcher.js` - Monitors build status
- `.claude/vite-watcher.js` - Monitors Vite dev server health

**Hook** (runs automatically on every tool use):
- `.claude/hooks/status-reporter.js` - Computes deltas and reports to Claude

**Status Files**:
- `.claude/status/*.json` - Current state written by watchers
- `.claude/status/.snapshot/*.json` - Previous state (for delta computation)

## Quick Start

```bash
# Start all watchers (easiest - runs in background)
npm run watch:all

# OR start individually (in separate terminals)
npm run watch:test
npm run watch:build
npm run watch:vite
```

Claude will now receive status updates like:
```
Tests: +2-3 (9 failing)
Build: ✓→✗ (Cannot find module 'foo')
Vite: ✗→✓
```

## Status Message Format

```
[category]: [transition] [details]
```

**Examples:**
- `Tests: +2-3 (9 failing)` - Fixed 2 tests, 3 new failures, 9 total failing
- `Build: ✓→✗ (Cannot find...)` - Build broke with error message
- `Vite: ✗→✓` - Vite server recovered
- `Deploy: staging updated (v2.1.3)` - Custom deployment message

**Symbols:**
- `✓→✗` - State transition (success to fail)
- `+X-Y` - Delta changes (added/fixed vs new failures)
- `(...)` - Critical details (error, count, version)

## Status File Schema

Each watcher writes JSON files to `.claude/status/`:

```json
{
  "category": "Tests",
  "state": "failing",
  "metrics": {
    "passing": 10,
    "failing": 9,
    "total": 19
  },
  "message": "Cannot find module 'foo'",
  "lastUpdated": "2025-10-04T10:30:00.000Z"
}
```

**Required fields:**
- `category` - Display name (e.g., "Tests", "Build", "Deploy")
- `state` - Current state (e.g., "success", "failing", "error")
- `lastUpdated` - ISO timestamp

**Optional fields:**
- `metrics` - Object with numeric values for delta computation
- `message` - Error details or info message

## Extending the System

### Add a Custom Watcher

Create a new watcher that writes to `.claude/status/your-category.json`:

```javascript
const fs = require('fs');
const path = require('path');

function updateStatus(state, message) {
  const status = {
    category: 'YourCategory',
    state,
    message,
    lastUpdated: new Date().toISOString()
  };

  fs.writeFileSync(
    path.join(__dirname, 'status', 'your-category.json'),
    JSON.stringify(status, null, 2)
  );
}

// Your monitoring logic
setInterval(() => {
  // Check something...
  if (allGood) {
    updateStatus('success');
  } else {
    updateStatus('fail', 'Something went wrong');
  }
}, 30000);
```

### Custom Metrics

For delta tracking, use the `metrics` field:

```json
{
  "category": "Database",
  "state": "warning",
  "metrics": {
    "diskPercent": 90,
    "connections": 45
  },
  "message": "Disk usage high",
  "lastUpdated": "2025-10-04T10:30:00.000Z"
}
```

The reporter will automatically detect changes like `87%→90% disk`.

### External Integrations

Any process can write status files. Examples:

**Deployment script:**
```bash
echo '{
  "category": "Deploy",
  "state": "success",
  "message": "Production v2.1.3 live",
  "lastUpdated": "'$(date -Iseconds)'"
}' > .claude/status/deploy.json
```

**CI/CD webhook:**
```javascript
// In your webhook handler
fs.writeFileSync('.claude/status/ci.json', JSON.stringify({
  category: 'CI',
  state: build.status === 'success' ? 'success' : 'fail',
  message: build.status === 'success' ? null : build.error,
  lastUpdated: new Date().toISOString()
}));
```

## How It Works

1. **Watchers run continuously**, monitoring their respective systems
2. **Status files updated** when state changes
3. **On every tool use**, the hook:
   - Reads all current status files
   - Compares with snapshots
   - Computes deltas
   - Reports changes to Claude
   - Updates snapshots

4. **Claude receives** concise status updates without being asked

## Design Principles

- **Discreet** - Messages are 1 line, no noise if nothing changed
- **Actionable** - Focus on deltas that matter
- **Extensible** - Drop in new status files, hook picks them up
- **Non-blocking** - Watchers run independently, hook is fast
- **Universal** - Any process can write status files

## Troubleshooting

**No status messages appearing:**
- Check watchers are running: `ps aux | grep watcher`
- Check status files exist: `ls .claude/status/`
- Check hook registered: `cat .claude/settings.json`

**Watchers not updating:**
- Check file permissions
- Look for errors in watcher console output
- Verify commands work: `npm test`, `npm run build`

**False positives:**
- Adjust `CHECK_INTERVAL` in watchers
- Increase `DEBOUNCE_DELAY` for file watching
- Check `consecutiveFailures` logic in vite-watcher

## Files

```
.claude/
├── README.md                      # This file
├── settings.json                  # Hook registration
├── test-watcher.js               # Run this
├── build-watcher.js              # Run this
├── vite-watcher.js               # Run this
├── hooks/
│   └── status-reporter.js        # Auto-runs on tool use
└── status/
    ├── tests.json                # Current test status
    ├── build.json                # Current build status
    ├── vite.json                 # Current vite status
    └── .snapshot/                # Previous states
        ├── tests.json
        ├── build.json
        └── vite.json
```
