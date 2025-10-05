---
description: Start all status watchers as background processes
allowed-tools: Bash(node .claude/*-watcher.js:*)
---

Starting all four watchers as background processes...

!node .claude/test-watcher.js
!node .claude/build-watcher.js
!node .claude/demo-typecheck-watcher.js
!node .claude/vite-watcher.js

All watchers started:
- Tests (npm test)
- Build (tsc)
- Demo TypeScript (tsc in demos)
- Vite dev server

You'll receive status updates automatically on every interaction.
