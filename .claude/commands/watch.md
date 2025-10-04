---
description: Start all status watchers (test, build, demo-ts, vite)
allowed-tools: Bash(node .claude/*-watcher.js:*)
---

Start all four watchers as background processes:

!node .claude/test-watcher.js &
!node .claude/build-watcher.js &
!node .claude/demo-typecheck-watcher.js &
!node .claude/vite-watcher.js &

This monitors:
- Tests (npm test)
- Build (tsc)
- Demo TypeScript (tsc in demos)
- Vite dev server

You'll receive status updates automatically.
