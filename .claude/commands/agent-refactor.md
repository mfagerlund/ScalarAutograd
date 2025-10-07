---
description: Use agent-refactor for type-safe refactoring operations (project, gitignored)
argument-hint: [ts command and args]
---

Use agent-refactor for type-safe refactoring operations on this project.

**IMPORTANT:** Run agent-refactor commands FROM THIS PROJECT DIRECTORY (C:/Dev/ScalarAutograd), NOT from the agent-refactor directory!

**Before using any command:** Run with `--help` to understand all options:
```bash
npx agent-refactor ts <command> --help
```

Execute the following command:
```bash
npx agent-refactor $ARGUMENTS
```

Common examples:
- `ts rename --file src/Value.ts --line 10 --col 5 --to newName`
- `ts add-import --to src/App.ts --from src/Value.ts --import Value`
- `ts move --from src/old.ts --to src/new.ts`
- `ts move --from src/File1.ts,src/File2.ts,src/File3.ts --to src/utils` (bulk move)
- `ts barrel-file --dir src/operations`
- `ts find-usages --file src/Value.ts --line 10 --col 5`
- `ts find-importers --file src/utils.ts`
- `ts fix-absolute-imports --project . --dry-run` (preview broken import fixes)
- `ts find-orphans --project . --exclude "*.test.ts"` (find unused files)

All file paths are relative to the ScalarAutograd project root.

**Safety flags:**
- `--dry-run` - Preview changes without saving
- `--accept-breaking` - Allow new type errors (use cautiously)

**Always check --help for comprehensive documentation including:**
- Full description of what the command does
- All available flags and options
- Examples and use cases
- Expected output format
