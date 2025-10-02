Yes—add read-only “intel” verbs:

### Useful subcommands

* `usages --file F --line L --col C [--json] [--summary]`
  Find all references of the symbol at cursor. Classify as `def|decl|read|write|import|export|jsx|type`.
* `usages-by-name --name Foo [--kind class|function|variable|type|method] [--json]`
  Resolve declarations by name, then list refs.
* `who-imports --file src/x.ts` or `who-imports --module '@/ui/Button'`
  List files importing that module/file.
* `callers --name fn` / `callees --file F --line L --col C`
  Reverse/forward call graph for functions.
* `prop-usages --component Button --prop variant`
  JSX attribute usages across TSX.
* `type-usages --name MyType`
  Type-position references only (ignore value-space).
* `exports-of --file src/x.ts` / `reexports-of --module '@/pkg'`
  Where symbols are exported/re-exported from.
* `--json` on any of the above to emit machine-readable results for the AI.

### Implementation sketch (ts-morph)

```ts
function findUsagesAt(project, file, line, col) {
  const sf = project.getSourceFileOrThrow(file);
  const pos = sf.getPositionFromLineAndColumn(line, col);
  const node = sf.getDescendantAtPos(pos)
              ?.getFirstDescendantByKind(SyntaxKind.Identifier) ?? null;
  if (!node) return [];

  const refs = node.findReferences(); // tsserver “Find All References”
  const out = [];
  for (const r of refs) {
    for (const ref of r.getReferences()) {
      const n = ref.getNode();
      const f = n.getSourceFile();
      const { line, column } = f.getLineAndColumnAtPos(n.getStart());
      out.push({
        file: f.getFilePath(),
        line, col: column,
        kind: classify(ref, n), // import/def/write/jsx/type/etc.
      });
    }
  }
  return out;
}

function classify(ref, n) {
  if (ref.isDefinition()) return "def";
  if (ref.isWriteAccess?.()) return "write";
  const p = n.getParent();
  if (Node.isImportSpecifier(p) || Node.isImportClause(p)) return "import";
  if (Node.isExportSpecifier(p)) return "export";
  if (Node.isJsxOpeningLikeElement(p)) return "jsx";
  if (isTypePosition(n)) return "type";
  return "read";
}
function isTypePosition(n) {
  const k = n.getParent()?.getKind();
  return k && [
    SyntaxKind.TypeReference, SyntaxKind.InterfaceDeclaration,
    SyntaxKind.TypeAliasDeclaration, SyntaxKind.HeritageClause,
    SyntaxKind.TypeLiteral, SyntaxKind.TypeQuery,
  ].includes(k);
}
```

### UX examples

```
# All refs at cursor, summarized
npx tsx ts-refactor.ts usages --file src/user.ts --line 12 --col 7 --summary

# JSON for AI
npx tsx ts-refactor.ts usages --file src/Button.tsx --line 5 --col 14 --json > out/refs.json

# Who imports a module
npx tsx ts-refactor.ts who-imports --module '@/ui/Button'

# Callers of a function by name
npx tsx ts-refactor.ts callers --name fetchUser --json
```

Keep your existing type-gates (`preflight/postflight`) but they can default to **read-only** for these verbs. If you want, add `--open-editor 'code --goto {file}:{line}:{col}'` to jump to each hit.

(Aside: there’s a small typo in `normalizeModuleSpecifier` replacing backslashes—flag if you want me to fix.)
