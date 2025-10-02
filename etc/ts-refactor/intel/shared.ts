import { Node, Project, ReferenceEntry, ReferenceFindableNode, SyntaxKind } from "ts-morph";
import path from "node:path";
import { CliError } from "../cli-error";
import { requireSourceFile } from "../project";

export type UsageKind =
  | "def"
  | "decl"
  | "read"
  | "write"
  | "import"
  | "export"
  | "jsx"
  | "type"
  | "call"
  | "callee";

export interface Location {
  readonly file: string;
  readonly line: number;
  readonly column: number;
}

export interface UsageHit extends Location {
  readonly kind: UsageKind;
  readonly context?: string;
}

export interface UsageReport {
  readonly symbol: string;
  readonly hits: UsageHit[];
}

export function toWorkspaceRelative(filePath: string): string {
  return path.relative(process.cwd(), filePath).split(path.sep).join("/");
}

export function classifyReference(ref: ReferenceEntry, node: Node): UsageKind {
  if (ref.isDefinition()) return "def";
  if (ref.isWriteAccess?.()) return "write";
  const parent = node.getParent();
  if (!parent) return "read";
  const kind = parent.getKind();
  if (
    kind === SyntaxKind.ImportSpecifier ||
    kind === SyntaxKind.ImportClause ||
    kind === SyntaxKind.ImportEqualsDeclaration
  ) {
    return "import";
  }
  if (kind === SyntaxKind.ExportSpecifier || kind === SyntaxKind.ExportAssignment) {
    return "export";
  }
  if (kind === SyntaxKind.JsxOpeningElement || kind === SyntaxKind.JsxSelfClosingElement) {
    return "jsx";
  }
  if (isTypePosition(node)) {
    return "type";
  }
  if (isDeclarationKind(kind)) {
    return "decl";
  }
  return "read";
}

export function isTypePosition(node: Node): boolean {
  for (let current: Node | undefined = node; current; current = current.getParent()) {
    const kind = current.getKind();
    switch (kind) {
      case SyntaxKind.TypeReference:
      case SyntaxKind.TypeAliasDeclaration:
      case SyntaxKind.InterfaceDeclaration:
      case SyntaxKind.HeritageClause:
      case SyntaxKind.TypeLiteral:
      case SyntaxKind.TypeQuery:
      case SyntaxKind.TypePredicate:
      case SyntaxKind.TypeParameter:
        return true;
      case SyntaxKind.CallExpression:
      case SyntaxKind.NewExpression:
      case SyntaxKind.BinaryExpression:
        return false;
    }
  }
  return false;
}

function isDeclarationKind(kind: SyntaxKind): boolean {
  switch (kind) {
    case SyntaxKind.FunctionDeclaration:
    case SyntaxKind.MethodDeclaration:
    case SyntaxKind.ClassDeclaration:
    case SyntaxKind.VariableDeclaration:
    case SyntaxKind.EnumDeclaration:
    case SyntaxKind.InterfaceDeclaration:
    case SyntaxKind.TypeAliasDeclaration:
    case SyntaxKind.PropertyDeclaration:
    case SyntaxKind.ModuleDeclaration:
      return true;
    default:
      return false;
  }
}

export function getIdentifierAtPosition(
  project: Project,
  file: string,
  line: number,
  column: number
): ReferenceFindableNode {
  const sf = requireSourceFile(project, file);
  const pos = sf.getPositionFromLineAndColumn(line, column);
  const node = sf.getDescendantAtPos(pos) ?? sf.getDescendantAtStart(pos);
  if (!node) {
    throw new CliError(`No node at ${sf.getFilePath()}:${line}:${column}`);
  }
  const identifier = node.asKind(SyntaxKind.Identifier) ?? node.getFirstDescendantByKind(SyntaxKind.Identifier);
  if (!identifier) {
    throw new CliError(`No identifier at ${sf.getFilePath()}:${line}:${column}`);
  }
  if (!isReferenceFindable(identifier)) {
    throw new CliError("Identifier cannot produce references at this location.");
  }
  return identifier;
}

function isReferenceFindable(node: Node): node is ReferenceFindableNode {
  return typeof (node as ReferenceFindableNode).findReferences === "function";
}

export function uniqueByKey<T>(items: Iterable<T>, key: (value: T) => string): T[] {
  const seen = new Set<string>();
  const out: T[] = [];
  for (const item of items) {
    const k = key(item);
    if (seen.has(k)) continue;
    seen.add(k);
    out.push(item);
  }
  return out;
}

export function getLineAndColumn(node: Node): { line: number; column: number } {
  const source = node.getSourceFile();
  const { line, column } = source.getLineAndColumnAtPos(node.getStart());
  return { line, column };
}

export function summarizeUsageKinds(hits: UsageHit[]): Record<UsageKind, number> {
  const summary: Record<UsageKind, number> = {
    def: 0,
    decl: 0,
    read: 0,
    write: 0,
    import: 0,
    export: 0,
    jsx: 0,
    type: 0,
    call: 0,
    callee: 0,
  };
  for (const hit of hits) {
    summary[hit.kind]++;
  }
  return summary;
}

export function formatUsageHits(hits: UsageHit[]): string {
  if (!hits.length) {
    return "(no results)";
  }
  const width = Math.max(...hits.map(h => h.kind.length));
  return hits
    .map(hit => `${hit.kind.padEnd(width)}  ${hit.file}:${hit.line}:${hit.column}`)
    .join("\n");
}

export function dedupeUsageHits(hits: UsageHit[]): UsageHit[] {
  const seen = new Set<string>();
  const out: UsageHit[] = [];
  for (const hit of hits) {
    const key = `${hit.file}:${hit.line}:${hit.column}:${hit.kind}`;
    if (seen.has(key)) {
      continue;
    }
    seen.add(key);
    out.push(hit);
  }
  return out;
}

export function sortUsageHits(hits: UsageHit[]): UsageHit[] {
  return [...hits].sort((a, b) =>
    a.file === b.file
      ? a.line === b.line
        ? a.column === b.column
          ? a.kind.localeCompare(b.kind)
          : a.column - b.column
        : a.line - b.line
      : a.file.localeCompare(b.file)
  );
}
