import { Node, Project, ReferenceFindableNode } from "ts-morph";

export const nameKinds = ["class", "function", "variable", "type", "method"] as const;
export type NameKind = (typeof nameKinds)[number];

export interface DeclarationQueryOptions {
  readonly kind?: NameKind;
}

export function findDeclarationIdentifiers(
  project: Project,
  name: string,
  options: DeclarationQueryOptions = {}
): ReferenceFindableNode[] {
  const results: ReferenceFindableNode[] = [];
  for (const sf of project.getSourceFiles()) {
    if (sf.isInNodeModules()) {
      continue;
    }
    sf.forEachDescendant(node => {
      if (!Node.isIdentifier(node)) {
        return;
      }
      if (node.getText() !== name) {
        return;
      }
      const parent = node.getParent();
      if (!parent) {
        return;
      }
      if (options.kind && !matchesKind(parent, options.kind)) {
        return;
      }
      if (!isDeclarationNameNode(parent, node)) {
        return;
      }
      if (!isReferenceFindable(node)) {
        return;
      }
      results.push(node);
    });
  }
  return dedupeByNode(results);
}

function matchesKind(node: Node, kind: NameKind): boolean {
  switch (kind) {
    case "class":
      return Node.isClassDeclaration(node);
    case "function":
      return Node.isFunctionDeclaration(node);
    case "variable":
      return Node.isVariableDeclaration(node);
    case "type":
      return Node.isTypeAliasDeclaration(node) || Node.isInterfaceDeclaration(node) || Node.isEnumDeclaration(node);
    case "method":
      return Node.isMethodDeclaration(node) || Node.isMethodSignature(node);
    default:
      return false;
  }
}

function isDeclarationNameNode(parent: Node, identifier: Node): boolean {
  const anyParent = parent as { getNameNode?: () => Node | undefined };
  const nameNode = anyParent.getNameNode?.();
  if (nameNode) {
    return nameNode === identifier;
  }
  if (Node.isBindingElement(parent)) {
    const name = parent.getNameNode();
    return name === identifier;
  }
  return false;
}

function isReferenceFindable(node: Node): node is ReferenceFindableNode {
  return typeof (node as ReferenceFindableNode).findReferences === "function";
}

function dedupeByNode(nodes: ReferenceFindableNode[]): ReferenceFindableNode[] {
  const seen = new Set<string>();
  const out: ReferenceFindableNode[] = [];
  for (const node of nodes) {
    const source = node.getSourceFile().getFilePath();
    const key = `${source}:${node.getStart()}`;
    if (seen.has(key)) {
      continue;
    }
    seen.add(key);
    out.push(node);
  }
  return out;
}
