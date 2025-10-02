import { CallExpression, Node, Project } from "ts-morph";
import { z } from "zod";
import { CliError } from "../cli-error";
import {
  dedupeUsageHits,
  getLineAndColumn,
  sortUsageHits,
  toWorkspaceRelative,
  UsageHit,
} from "./shared";
import { findDeclarationIdentifiers } from "./declarations";

export const callersSchema = z.object({
  name: z.string().min(1, "--name is required"),
  kind: z.enum(["function", "method", "variable"]).optional(),
  json: z.boolean().optional(),
});

export type CallersArgs = z.infer<typeof callersSchema>;

export interface CallersResult {
  readonly symbol: string;
  readonly hits: UsageHit[];
}

export function findCallers(project: Project, args: CallersArgs): CallersResult {
  const identifiers = findDeclarationIdentifiers(project, args.name, { kind: args.kind });
  if (!identifiers.length) {
    const suffix = args.kind ? ` (kind=${args.kind})` : "";
    throw new CliError(`No declarations named '${args.name}' found${suffix}.`);
  }

  const hits: UsageHit[] = [];
  for (const identifier of identifiers) {
    const references = identifier.findReferences();
    for (const ref of references) {
      for (const entry of ref.getReferences()) {
        const callExpr = getEnclosingCallExpression(entry.getNode());
        if (!callExpr) {
          continue;
        }
        const source = callExpr.getSourceFile();
        if (source.isInNodeModules()) {
          continue;
        }
        const { line, column } = getLineAndColumn(callExpr);
        hits.push({
          file: toWorkspaceRelative(source.getFilePath()),
          line,
          column,
          kind: "call",
          context: callExpr.getText(),
        });
      }
    }
  }

  const deduped = sortUsageHits(dedupeUsageHits(hits));
  return {
    symbol: args.name,
    hits: deduped,
  };
}

function getEnclosingCallExpression(node: Node): CallExpression | undefined {
  const parent = node.getParent();
  if (!parent) {
    return undefined;
  }
  if (Node.isCallExpression(parent) && parent.getExpression() === node) {
    return parent;
  }
  if (Node.isPropertyAccessExpression(parent) || Node.isElementAccessExpression(parent)) {
    const grand = parent.getParent();
    if (Node.isCallExpression(grand) && grand.getExpression() === parent) {
      return grand;
    }
  }
  return undefined;
}



