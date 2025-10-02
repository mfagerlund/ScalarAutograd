import { FunctionLikeDeclaration, Node, Project, SyntaxKind } from "ts-morph";
import { z } from "zod";
import { CliError } from "../cli-error";
import { requireSourceFile } from "../project";
import {
  dedupeUsageHits,
  getLineAndColumn,
  sortUsageHits,
  toWorkspaceRelative,
  UsageHit,
} from "./shared";

export const calleesSchema = z.object({
  file: z.string().min(1, "--file is required"),
  line: z.coerce.number().int().min(1, "--line must be 1-based"),
  col: z.coerce.number().int().min(1, "--col must be 1-based"),
  json: z.boolean().optional(),
});

export type CalleesArgs = z.infer<typeof calleesSchema>;

export interface CalleesResult {
  readonly functionName: string;
  readonly hits: UsageHit[];
}

export function findCallees(project: Project, args: CalleesArgs): CalleesResult {
  const sf = requireSourceFile(project, args.file);
  const pos = sf.getPositionFromLineAndColumn(args.line, args.col);
  const node = sf.getDescendantAtPos(pos) ?? sf.getDescendantAtStart(pos);
  if (!node) {
    throw new CliError(`No node at ${sf.getFilePath()}:${args.line}:${args.col}`);
  }

  const fn = findFunctionLikeAncestor(node);
  if (!fn) {
    throw new CliError("No function-like declaration at the provided location.");
  }

  const functionName = fn.getSymbol()?.getName() ?? fn.getName?.() ?? "(anonymous)";
  const hits: UsageHit[] = [];

  const callExpressions = fn.getDescendantsOfKind(SyntaxKind.CallExpression);
  for (const call of callExpressions) {
    const decls = resolveCalleeDeclarations(call);
    if (!decls.length) {
      continue;
    }
    for (const decl of decls) {
      const source = decl.getSourceFile();
      if (source.isInNodeModules()) {
        continue;
      }
      const { line, column } = getLineAndColumn(decl);
      hits.push({
        file: toWorkspaceRelative(source.getFilePath()),
        line,
        column,
        kind: "callee",
        context: call.getText(),
      });
    }
  }

  const deduped = sortUsageHits(dedupeUsageHits(hits));
  return {
    functionName,
    hits: deduped,
  };
}

function findFunctionLikeAncestor(node: Node): FunctionLikeDeclaration | undefined {
  return node.getAncestors().find(ancestor => Node.isFunctionLikeDeclaration(ancestor)) as FunctionLikeDeclaration | undefined;
}

function resolveCalleeDeclarations(call: import("ts-morph").CallExpression): Node[] {
  const symbol = call.getExpression().getSymbol();
  if (symbol) {
    const decls = symbol.getDeclarations();
    if (decls && decls.length) {
      return decls;
    }
  }
  const signature = call.getSignature();
  const decl = signature?.getDeclaration();
  return decl ? [decl] : [];
}
