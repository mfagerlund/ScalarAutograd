import { Project, SyntaxKind, Node } from "ts-morph";
import { z } from "zod";
import { requireSourceFile } from "../project";
import { CliError } from "../cli-error";

export const renameSchema = z.object({
  file: z.string().min(1, "--file is required"),
  line: z.coerce.number().int().min(1, "--line must be a 1-based integer"),
  col: z.coerce.number().int().min(1, "--col must be a 1-based integer"),
  to: z.string().min(1, "--to is required"),
});

export type RenameArgs = z.infer<typeof renameSchema>;

export function rename(project: Project, args: RenameArgs): void {
  const sf = requireSourceFile(project, args.file);
  const pos = sf.getPositionFromLineAndColumn(args.line, args.col);
  const node = sf.getDescendantAtPos(pos) ?? sf.getDescendantAtStart(pos);
  if (!node) {
    throw new CliError(`No node at ${sf.getFilePath()}:${args.line}:${args.col}`);
  }

  const identifier = getIdentifierForRename(node);
  if (identifier) {
    identifier.rename(args.to);
    return;
  }

  const symbol = (node as Node & { getSymbol?: () => any }).getSymbol?.();
  const declaration = symbol?.getDeclarations?.()?.[0];
  if (declaration && typeof (declaration as any).rename === "function") {
    (declaration as any).rename(args.to);
    return;
  }

  throw new CliError("Cannot determine a rename target at the given position.");
}

function getIdentifierForRename(node: Node) {
  return node.asKind(SyntaxKind.Identifier) ?? node.getFirstDescendantByKind(SyntaxKind.Identifier);
}
