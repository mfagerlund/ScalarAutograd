import { Project, SyntaxKind } from "ts-morph";
import { z } from "zod";
import { CliError } from "../cli-error";

const renameKinds = ["class", "function", "variable", "method"] as const;

export const renameByNameSchema = z.object({
  name: z.string().min(1, "--name is required"),
  to: z.string().min(1, "--to is required"),
  kind: z.enum(renameKinds).optional(),
});

export type RenameByNameArgs = z.infer<typeof renameByNameSchema>;

export function renameByName(project: Project, args: RenameByNameArgs): void {
  let count = 0;
  for (const sf of project.getSourceFiles()) {
    const matches = sf.getDescendants().filter(node => {
      const id = node.getFirstDescendantByKind(SyntaxKind.Identifier);
      if (!id || id.getText() !== args.name) {
        return false;
      }
      if (!args.kind) {
        return true;
      }
      switch (args.kind) {
        case "class": return node.getKindName() === "ClassDeclaration";
        case "function": return node.getKindName() === "FunctionDeclaration";
        case "variable": return node.getKindName() === "VariableDeclaration";
        case "method": return node.getKindName() === "MethodDeclaration";
        default: return false;
      }
    });

    for (const match of matches) {
      const id = match.getFirstDescendantByKind(SyntaxKind.Identifier);
      if (id) {
        id.rename(args.to);
        count++;
      }
    }
  }

  if (count === 0) {
    const suffix = args.kind ? ` (kind=${args.kind})` : "";
    throw new CliError(`No declarations named '${args.name}' found${suffix}.`);
  }
}
