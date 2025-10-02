import { Project, SyntaxKind, ExportAssignment } from "ts-morph";
import { z } from "zod";
import { requireSourceFile } from "../project";
import { CliError } from "../cli-error";

export const wrapWithMemoSchema = z.object({
  file: z.string().min(1, "--file is required"),
  export: z.string().min(1, "--export is required"),
});

export type WrapWithMemoArgs = z.infer<typeof wrapWithMemoSchema>;

export function wrapWithMemo(project: Project, args: WrapWithMemoArgs): void {
  const sf = requireSourceFile(project, args.file);
  const hasMemo = sf.getImportDeclarations().some(imp =>
    imp.getModuleSpecifierValue() === "react" && imp.getNamedImports().some(n => n.getName() === "memo"));

  if (!hasMemo) {
    const reactImport = sf.getImportDeclaration(imp => imp.getModuleSpecifierValue() === "react");
    if (reactImport) {
      reactImport.addNamedImport({ name: "memo" });
    } else {
      sf.addImportDeclaration({ moduleSpecifier: "react", namedImports: [{ name: "memo" }] });
    }
  }

  if (args.export === "default") {
    const assignment = sf.getFirstDescendantByKind(SyntaxKind.ExportAssignment) as ExportAssignment | undefined;
    if (!assignment) {
      throw new CliError("No default export assignment found");
    }
    const inner = assignment.getExpression().getText();
    assignment.setExpression(`memo(${inner})`);
    return;
  }

  const decls = sf.getExportedDeclarations().get(args.export);
  if (!decls || decls.length === 0) {
    throw new CliError(`No named export '${args.export}'`);
  }
  const id = `__Memo_${args.export}`;
  sf.addStatements(`const ${id} = memo(${args.export});\nexport { ${id} as ${args.export} };`);
}
