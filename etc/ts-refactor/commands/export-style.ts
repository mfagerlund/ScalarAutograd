import { Project, SourceFile, SyntaxKind, ExportAssignment } from "ts-morph";
import { z } from "zod";
import { requireSourceFile } from "../project";
import { CliError } from "../cli-error";

export const exportStyleSchema = z.object({
  file: z.string().min(1, "--file is required"),
  to: z.enum(["named", "default"], { errorMap: () => ({ message: "--to must be 'named' or 'default'" }) }),
  name: z.string().min(1).optional(),
});

export type ExportStyleArgs = z.infer<typeof exportStyleSchema>;

export function exportStyle(project: Project, args: ExportStyleArgs): void {
  const sf = requireSourceFile(project, args.file);
  if (args.to === "named") {
    convertDefaultToNamed(project, sf, args.name);
  } else {
    convertNamedToDefault(project, sf, args.name);
  }
}

function convertDefaultToNamed(project: Project, sf: SourceFile, nameHint?: string): void {
  let targetName = nameHint;
  const defFn = sf.getFirstDescendantByKind(SyntaxKind.FunctionDeclaration);
  const defCls = sf.getFirstDescendantByKind(SyntaxKind.ClassDeclaration);
  const expAssign = sf.getFirstDescendantByKind(SyntaxKind.ExportAssignment) as ExportAssignment | undefined;

  if (defFn?.isDefaultExport()) {
    targetName ||= defFn.getName() || "DefaultExport";
    defFn.toggleModifier("default", false);
    defFn.setIsExported(true);
  } else if (defCls?.isDefaultExport()) {
    targetName ||= defCls.getName() || "DefaultExport";
    defCls.toggleModifier("default", false);
    defCls.setIsExported(true);
  } else if (expAssign) {
    const expr = expAssign.getExpression().getText();
    targetName ||= expr;
    expAssign.replaceWithText(`export { ${expr} as ${targetName} };`);
  }

  if (!targetName) {
    throw new CliError("Cannot determine export name; pass --name");
  }

  for (const file of project.getSourceFiles()) {
    for (const imp of file.getImportDeclarations()) {
      const target = imp.getModuleSpecifierSourceFile();
      if (!target || target.getFilePath() !== sf.getFilePath()) {
        continue;
      }
      const defaultImport = imp.getDefaultImport();
      if (defaultImport) {
        const local = defaultImport.getText();
        defaultImport.remove();
        const named = imp.getNamedImports();
        if (!named.some(n => n.getName() === targetName!)) {
          imp.addNamedImport({
            name: targetName!,
            alias: local !== targetName ? local : undefined,
          });
        }
      }
    }
  }
}

function convertNamedToDefault(project: Project, sf: SourceFile, nameHint?: string): void {
  const defaultName = nameHint || sf.getDefaultExportSymbol()?.getName() || sf.getBaseNameWithoutExtension();
  if (!defaultName) {
    throw new CliError("Pass --name for default export");
  }

  const named = sf.getExportedDeclarations().get(defaultName);
  if (!named || named.length === 0) {
    throw new CliError(`No named export '${defaultName}' in ${sf.getFilePath()}`);
  }

  if (!sf.getDefaultExportSymbol()) {
    sf.addExportAssignment({ isExportEquals: false, expression: defaultName });
  }

  for (const file of project.getSourceFiles()) {
    for (const imp of file.getImportDeclarations()) {
      const target = imp.getModuleSpecifierSourceFile();
      if (!target || target.getFilePath() !== sf.getFilePath()) {
        continue;
      }
      const namedImport = imp.getNamedImports().find(n => n.getName() === defaultName);
      if (!namedImport) {
        continue;
      }
      const alias = namedImport.getAliasNode()?.getText();
      const local = alias || defaultName;
      namedImport.remove();
      if (!imp.getDefaultImport()) {
        imp.setDefaultImport(local);
      }
    }
  }
}
