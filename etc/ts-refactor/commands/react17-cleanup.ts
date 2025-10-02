import { Project, SyntaxKind } from "ts-morph";
import { z } from "zod";

export const react17CleanupSchema = z.object({});

export function react17Cleanup(project: Project): number {
  let removed = 0;
  for (const sf of project.getSourceFiles()) {
    for (const imp of sf.getImportDeclarations()) {
      if (imp.getModuleSpecifierValue() !== "react") {
        continue;
      }
      const defaultImport = imp.getDefaultImport();
      if (!defaultImport) {
        continue;
      }
      const ident = defaultImport.getText();
      const references = sf.getDescendantsOfKind(SyntaxKind.Identifier).filter(id => id.getText() === ident);
      if (references.length <= 1) {
        defaultImport.remove();
        removed++;
      }
      if (!imp.getDefaultImport() && imp.getNamedImports().length === 0) {
        imp.remove();
      }
    }
  }
  return removed;
}
