import { Project } from "ts-morph";
import { z } from "zod";
import { normalizeModuleSpecifier, requireSourceFile } from "../project";

export const moveSchema = z.object({
  from: z.string().min(1, "--from is required"),
  to: z.string().min(1, "--to is required"),
});

export type MoveArgs = z.infer<typeof moveSchema>;

export async function moveFile(project: Project, args: MoveArgs): Promise<void> {
  const sf = requireSourceFile(project, args.from);
  const oldPath = sf.getFilePath();
  sf.move(args.to);
  const newPath = sf.getFilePath();

  for (const file of project.getSourceFiles()) {
    for (const imp of file.getImportDeclarations()) {
      const target = imp.getModuleSpecifierSourceFile();
      if (!target) {
        continue;
      }
      const resolvedPath = target.getFilePath();
      if (resolvedPath === newPath || resolvedPath === oldPath) {
        imp.setModuleSpecifier(normalizeModuleSpecifier(file.getFilePath(), newPath));
      }
    }
  }
}
