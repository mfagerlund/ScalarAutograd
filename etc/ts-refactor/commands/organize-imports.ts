import { Project } from "ts-morph";
import { z } from "zod";

export const organizeImportsSchema = z.object({});

export function organizeImports(project: Project): void {
  for (const sf of project.getSourceFiles()) {
    sf.organizeImports();
  }
}
