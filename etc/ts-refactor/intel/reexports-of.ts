import { Project } from "ts-morph";
import { z } from "zod";
import { toWorkspaceRelative } from "./shared";

export const reexportsOfSchema = z.object({
  module: z.string().min(1, "--module is required"),
  json: z.boolean().optional(),
});

export type ReexportsOfArgs = z.infer<typeof reexportsOfSchema>;

export interface ReexportEntry {
  readonly file: string;
  readonly moduleSpecifier: string;
  readonly named: { name: string; alias?: string }[];
  readonly namespace?: string;
  readonly isStar: boolean;
}

export interface ReexportsOfResult {
  readonly module: string;
  readonly hits: ReexportEntry[];
}

export function listReexports(project: Project, args: ReexportsOfArgs): ReexportsOfResult {
  const hits: ReexportEntry[] = [];
  for (const sf of project.getSourceFiles()) {
    if (sf.isInNodeModules()) {
      continue;
    }
    for (const exp of sf.getExportDeclarations()) {
      const spec = exp.getModuleSpecifierValue();
      if (spec !== args.module) {
        continue;
      }
      const named = exp.getNamedExports().map(n => ({
        name: n.getName(),
        alias: n.getAliasNode()?.getText() ?? undefined,
      }));
            const namespace = exp.getNamespaceExport()?.getName();
      hits.push({
        file: toWorkspaceRelative(sf.getFilePath()),
        moduleSpecifier: spec,
        named,
        namespace,
        isStar: namespace !== undefined || named.length === 0,
      });
    }
  }

  return {
    module: args.module,
    hits: hits.sort((a, b) => a.file.localeCompare(b.file)),
  };
}

