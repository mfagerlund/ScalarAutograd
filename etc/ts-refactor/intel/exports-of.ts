import { Project } from "ts-morph";
import { z } from "zod";
import { requireSourceFile } from "../project";
import { getLineAndColumn, toWorkspaceRelative } from "./shared";

export const exportsOfSchema = z.object({
  file: z.string().min(1, "--file is required"),
  json: z.boolean().optional(),
});

export type ExportsOfArgs = z.infer<typeof exportsOfSchema>;

export interface ExportEntry {
  readonly name: string;
  readonly kind: string;
  readonly isDefault: boolean;
  readonly location: {
    readonly file: string;
    readonly line: number;
    readonly column: number;
  };
}

export interface ExportsOfResult {
  readonly file: string;
  readonly exports: ExportEntry[];
}

export function listExports(project: Project, args: ExportsOfArgs): ExportsOfResult {
  const sf = requireSourceFile(project, args.file);
  const exported = sf.getExportedDeclarations();
  const entries: ExportEntry[] = [];

  exported.forEach((decls, name) => {
    for (const decl of decls) {
      const { line, column } = getLineAndColumn(decl);
      entries.push({
        name,
        kind: decl.getKindName(),
        isDefault: name === "default",
        location: {
          file: toWorkspaceRelative(decl.getSourceFile().getFilePath()),
          line,
          column,
        },
      });
    }
  });

  const sorted = entries.sort((a, b) =>
    a.name === b.name ? a.location.file.localeCompare(b.location.file) : a.name.localeCompare(b.name)
  );

  return {
    file: toWorkspaceRelative(sf.getFilePath()),
    exports: sorted,
  };
}

