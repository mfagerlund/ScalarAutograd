import { ImportDeclaration, Project } from "ts-morph";
import { z } from "zod";
import { CliError } from "../cli-error";
import { requireSourceFile } from "../project";
import { toWorkspaceRelative } from "./shared";

export const whoImportsSchema = z
  .object({
    file: z.string().optional(),
    module: z.string().optional(),
    json: z.boolean().optional(),
  })
  .refine(args => Boolean(args.file) !== Boolean(args.module), {
    message: "Provide exactly one of --file or --module",
    path: ["file"],
  });

export type WhoImportsArgs = z.infer<typeof whoImportsSchema>;

export interface ImportHit {
  readonly file: string;
  readonly moduleSpecifier: string;
  readonly defaultImport?: string;
  readonly namedImports: string[];
  readonly namespaceImport?: string;
}

export interface WhoImportsResult {
  readonly target: string;
  readonly hits: ImportHit[];
}

export function findImporters(project: Project, args: WhoImportsArgs): WhoImportsResult {
  let targetFilePath: string | undefined;
  let targetSpecifier: string | undefined;

  if (args.file) {
    const sf = requireSourceFile(project, args.file);
    targetFilePath = sf.getFilePath();
  } else if (args.module) {
    targetSpecifier = args.module;
  }

  const hits: ImportHit[] = [];

  for (const sf of project.getSourceFiles()) {
    if (sf.isInNodeModules()) {
      continue;
    }
    for (const imp of sf.getImportDeclarations()) {
      if (targetFilePath) {
        const resolved = imp.getModuleSpecifierSourceFile();
        if (!resolved || resolved.getFilePath() !== targetFilePath) {
          continue;
        }
      } else if (targetSpecifier && imp.getModuleSpecifierValue() !== targetSpecifier) {
        continue;
      }
      hits.push(describeImport(sf.getFilePath(), imp));
    }
  }

  const targetLabel = targetSpecifier ?? toWorkspaceRelative(targetFilePath ?? "");
  return {
    target: targetLabel,
    hits: hits.sort((a, b) => a.file.localeCompare(b.file)),
  };
}

function describeImport(sourceFilePath: string, imp: ImportDeclaration): ImportHit {
  const named = imp.getNamedImports().map(n => n.getText());
  const namespace = imp.getNamespaceImport()?.getText();
  const defaultImport = imp.getDefaultImport()?.getText();
  return {
    file: toWorkspaceRelative(sourceFilePath),
    moduleSpecifier: imp.getModuleSpecifierValue(),
    defaultImport,
    namedImports: named,
    namespaceImport: namespace,
  };
}
