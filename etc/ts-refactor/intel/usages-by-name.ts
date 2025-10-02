import { Project, ReferenceFindableNode } from "ts-morph";
import { z } from "zod";
import { CliError } from "../cli-error";
import {
  classifyReference,
  dedupeUsageHits,
  getLineAndColumn,
  sortUsageHits,
  summarizeUsageKinds,
  toWorkspaceRelative,
  UsageHit,
  UsageReport,
} from "./shared";
import { findDeclarationIdentifiers, nameKinds, NameKind } from "./declarations";

export const usagesByNameSchema = z.object({
  name: z.string().min(1, "--name is required"),
  kind: z.enum(nameKinds).optional(),
  json: z.boolean().optional(),
  summary: z.boolean().optional(),
});

export type UsagesByNameArgs = z.infer<typeof usagesByNameSchema>;

export interface UsagesByNameResult extends UsageReport {
  readonly summary: Record<string, number>;
}

export function findUsagesByName(project: Project, args: UsagesByNameArgs): UsagesByNameResult {
  const identifiers = findDeclarationIdentifiers(project, args.name, { kind: args.kind });
  if (!identifiers.length) {
    const suffix = args.kind ? ` (kind=${args.kind})` : "";
    throw new CliError(`No declarations named '${args.name}' found${suffix}.`);
  }

  const hits: UsageHit[] = [];
  for (const identifier of identifiers) {
    collectReferences(identifier, hits);
  }

  const deduped = sortUsageHits(dedupeUsageHits(hits));
  const summary = summarizeUsageKinds(deduped);
  return {
    symbol: args.name,
    hits: deduped,
    summary,
  };
}

function collectReferences(identifier: ReferenceFindableNode, hits: UsageHit[]): void {
  const references = identifier.findReferences();
  for (const ref of references) {
    for (const entry of ref.getReferences()) {
      const node = entry.getNode();
      const source = node.getSourceFile();
      if (source.isInNodeModules()) {
        continue;
      }
      const { line, column } = getLineAndColumn(node);
      hits.push({
        file: toWorkspaceRelative(source.getFilePath()),
        line,
        column,
        kind: classifyReference(entry, node),
      });
    }
  }
}
