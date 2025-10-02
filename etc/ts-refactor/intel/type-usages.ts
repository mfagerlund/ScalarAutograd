import { Project, ReferenceFindableNode } from "ts-morph";
import { z } from "zod";
import { summarizeUsageKinds, UsageHit, UsageReport, dedupeUsageHits, sortUsageHits, getLineAndColumn, classifyReference, toWorkspaceRelative } from "./shared";
import { findDeclarationIdentifiers } from "./declarations";

export const typeUsagesSchema = z.object({
  name: z.string().min(1, "--name is required"),
  json: z.boolean().optional(),
  summary: z.boolean().optional(),
});

export type TypeUsagesArgs = z.infer<typeof typeUsagesSchema>;

export interface TypeUsagesResult extends UsageReport {
  readonly summary: Record<string, number>;
}

export function findTypeUsages(project: Project, args: TypeUsagesArgs): TypeUsagesResult {
  const identifiers = collectTypeLikeIdentifiers(project, args.name);
  const hits: UsageHit[] = [];
  for (const identifier of identifiers) {
    const references = identifier.findReferences();
    for (const ref of references) {
      for (const entry of ref.getReferences()) {
        const node = entry.getNode();
        const source = node.getSourceFile();
        if (source.isInNodeModules()) {
          continue;
        }
        const kind = classifyReference(entry, node);
        if (kind !== "type" && !entry.isDefinition()) {
          continue;
        }
        const { line, column } = getLineAndColumn(node);
        hits.push({
          file: toWorkspaceRelative(source.getFilePath()),
          line,
          column,
          kind,
        });
      }
    }
  }

  const deduped = sortUsageHits(dedupeUsageHits(hits));
  const summary = summarizeUsageKinds(deduped);
  return {
    symbol: args.name,
    hits: deduped,
    summary,
  };
}

function collectTypeLikeIdentifiers(project: Project, name: string): ReferenceFindableNode[] {
  const identifiers = [
    ...findDeclarationIdentifiers(project, name, { kind: "type" }),
    ...findDeclarationIdentifiers(project, name, { kind: "class" }),
  ];
  const seen = new Set<string>();
  const out: ReferenceFindableNode[] = [];
  for (const identifier of identifiers) {
    const key = `${identifier.getSourceFile().getFilePath()}:${identifier.getStart()}`;
    if (seen.has(key)) {
      continue;
    }
    seen.add(key);
    out.push(identifier);
  }
  return out;
}
