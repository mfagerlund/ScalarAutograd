import { Project } from "ts-morph";
import { z } from "zod";
import {
  classifyReference,
  getIdentifierAtPosition,
  getLineAndColumn,
  summarizeUsageKinds,
  dedupeUsageHits,
  sortUsageHits,
  toWorkspaceRelative,
  UsageHit,
  UsageReport,
} from "./shared";

export const usagesSchema = z.object({
  file: z.string().min(1, "--file is required"),
  line: z.coerce.number().int().min(1, "--line must be 1-based"),
  col: z.coerce.number().int().min(1, "--col must be 1-based"),
  json: z.boolean().optional(),
  summary: z.boolean().optional(),
});

export type UsagesArgs = z.infer<typeof usagesSchema>;

export interface UsagesResult extends UsageReport {
  readonly summary: Record<string, number>;
}

export function findUsagesAt(project: Project, args: UsagesArgs): UsagesResult {
  const identifier = getIdentifierAtPosition(project, args.file, args.line, args.col);
  const symbol = identifier.getSymbol();
  const symbolName = symbol?.getName() ?? identifier.getText();
  const references = identifier.findReferences();
  const hits: UsageHit[] = [];

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

  const deduped = sortUsageHits(dedupeUsageHits(hits));
  const summary = summarizeUsageKinds(deduped);
  return {
    symbol: symbolName,
    hits: deduped,
    summary,
  };
}
