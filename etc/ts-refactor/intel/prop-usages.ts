import { JsxAttribute, Project, SyntaxKind } from "ts-morph";
import { z } from "zod";
import { dedupeUsageHits, getLineAndColumn, sortUsageHits, toWorkspaceRelative, UsageHit } from "./shared";

export const propUsagesSchema = z.object({
  component: z.string().min(1, "--component is required"),
  prop: z.string().min(1, "--prop is required"),
  json: z.boolean().optional(),
});

export type PropUsagesArgs = z.infer<typeof propUsagesSchema>;

export interface PropUsageResult {
  readonly component: string;
  readonly prop: string;
  readonly hits: UsageHit[];
}

export function findPropUsages(project: Project, args: PropUsagesArgs): PropUsageResult {
  const hits: UsageHit[] = [];
  for (const sf of project.getSourceFiles()) {
    if (!sf.getFilePath().endsWith(".tsx")) {
      continue;
    }
    if (sf.isInNodeModules()) {
      continue;
    }
    const elements = sf.getDescendantsOfKind(SyntaxKind.JsxOpeningElement).concat(
      sf.getDescendantsOfKind(SyntaxKind.JsxSelfClosingElement)
    );
    for (const el of elements) {
      const tag = el.getTagNameNode();
      if (!tag || tag.getText() !== args.component) {
        continue;
      }
      const attrs = el.getAttributes().filter(a => a.getKind() === SyntaxKind.JsxAttribute) as JsxAttribute[];
      for (const attr of attrs) {
        if (attr.getName() !== args.prop) {
          continue;
        }
        const { line, column } = getLineAndColumn(attr.getNameNode());
        hits.push({
          file: toWorkspaceRelative(sf.getFilePath()),
          line,
          column,
          kind: "jsx",
          context: attr.getText(),
        });
      }
    }
  }

  const deduped = sortUsageHits(dedupeUsageHits(hits));
  return {
    component: args.component,
    prop: args.prop,
    hits: deduped,
  };
}
