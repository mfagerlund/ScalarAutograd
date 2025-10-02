import { Project, SourceFile, SyntaxKind, JsxOpeningLikeElement, JsxAttribute } from "ts-morph";
import { z } from "zod";

const baseSchema = z.object({
  from: z.string().min(1, "--from is required"),
  to: z.string().min(1, "--to is required"),
  tag: z.string().min(1).optional(),
  module: z.string().min(1).optional(),
  import: z.string().min(1).optional(),
  addDefault: z.string().min(1).optional(),
  map: z.string().min(1).optional(),
});

export const jsxPropRenameSchema = baseSchema;
export const jsxPropMigrateSchema = baseSchema.extend({
  map: z.string().min(1, "--map is required"),
});

export type JsxPropArgs = z.infer<typeof baseSchema>;

export function renameJsxProps(project: Project, args: JsxPropArgs): number {
  const parsedMap = parseMap(args.map);
  let totalChanged = 0;
  for (const sf of project.getSourceFiles()) {
    totalChanged += renameJsxPropInFile(sf, {
      from: args.from,
      to: args.to,
      tag: args.tag,
      module: args.module,
      importName: args.import,
      addDefault: args.addDefault,
      map: parsedMap,
    });
  }
  return totalChanged;
}

interface JsxRenameOptions {
  from: string;
  to: string;
  tag?: string;
  module?: string;
  importName?: string;
  addDefault?: string;
  map: Record<string, string>;
}

function renameJsxPropInFile(sf: SourceFile, opts: JsxRenameOptions): number {
  const { from, to, tag, module: modSpec, importName, addDefault, map } = opts;
  const localTags = collectLocalTags(sf, tag, modSpec, importName);
  if (!localTags.size) {
    return 0;
  }

  let changed = 0;
  eachJsxElement(sf, el => {
    const tagNode = el.getTagNameNode();
    if (tagNode.getKind() !== SyntaxKind.Identifier) {
      return;
    }
    const tagName = tagNode.getText();
    if (!localTags.has(tagName)) {
      return;
    }

    const attrs = el.getAttributes().filter(a => a.getKind() === SyntaxKind.JsxAttribute) as JsxAttribute[];
    let hasRenamed = false;

    for (const attr of attrs) {
      if (attr.getName() !== from) {
        continue;
      }
      attr.set({ name: to });
      hasRenamed = true;
      changed++;
      applyValueMapping(attr, map);
    }

    if (!hasRenamed && addDefault && !attrs.some(a => a.getName() === to)) {
      const [name, value = ""] = addDefault.split("=");
      if (name === to) {
        const initializer = /^("|\{)/.test(value) ? value : `"${value}"`;
        el.addAttribute({ name: to, initializer });
        changed++;
      }
    }
  });

  return changed;
}

function collectLocalTags(sf: SourceFile, tag?: string, modSpec?: string, importName?: string): Set<string> {
  const names = new Set<string>();
  if (tag) {
    names.add(tag);
  }
  if (modSpec && importName) {
    for (const imp of sf.getImportDeclarations()) {
      if (imp.getModuleSpecifierValue() !== modSpec) {
        continue;
      }
      const defaultImport = imp.getDefaultImport();
      if (importName === "default" && defaultImport) {
        names.add(defaultImport.getText());
      }
      for (const named of imp.getNamedImports()) {
        const name = named.getName();
        const alias = named.getAliasNode()?.getText();
        if (name === importName) {
          names.add(alias ?? name);
        }
      }
    }
  }
  return names;
}

function eachJsxElement(sf: SourceFile, fn: (el: JsxOpeningLikeElement) => void): void {
  const openings = sf.getDescendantsOfKind(SyntaxKind.JsxOpeningElement)
    .concat(sf.getDescendantsOfKind(SyntaxKind.JsxSelfClosingElement));
  for (const el of openings) {
    fn(el);
  }
}

function applyValueMapping(attr: JsxAttribute, map: Record<string, string>): void {
  if (!Object.keys(map).length) {
    return;
  }
  const init = attr.getInitializer();
  if (!init) {
    if (map["true"]) {
      attr.setInitializer(`"${map["true"]}"`);
    }
    return;
  }
  if (init.getKind() === SyntaxKind.StringLiteral) {
    const literal = (init as any).getLiteralText();
    if (literal in map) {
      init.replaceWithText(`"${map[literal]}"`);
    }
    return;
  }
  if (init.getKind() === SyntaxKind.JsxExpression) {
    const expr = (init as any).getExpression();
    if (!expr) {
      return;
    }
    const text = expr.getText();
    if (text in map) {
      (init as any).setExpression(`"${map[text]}"`);
    }
    if (text === "true" && map["true"]) {
      (init as any).setExpression(`"${map["true"]}"`);
    }
    if (text === "false" && map["false"]) {
      (init as any).setExpression(`"${map["false"]}"`);
    }
  }
}

function parseMap(map: string | undefined): Record<string, string> {
  const out: Record<string, string> = {};
  if (!map) {
    return out;
  }
  for (const pair of map.split(";")) {
    const [rawKey, rawValue] = pair.split("->");
    if (!rawKey) {
      continue;
    }
    const key = rawKey.trim();
    const value = (rawValue ?? "").trim().replace(/^['\"]|['\"]$/g, "");
    out[key] = value;
  }
  return out;
}
