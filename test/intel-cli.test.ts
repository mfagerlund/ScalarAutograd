import { describe, it, expect, beforeAll, afterAll } from "vitest";
import { mkdtempSync, rmSync, writeFileSync, mkdirSync } from "node:fs";
import path from "node:path";
import { Project, ts } from "ts-morph";
import { findUsagesAt } from "../etc/ts-refactor/intel/usages";
import { findUsagesByName } from "../etc/ts-refactor/intel/usages-by-name";
import { findImporters } from "../etc/ts-refactor/intel/who-imports";
import { findCallers } from "../etc/ts-refactor/intel/callers";
import { findCallees } from "../etc/ts-refactor/intel/callees";
import { findPropUsages } from "../etc/ts-refactor/intel/prop-usages";
import { findTypeUsages } from "../etc/ts-refactor/intel/type-usages";
import { listExports } from "../etc/ts-refactor/intel/exports-of";
import { listReexports } from "../etc/ts-refactor/intel/reexports-of";

interface FixtureContext {
  project: Project;
  dir: string;
  paths: Record<string, string>;
}

let fixture: FixtureContext;

beforeAll(() => {
  fixture = createFixtureProject();
});

afterAll(() => {
  rmSync(fixture.dir, { recursive: true, force: true });
});

describe("intel commands", () => {
  it("finds usages at a position", () => {
    const { project, paths } = fixture;
    const lib = project.getSourceFileOrThrow(paths.lib);
    const foo = lib.getFunctionOrThrow("foo");
    const nameNode = foo.getNameNode();
    const { line, column } = lib.getLineAndColumnAtPos(nameNode.getStart());
    const result = findUsagesAt(project, { file: paths.lib, line, col: column });
    expect(result.symbol).toBe("foo");
    const files = result.hits.map(hit => hit.file);
    expect(files.some(f => f.endsWith("consumer.tsx"))).toBeTruthy();
    expect(files.some(f => f.endsWith("other-consumer.ts"))).toBeTruthy();
  });

  it("finds usages by name with summary", () => {
    const { project } = fixture;
    const result = findUsagesByName(project, { name: "foo" });
    expect(result.summary.call ?? 0).toBeGreaterThan(0);
  });

  it("detects callers", () => {
    const { project } = fixture;
    const result = findCallers(project, { name: "foo" });
    expect(result.hits.length).toBeGreaterThanOrEqual(2);
    expect(result.hits.every(hit => hit.kind === "call")).toBe(true);
  });

  it("resolves callees for a function", () => {
    const { project, paths } = fixture;
    const lib = project.getSourceFileOrThrow(paths.lib);
    const bar = lib.getFunctionOrThrow("bar");
    const nameNode = bar.getNameNode();
    const { line, column } = lib.getLineAndColumnAtPos(nameNode.getStart());
    const result = findCallees(project, { file: paths.lib, line, col: column });
    expect(result.functionName).toBe("bar");
    expect(result.hits.some(hit => hit.file.endsWith("lib.ts"))).toBe(true);
    expect(result.hits.every(hit => hit.kind === "callee")).toBe(true);
  });

  it("lists importers for a file", () => {
    const { project, paths } = fixture;
    const result = findImporters(project, { file: paths.lib });
    const files = result.hits.map(hit => hit.file);
    expect(files.some(f => f.endsWith("consumer.tsx"))).toBeTruthy();
    expect(files.some(f => f.endsWith("other-consumer.ts"))).toBeTruthy();
  });

  it("collects prop usages", () => {
    const { project } = fixture;
    const result = findPropUsages(project, { component: "Button", prop: "variant" });
    expect(result.hits.length).toBe(1);
    expect(result.hits[0].context).toContain("variant=\"primary\"");
  });

  it("filters type usages", () => {
    const { project } = fixture;
    const result = findTypeUsages(project, { name: "Widget" });
    expect(result.hits.some(hit => hit.file.endsWith("consumer.tsx"))).toBe(true);
    expect(result.hits.every(hit => ["type", "def", "decl"].includes(hit.kind))).toBe(true);
  });

  it("summarises exports", () => {
    const { project, paths } = fixture;
    const result = listExports(project, { file: paths.lib });
    const exportNames = result.exports.map(entry => entry.name);
    expect(exportNames).toContain("foo");
    expect(exportNames).toContain("default");
  });

  it("lists re-export declarations", () => {
    const { project } = fixture;
    const result = listReexports(project, { module: "./lib" });
    expect(result.hits.length).toBeGreaterThanOrEqual(2);
    expect(result.hits.some(hit => hit.file.endsWith("reexports.ts"))).toBe(true);
  });
});

function createFixtureProject(): FixtureContext {
  const base = path.join(process.cwd(), "test", ".tmp-intel");
  mkdirSync(base, { recursive: true });
  const dir = mkdtempSync(path.join(base, "run-"));

  const files: Record<string, string> = {
    "src/lib.ts": `
export function foo() {
  return "foo";
}

export function bar() {
  return foo();
}

export const arrow = () => foo();

export class Greeter {
  greet() {
    return foo();
  }
}

export type Widget = { id: string };

export interface Gadget {
  id: string;
}

export const Button = (props: { variant: string; disabled?: boolean }) => props.variant;

export default function main() {
  return foo();
}
`,
    "src/consumer.tsx": `
import { foo, Widget, Gadget, Greeter, Button } from "./lib";
import main from "./lib";

export function useAll(value: Widget, gadget: Gadget) {
  foo();
  const greeter = new Greeter();
  greeter.greet();
  const button = <Button variant="primary" disabled />;
  const arrow = () => foo();
  main();
  return { value, gadget, button, arrow };
}
`,
    "src/other-consumer.ts": `
import { foo as fooAlias } from "./lib";

export function caller() {
  return fooAlias();
}
`,
    "src/reexports.ts": `
export { foo, bar as renamedBar } from "./lib";
export * from "./lib";
export * as libNamespace from "./lib";
`,
  };

  const paths: Record<string, string> = {};
  for (const [rel, content] of Object.entries(files)) {
    const full = path.join(dir, rel);
    mkdirSync(path.dirname(full), { recursive: true });
    writeFileSync(full, content);
    paths[rel.split("/").pop()!.replace(/\.(tsx?)$/, "$1")] = full;
  }
  const project = new Project({
    compilerOptions: {
      target: ts.ScriptTarget.ESNext,
      module: ts.ModuleKind.CommonJS,
      moduleResolution: ts.ModuleResolutionKind.NodeJs,
      jsx: ts.JsxEmit.ReactJSX,
    },
  });
  for (const full of Object.values(paths)) {
    project.addSourceFileAtPath(full);
  }
  project.resolveSourceFileDependencies();

  return {
    project,
    dir,
    paths: {
      lib: paths["lib"],
      consumer: paths["consumer"],
      other: paths["other-consumer"],
      reexports: paths["reexports"],
    },
  };
}


