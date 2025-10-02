import { Project, SourceFile, Diagnostic, ts } from "ts-morph";
import path from "node:path";
import { CliError } from "./cli-error";

export async function loadProject(): Promise<Project> {
  const project = new Project({ tsConfigFilePath: "tsconfig.json" });
  if (project.getSourceFiles().length === 0) {
    project.addSourceFilesAtPaths(["src/**/*.ts", "src/**/*.tsx", "src/**/*.js", "src/**/*.jsx"]);
  }
  return project;
}

export function requireSourceFile(project: Project, filePath: string): SourceFile {
  const sf = project.getSourceFile(filePath);
  if (!sf) {
    throw new CliError(`File not found in project: ${filePath}`);
  }
  return sf;
}

export interface DiagnosticsSummary {
  readonly diags: Diagnostic[];
  readonly errors: Diagnostic[];
  readonly warnings: Diagnostic[];
}

export function gatherDiagnostics(project: Project): DiagnosticsSummary {
  const diags = project.getPreEmitDiagnostics();
  const errors = diags.filter(d => d.getCategory() === ts.DiagnosticCategory.Error);
  const warnings = diags.filter(d => d.getCategory() === ts.DiagnosticCategory.Warning);
  return { diags, errors, warnings };
}

export function printDiagSummary(project: Project, label: string, diags: Diagnostic[]): void {
  const text = project.formatDiagnosticsWithColorAndContext(diags);
  console.log(`\n[types] ${label}: ${diags.length} diagnostics`);
  if (diags.length) {
    console.log(text);
  }
}

export function normalizeModuleSpecifier(fromPath: string, toPath: string): string {
  let rel = path.relative(path.dirname(fromPath), toPath).replace(/\\/g, "/");
  rel = rel.replace(/\.(ts|tsx|js|jsx)$/, "");
  if (!rel.startsWith(".")) rel = "./" + rel;
  return rel;
}
