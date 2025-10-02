import { Project } from "ts-morph";
import { CliError } from "./cli-error";
import { gatherDiagnostics, loadProject, printDiagSummary, DiagnosticsSummary } from "./project";

export interface GlobalFlags {
  readonly allowPreexistingErrors: boolean;
  readonly acceptBreaking: boolean;
  readonly dryRun: boolean;
}

export type RefactorFn = (project: Project) => Promise<void> | void;

export interface RefactorRunSnapshot {
  readonly project: Project;
  readonly baseline: DiagnosticsSummary;
  readonly after: DiagnosticsSummary;
  readonly delta: number;
}

export async function runRefactor(flags: GlobalFlags, refactor: RefactorFn): Promise<RefactorRunSnapshot> {
  const project = await loadProject();

  const baseline = gatherDiagnostics(project);
  console.log(`[types] baseline errors: ${baseline.errors.length}`);
  if (baseline.errors.length > 0 && !flags.allowPreexistingErrors) {
    printDiagSummary(project, "baseline", baseline.diags);
    throw new CliError("Refactor refused: baseline has type errors. Pass --allow-preexisting-errors to proceed.", 2);
  }

  await refactor(project);

  const after = gatherDiagnostics(project);
  const delta = after.errors.length - baseline.errors.length;
  console.log(`[types] after errors: ${after.errors.length} (delta ${delta >= 0 ? "+" : ""}${delta})`);

  if (delta > 0 && !flags.acceptBreaking) {
    printDiagSummary(project, "after (breaking)", after.diags);
    throw new CliError("Refactor reverted: new type errors introduced. Pass --accept-breaking to force.", 3);
  }

  if (flags.dryRun) {
    console.log("Dry run: changes not saved.");
  } else {
    await project.save();
    console.log("Saved.");
  }

  return { project, baseline, after, delta };
}
