#!/usr/bin/env tsx
/**
 * ts-refactor-cli - modular CLI for TypeScript/React refactors agents can trigger safely.
 *
 * Built with Yargs + Zod for strict, typed subcommands and useful help output.
 * Safety gates run baseline/after TypeScript diagnostics so automated agents avoid breaking changes.
 */

import yargs, { ArgumentsCamelCase } from "yargs";
import { hideBin } from "yargs/helpers";
import { z } from "zod";
import { CliError } from "./ts-refactor/cli-error";
import { runRefactor, GlobalFlags } from "./ts-refactor/runner";
import { rename, renameSchema } from "./ts-refactor/commands/rename";
import { renameByName, renameByNameSchema } from "./ts-refactor/commands/rename-by-name";
import { moveFile, moveSchema } from "./ts-refactor/commands/move";
import { jsxPropRenameSchema, jsxPropMigrateSchema, renameJsxProps } from "./ts-refactor/commands/jsx-props";
import { exportStyle, exportStyleSchema } from "./ts-refactor/commands/export-style";
import { organizeImports, organizeImportsSchema } from "./ts-refactor/commands/organize-imports";
import { react17Cleanup, react17CleanupSchema } from "./ts-refactor/commands/react17-cleanup";
import { wrapWithMemo, wrapWithMemoSchema } from "./ts-refactor/commands/wrap-with-memo";

const globalFlagsSchema = z.object({
  allowPreexistingErrors: z.boolean().optional(),
  acceptBreaking: z.boolean().optional(),
  dryRun: z.boolean().optional(),
});

type Argv = ArgumentsCamelCase<Record<string, unknown>>;
const terminalWidth = typeof process.stdout.columns === "number" ? process.stdout.columns : 120;

yargs(hideBin(process.argv))
  .scriptName("ts-refactor-cli")
  .usage("$0 <command> [options]")
  .strict()
  .demandCommand(1, "Specify a refactor command.")
  .options({
    "allow-preexisting-errors": {
      type: "boolean",
      describe: "Proceed even if baseline TypeScript diagnostics contain errors.",
      default: false,
    },
    "accept-breaking": {
      type: "boolean",
      describe: "Persist changes even if new errors are introduced.",
      default: false,
    },
    "dry-run": {
      type: "boolean",
      describe: "Do not write files; useful for validation runs.",
      default: false,
    },
  })
  .group([
    "allow-preexisting-errors",
    "accept-breaking",
    "dry-run",
  ], "Safety flags:")
  .command(
    "rename",
    "Rename the identifier found at a file/line/column position.",
    cmd => cmd
      .option("file", { type: "string", demandOption: true, describe: "File path relative to repo root." })
      .option("line", { type: "number", demandOption: true, describe: "1-based line number." })
      .option("col", { type: "number", demandOption: true, describe: "1-based column number." })
      .option("to", { type: "string", demandOption: true, describe: "New identifier name." }),
    async argv => {
      const flags = parseFlags(argv);
      const options = renameSchema.parse(argv);
      await runRefactor(flags, project => rename(project, options));
    }
  )
  .command(
    "rename-by-name",
    "Rename declarations across the project by name (optionally scoped by kind).",
    cmd => cmd
      .option("name", { type: "string", demandOption: true, describe: "Existing declaration name." })
      .option("to", { type: "string", demandOption: true, describe: "Replacement name." })
      .option("kind", {
        type: "string",
        choices: ["class", "function", "variable", "method"],
        describe: "Limit rename to a specific declaration kind.",
      }),
    async argv => {
      const flags = parseFlags(argv);
      const options = renameByNameSchema.parse(argv);
      await runRefactor(flags, project => renameByName(project, options));
    }
  )
  .command(
    "move",
    "Move a file and update import specifiers that reference it.",
    cmd => cmd
      .option("from", { type: "string", demandOption: true, describe: "Current path of the file." })
      .option("to", { type: "string", demandOption: true, describe: "New path for the file." }),
    async argv => {
      const flags = parseFlags(argv);
      const options = moveSchema.parse(argv);
      await runRefactor(flags, project => moveFile(project, options));
    }
  )
  .command(
    "jsx-prop-rename",
    "Rename a prop on targeted JSX elements (with optional default insertion).",
    cmd => cmd
      .option("from", { type: "string", demandOption: true, describe: "Existing prop name." })
      .option("to", { type: "string", demandOption: true, describe: "New prop name." })
      .option("tag", { type: "string", describe: "Limit to a specific JSX tag within the file." })
      .option("module", { type: "string", describe: "Module specifier where the component is imported from." })
      .option("import", { type: "string", describe: "Component export name to match when module is provided." })
      .option("add-default", { type: "string", describe: "Add prop=defaultValue when missing." })
      .option("map", { type: "string", describe: "Value mapping in form old->new;false->ghost." }),
    async argv => {
      const flags = parseFlags(argv);
      const options = jsxPropRenameSchema.parse(argv);
      await runRefactor(flags, project => {
        const changed = renameJsxProps(project, options);
        console.log(`[jsx] changed attributes: ${changed}`);
      });
    }
  )
  .command(
    "jsx-prop-migrate",
    "Rename a JSX prop and migrate literal/boolean values via a mapping.",
    cmd => cmd
      .option("from", { type: "string", demandOption: true, describe: "Existing prop name." })
      .option("to", { type: "string", demandOption: true, describe: "New prop name." })
      .option("tag", { type: "string", describe: "Limit to a specific JSX tag within the file." })
      .option("module", { type: "string", describe: "Module specifier where the component is imported from." })
      .option("import", { type: "string", describe: "Component export name to match when module is provided." })
      .option("add-default", { type: "string", describe: "Add prop=defaultValue when missing." })
      .option("map", { type: "string", demandOption: true, describe: "Value mapping in form old->new;false->ghost." }),
    async argv => {
      const flags = parseFlags(argv);
      const options = jsxPropMigrateSchema.parse(argv);
      await runRefactor(flags, project => {
        const changed = renameJsxProps(project, options);
        console.log(`[jsx] changed attributes: ${changed}`);
      });
    }
  )
  .command(
    "export-style",
    "Convert between default and named exports (updates imports project-wide).",
    cmd => cmd
      .option("file", { type: "string", demandOption: true, describe: "Path to the module with the export." })
      .option("to", { type: "string", demandOption: true, choices: ["named", "default"], describe: "Target export style." })
      .option("name", { type: "string", describe: "Explicit export name when required." }),
    async argv => {
      const flags = parseFlags(argv);
      const options = exportStyleSchema.parse(argv);
      await runRefactor(flags, project => exportStyle(project, options));
    }
  )
  .command(
    "organize-imports",
    "Run TypeScript organizeImports across the project.",
    cmd => cmd,
    async argv => {
      const flags = parseFlags(argv);
      organizeImportsSchema.parse(argv);
      await runRefactor(flags, project => organizeImports(project));
    }
  )
  .command(
    "react17-cleanup",
    "Remove unused default React imports left from pre-React-17 patterns.",
    cmd => cmd,
    async argv => {
      const flags = parseFlags(argv);
      react17CleanupSchema.parse(argv);
      await runRefactor(flags, project => {
        const removed = react17Cleanup(project);
        console.log(`[react17] removed default React import in ${removed} files`);
      });
    }
  )
  .command(
    "wrap-with-memo",
    "Wrap a component export with React.memo (adds memo import as needed).",
    cmd => cmd
      .option("file", { type: "string", demandOption: true, describe: "File that exports the component." })
      .option("export", { type: "string", demandOption: true, describe: "Named export or 'default'." }),
    async argv => {
      const flags = parseFlags(argv);
      const options = wrapWithMemoSchema.parse(argv);
      await runRefactor(flags, project => wrapWithMemo(project, options));
    }
  )
  .fail((msg, err, instance) => {
    const error = err as Error | undefined;
    if (error instanceof CliError) {
      console.error(error.message);
      process.exit(error.exitCode);
    }
    if (msg) {
      console.error(msg);
    }
    if (error && !(error instanceof CliError)) {
      console.error(error.message);
    }
    console.log(instance.help());
    process.exit(1);
  })
  .help()
  .epilogue("Designed for automated agents to run safe TypeScript refactors with guardrails.")
  .wrap(Math.min(terminalWidth, 120))
  .parse();

function parseFlags(argv: Argv): GlobalFlags {
  const parsed = globalFlagsSchema.parse(argv);
  return {
    allowPreexistingErrors: parsed.allowPreexistingErrors ?? false,
    acceptBreaking: parsed.acceptBreaking ?? false,
    dryRun: parsed.dryRun ?? false,
  };
}
