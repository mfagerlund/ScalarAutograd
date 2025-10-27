#!/usr/bin/env node

/**
 * CLI tool for symbolic gradient generation.
 * Accepts mathematical expressions and outputs gradient computation code.
 */

import * as fs from 'fs';
import * as path from 'path';
import { parse } from '../symbolic/Parser';
import { computeGradients } from '../symbolic/SymbolicDiff';
import { simplify } from '../symbolic/Simplify';
import { generateGradientCode, generateGradientFunction } from '../symbolic/CodeGen';

interface CliOptions {
  input?: string;
  output?: string;
  wrt?: string[];
  format: 'js' | 'ts' | 'function' | 'inline';
  simplify: boolean;
  functionName?: string;
  help: boolean;
  version: boolean;
}

const VERSION = '1.0.0';

function printHelp(): void {
  console.log(`
ScalarAutograd Symbolic Gradient Generator v${VERSION}

Generate symbolic gradient formulas from mathematical expressions.

USAGE:
  npx scalar-grad [options] [expression]

OPTIONS:
  -i, --input <file>       Input file containing expressions
  -o, --output <file>      Output file (default: stdout)
  --wrt <params>           Comma-separated list of parameters to differentiate
                           (default: auto-detect from expressions)
  --format <type>          Output format: js, ts, function, inline (default: js)
  --no-simplify            Disable expression simplification
  --function <name>        Function name when using --format=function (default: computeGradient)
  -h, --help               Show this help message
  -v, --version            Show version

EXAMPLES:
  # Simple expression
  npx scalar-grad "x = 2; y = 3; output = x*x + y*y" --wrt x,y

  # From file
  npx scalar-grad --input forward.txt --wrt a,b,c --output gradients.js

  # Generate function
  npx scalar-grad "z = x*x + y*y; output = sqrt(z)" --format function --function distanceGradient

  # Inline expression (no file)
  echo "output = sin(x) * cos(y)" | npx scalar-grad --wrt x,y

INPUT FORMAT:
  Expressions use operator overloading syntax:
    c = a + b          // Addition
    d = c * 2          // Multiplication
    e = sin(d)         // Function call
    output = e         // Mark output variable

  Supported operators: +, -, *, /, ** (power)
  Supported functions: sin, cos, tan, exp, log, sqrt, abs, asin, acos, atan, etc.

  Vec2/Vec3 support:
    v = Vec2(x, y)     // Vector constructor
    mag = v.magnitude  // Vector property
    dot = u.dot(v)     // Vector method

OUTPUT:
  Generated code includes:
  - Forward pass computation
  - Gradient formulas with mathematical notation in comments
  - Executable JavaScript/TypeScript code
`);
}

function parseArgs(args: string[]): CliOptions {
  const options: CliOptions = {
    format: 'js',
    simplify: true,
    help: false,
    version: false
  };

  let i = 0;
  while (i < args.length) {
    const arg = args[i];

    if (arg === '-h' || arg === '--help') {
      options.help = true;
      i++;
    } else if (arg === '-v' || arg === '--version') {
      options.version = true;
      i++;
    } else if (arg === '-i' || arg === '--input') {
      options.input = args[++i];
      i++;
    } else if (arg === '-o' || arg === '--output') {
      options.output = args[++i];
      i++;
    } else if (arg === '--wrt') {
      options.wrt = args[++i].split(',').map(s => s.trim());
      i++;
    } else if (arg === '--format') {
      const format = args[++i];
      if (format !== 'js' && format !== 'ts' && format !== 'function' && format !== 'inline') {
        throw new Error(`Invalid format: ${format}. Must be js, ts, function, or inline`);
      }
      options.format = format;
      i++;
    } else if (arg === '--no-simplify') {
      options.simplify = false;
      i++;
    } else if (arg === '--function') {
      options.functionName = args[++i];
      i++;
    } else if (arg.startsWith('-')) {
      throw new Error(`Unknown option: ${arg}`);
    } else {
      // Positional argument - treat as expression
      options.input = arg;
      i++;
    }
  }

  return options;
}

function readInput(options: CliOptions): string {
  if (options.input) {
    // Check if it's a file
    if (fs.existsSync(options.input)) {
      return fs.readFileSync(options.input, 'utf-8');
    } else {
      // Treat as direct expression
      return options.input;
    }
  }

  // Read from stdin
  const stdin = fs.readFileSync(0, 'utf-8');
  return stdin;
}

function writeOutput(content: string, options: CliOptions): void {
  if (options.output) {
    fs.writeFileSync(options.output, content, 'utf-8');
    console.error(`Output written to ${options.output}`);
  } else {
    console.log(content);
  }
}

function extractParameters(input: string): string[] {
  // Simple heuristic: find all single-letter variables that appear on the right-hand side
  // but are never assigned
  const assignedVars = new Set<string>();
  const usedVars = new Set<string>();

  // Match assignments: var = expr
  const assignmentRegex = /(\w+)\s*=/g;
  let match;
  while ((match = assignmentRegex.exec(input)) !== null) {
    assignedVars.add(match[1]);
  }

  // Match all identifiers
  const identifierRegex = /\b([a-zA-Z_]\w*)\b/g;
  while ((match = identifierRegex.exec(input)) !== null) {
    const ident = match[1];
    // Skip known functions and keywords
    const keywords = ['Vec2', 'Vec3', 'sin', 'cos', 'tan', 'exp', 'log', 'sqrt', 'abs',
                      'min', 'max', 'output', 'Math', 'const', 'let', 'var'];
    if (!keywords.includes(ident)) {
      usedVars.add(ident);
    }
  }

  // Parameters are variables used but never assigned
  const params: string[] = [];
  for (const v of usedVars) {
    if (!assignedVars.has(v)) {
      params.push(v);
    }
  }

  return params.sort();
}

function main(): void {
  try {
    const args = process.argv.slice(2);

    if (args.length === 0) {
      printHelp();
      process.exit(0);
    }

    const options = parseArgs(args);

    if (options.help) {
      printHelp();
      process.exit(0);
    }

    if (options.version) {
      console.log(`ScalarAutograd Symbolic Gradient Generator v${VERSION}`);
      process.exit(0);
    }

    // Read input
    const input = readInput(options);

    if (!input.trim()) {
      console.error('Error: No input provided');
      process.exit(1);
    }

    // Parse expression
    console.error('Parsing expressions...');
    const program = parse(input);

    // Auto-detect parameters if not specified
    let parameters = options.wrt;
    if (!parameters) {
      parameters = extractParameters(input);
      console.error(`Auto-detected parameters: ${parameters.join(', ')}`);
    }

    if (parameters.length === 0) {
      console.error('Error: No parameters to differentiate with respect to');
      console.error('Hint: Use --wrt to specify parameters');
      process.exit(1);
    }

    // Compute gradients
    console.error('Computing symbolic gradients...');
    let gradients = computeGradients(program, parameters);

    // Simplify if enabled
    if (options.simplify) {
      console.error('Simplifying expressions...');
      const simplified = new Map<string, any>();
      for (const [param, gradExpr] of gradients.entries()) {
        simplified.set(param, simplify(gradExpr));
      }
      gradients = simplified;
    }

    // Generate code
    console.error('Generating code...');
    let output: string;

    if (options.format === 'function') {
      const funcName = options.functionName || 'computeGradient';
      output = generateGradientFunction(program, gradients, funcName, parameters, {
        includeMath: true
      });
    } else {
      output = generateGradientCode(program, gradients, {
        includeMath: true,
        varStyle: 'const',
        includeForward: true
      });
    }

    // Add TypeScript annotations if requested
    if (options.format === 'ts') {
      output = `// TypeScript version\n${output}`;
      // Could add type annotations here
    }

    // Write output
    writeOutput(output, options);

    console.error('✓ Success!');

  } catch (error) {
    console.error('Error:', (error as Error).message);
    if (process.env.DEBUG) {
      console.error((error as Error).stack);
    }
    process.exit(1);
  }
}

// Run if called directly
if (require.main === module) {
  main();
}

export { main, parseArgs, extractParameters };
