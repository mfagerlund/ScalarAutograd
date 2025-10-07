import { Value, V, Vec3, lbfgs, nonlinearLeastSquares, CompiledResiduals } from 'scalar-autograd';
import { TriangleMesh } from '../mesh/TriangleMesh';
import { EnergyRegistry } from '../energy/utils/EnergyRegistry';

// Import all energies to ensure they register
import '../energy/DevelopableEnergy';
import '../energy/PaperCovarianceEnergyELambda';
import '../energy/RidgeBasedEnergy';
import '../energy/AlignmentBimodalEnergy';
import '../energy/PaperPartitionEnergyEP';
import '../energy/PaperPartitionEnergyEPStochastic';
import '../energy/EigenProxyEnergy';
import '../energy/FastCovarianceEnergy';

export type CompilationMode = 'none' | 'eager' | 'lazy';

export interface OptimizationOptions {
  maxIterations?: number;
  gradientTolerance?: number;
  verbose?: boolean;
  captureInterval?: number; // Save mesh every N iterations
  onProgress?: (iteration: number, energy: number, history: TriangleMesh[]) => void;
  onCompileProgress?: (current: number, total: number, percent: number) => void;
  onCompileComplete?: () => void; // Called when compilation finishes
  chunkSize?: number; // Number of iterations per chunk (for async optimization)
  energyType?: string; // Energy function name (from registry)
  useCompiled?: boolean; // Use compiled gradients (default: true)
  compilationMode?: CompilationMode; // 'none' | 'eager' | 'lazy' (default: 'eager')
  optimizer?: 'lbfgs' | 'leastsquares'; // Optimizer selection (default: lbfgs)
}

export interface OptimizationResult {
  success: boolean;
  iterations: number;
  finalEnergy: number;
  history: TriangleMesh[];
  convergenceReason: string;
  gradientNorm?: number;
  functionEvaluations?: number;
  kernelCount?: number;
  kernelReuseFactor?: number;
  compilationTime?: number; // Time spent compiling in seconds
  numFunctions?: number; // Total number of functions compiled
}


export class DevelopableOptimizer {
  private mesh: TriangleMesh;
  private history: TriangleMesh[] = [];
  private shouldStop = false;
  private params: Value[] = [];

  constructor(mesh: TriangleMesh) {
    this.mesh = mesh;
  }

  stop(): void {
    this.shouldStop = true;
  }

  async optimize(options: OptimizationOptions = {}): Promise<OptimizationResult> {
    const {
      maxIterations = 200,
      gradientTolerance = 1e-5,
      verbose = true,
      energyType = EnergyRegistry.getNames()[0] || 'bimodal',
      optimizer = 'lbfgs', // Default optimizer
    } = options;

    // Convert mesh vertices to flat parameter array
    const params = this.meshToParams();

    // Get energy function from registry
    const energyFunction = EnergyRegistry.get(energyType || '');
    if (!energyFunction) {
      throw new Error(`Unknown energy type: ${energyType}. Available: ${EnergyRegistry.getNames().join(', ')}`);
    }

    // Compile residuals for efficient gradient computation
    if (verbose) {
      console.log(`Using ${energyType} energy with ${optimizer} optimizer and compiled residuals...`);
    }

    const compiled = CompiledResiduals.compile(params, (p: Value[]) => {
      this.paramsToMesh(p);
      return energyFunction.computeResiduals(this.mesh);
    });

    if (verbose) {
      console.log(`Compiled: ${compiled.kernelCount} kernels, ${compiled.kernelReuseFactor.toFixed(1)}x reuse`);
    }

    // IMPORTANT: Restore mesh to initial state after compilation
    // (compilation modifies the mesh during graph building)
    this.paramsToMesh(params);

    // Capture initial state
    this.captureSnapshot();

    // Run optimization based on selected optimizer
    let result;
    if (optimizer === 'leastsquares') {
      result = nonlinearLeastSquares(params, compiled, {
        maxIterations,
        gradientTolerance,
        verbose,
      });
    } else {
      result = await lbfgs(params, compiled, {
        maxIterations,
        gradientTolerance,
        verbose,
      });
    }

    // Update mesh with final parameters
    this.paramsToMesh(params);

    // Capture final state
    this.captureSnapshot();

    return {
      success: result.success,
      iterations: result.iterations,
      finalEnergy: result.finalCost,
      history: this.history,
      convergenceReason: result.convergenceReason,
      functionEvaluations: 'functionEvaluations' in result ? (result as any).functionEvaluations : 0,
      kernelCount: compiled.kernelCount,
      kernelReuseFactor: compiled.kernelReuseFactor,
    };
  }

  private meshToParams(): Value[] {
    const params: Value[] = [];
    for (const v of this.mesh.vertices) {
      params.push(V.W(v.x.data), V.W(v.y.data), V.W(v.z.data));
    }
    return params;
  }

  private paramsToMesh(params: Value[]): void {
    const numVertices = this.mesh.vertices.length;
    for (let i = 0; i < numVertices; i++) {
      const x = params[3 * i];
      const y = params[3 * i + 1];
      const z = params[3 * i + 2];
      this.mesh.setVertexPosition(i, new Vec3(x, y, z));
    }
  }

  private captureSnapshot(): void {
    this.history.push(this.mesh.clone());
  }

  /**
   * Async optimization that yields to the browser between chunks
   */
  async optimizeAsync(options: OptimizationOptions = {}): Promise<OptimizationResult> {
    const {
      maxIterations = 200,
      gradientTolerance = 1e-5,
      verbose = true,
      captureInterval = 5,
      chunkSize = 5,
      onProgress,
      energyType = EnergyRegistry.getNames()[0] || 'bimodal',
      useCompiled = true,
      optimizer = 'lbfgs',
    } = options;

    this.shouldStop = false;
    this.history = [];

    this.params = this.meshToParams();

    // Get energy function from registry
    const energyFunction = EnergyRegistry.get(energyType || '');
    if (!energyFunction) {
      throw new Error(`Unknown energy type: ${energyType}. Available: ${EnergyRegistry.getNames().join(', ')}`);
    }

    if (verbose) {
      console.log(`Using ${energyType} energy (${useCompiled ? 'compiled' : 'non-compiled'})...`);
      console.log(`Optimizer: ${optimizer}, chunk size: ${chunkSize}`);
    }

    // Compile if enabled
    let compiled: any = null;
    let kernelCount = 0;
    let kernelReuseFactor = 0;
    let compilationTime = 0;
    let numFunctions = 0;

    if (useCompiled) {
      const compileStart = Date.now();

      // Use async compilation to avoid freezing the browser
      compiled = await CompiledResiduals.compileAsync(this.params, (p: Value[]) => {
        this.paramsToMesh(p);
        const residuals = energyFunction.computeResiduals(this.mesh);
        return residuals;
      }, 50); // Process 50 residuals at a time, with progress callback , options.onCompileProgressyou're 
        
      compilationTime = (Date.now() - compileStart) / 1000;
      kernelCount = compiled.kernelCount;
      kernelReuseFactor = compiled.kernelReuseFactor;
      numFunctions = compiled.numFunctions;

      if (verbose) {
        console.log(`Compiled: ${numFunctions} functions → ${kernelCount} kernels, ${kernelReuseFactor.toFixed(1)}x reuse in ${compilationTime.toFixed(2)}s`);
      }

      // IMPORTANT: Restore mesh to initial state after compilation
      // (compilation modifies the mesh during graph building)
      this.paramsToMesh(this.params);

      // Signal compilation complete
      if (options.onCompileComplete) {
        options.onCompileComplete();
      }

      // Yield to browser to update UI before starting optimization
      await new Promise(resolve => setTimeout(resolve, 0));
    }

    let totalIterations = 0;
    let totalFunctionEvals = 0;
    let currentEnergy = energyFunction.compute(this.mesh).data;
    let lastResult: any = null;

    // Initial capture
    this.captureSnapshot();
    if (onProgress) {
      onProgress(0, currentEnergy, this.history);
    }

    // Objective function (compiled or non-compiled)
    const objectiveFn = compiled || ((p: Value[]) => {
      this.paramsToMesh(p);
      return energyFunction.compute(this.mesh);
    });

    while (totalIterations < maxIterations && !this.shouldStop) {
      // Run a chunk of iterations
      const chunkIterations = Math.min(chunkSize, maxIterations - totalIterations);

      // Run optimization chunk based on selected optimizer
      const result = optimizer === 'leastsquares'
        ? nonlinearLeastSquares(this.params, objectiveFn, {
            maxIterations: chunkIterations,
            gradientTolerance,
            verbose,
          })
        : await lbfgs(this.params, objectiveFn, {
            maxIterations: chunkIterations,
            gradientTolerance,
            verbose,
          });

      totalIterations += result.iterations;
      totalFunctionEvals += ('functionEvaluations' in result ? (result as any).functionEvaluations : 0) ?? 0;
      currentEnergy = result.finalCost;
      lastResult = result;

      // Update mesh and capture for visualization less frequently
      this.paramsToMesh(this.params);
      if (totalIterations % captureInterval === 0) {
        this.captureSnapshot();
      }

      if (verbose) {
        console.log(`Chunk complete: ${totalIterations}/${maxIterations}, energy=${currentEnergy.toExponential(3)}, reason: ${result.convergenceReason}`);
      }

      // Always update progress numbers
      if (onProgress) {
        onProgress(totalIterations, currentEnergy, this.history);
      }

      // YIELD TO BROWSER IMMEDIATELY AFTER PROGRESS UPDATE
      await new Promise(resolve => setTimeout(resolve, 0));

      // Check if optimization converged or failed
      if (result.convergenceReason !== "Maximum iterations reached") {
        this.paramsToMesh(this.params);
        this.captureSnapshot();

        return {
          success: result.success,
          iterations: totalIterations,
          finalEnergy: currentEnergy,
          history: this.history,
          convergenceReason: result.convergenceReason,
          gradientNorm: undefined,
          functionEvaluations: totalFunctionEvals,
          kernelCount,
          kernelReuseFactor,
          compilationTime,
          numFunctions,
        };
      }
    }

    this.paramsToMesh(this.params);
    this.captureSnapshot();

    const convergenceReason = this.shouldStop
      ? "Stopped by user"
      : lastResult?.convergenceReason || "Maximum iterations reached";

    return {
      success: !this.shouldStop,
      iterations: totalIterations,
      finalEnergy: currentEnergy,
      history: this.history,
      convergenceReason,
      gradientNorm: undefined,
      functionEvaluations: totalFunctionEvals,
      kernelCount,
      kernelReuseFactor,
      compilationTime,
      numFunctions,
    };
  }

}
