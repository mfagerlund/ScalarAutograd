import { Value, V, Vec3, lbfgs, nonlinearLeastSquares, CompiledResiduals } from 'scalar-autograd';
import { TriangleMesh } from '../mesh/TriangleMesh';
import { SubdividedMesh } from '../mesh/SubdividedMesh';
import { EnergyRegistry } from '../energy/EnergyRegistry';

// Import all energies to ensure they register
import '../energy/DevelopableEnergy';
import '../energy/CovarianceEnergy';
import '../energy/StochasticBimodalEnergy';
import '../energy/RidgeBasedEnergy';
import '../energy/AlignmentBimodalEnergy';
import '../energy/BoundingBoxEnergy';
import '../energy/CombinatorialEnergy';
import '../energy/ContiguousBimodalEnergy';
import '../energy/EigenProxyEnergy';
import '../energy/StochasticCovarianceEnergy';

export type CompilationMode = 'none' | 'eager' | 'lazy';

export interface OptimizationOptions {
  maxIterations?: number;
  gradientTolerance?: number;
  verbose?: boolean;
  captureInterval?: number; // Save mesh every N iterations
  onProgress?: (iteration: number, energy: number, history: TriangleMesh[]) => void;
  onCompileProgress?: (current: number, total: number, percent: number) => void;
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

export interface MultiResolutionOptions {
  startLevel: number; // Starting subdivision level (default: 1)
  targetLevel: number; // Target subdivision level (default: 2)
  iterationsPerLevel?: number; // Iterations at each level (default: 50)
  gradientTolerance?: number;
  verbose?: boolean;
  coarseEnergyType?: 'bimodal' | 'contiguous' | 'alignment' | 'boundingbox' | 'eigenproxy' | 'combinatorial' | 'covariance' | 'stochastic'; // Energy for coarse levels (default: combinatorial)
  fineEnergyType?: 'bimodal' | 'contiguous' | 'alignment' | 'boundingbox' | 'eigenproxy' | 'combinatorial' | 'covariance' | 'stochastic'; // Energy for fine levels (default: covariance)
  useCompiled?: boolean;
  onProgress?: (level: number, iteration: number, energy: number) => void;
}

export interface MultiResolutionResult extends OptimizationResult {
  subdivisionLevels: number[];
  energiesPerLevel: number[];
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

  optimize(options: OptimizationOptions = {}): OptimizationResult {
    const {
      maxIterations = 200,
      gradientTolerance = 1e-5,
      verbose = true,
      energyType = 'boundingbox', // Default to better energy function
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
      result = lbfgs(params, compiled, {
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
      functionEvaluations: result.functionEvaluations,
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
      energyType = 'boundingbox',
      useCompiled = true,
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
        return energyFunction.computeResiduals(this.mesh);
      }, 50); // Process 50 residuals at a time

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

      // Run optimization chunk
      const result = lbfgs(this.params, objectiveFn, {
        maxIterations: chunkIterations,
        gradientTolerance,
        verbose,
      });

      totalIterations += result.iterations;
      totalFunctionEvals += result.functionEvaluations;
      currentEnergy = result.finalCost;
      lastResult = result;

      // Always update progress numbers
      if (onProgress) {
        onProgress(totalIterations, currentEnergy, this.history);
      }

      // Update mesh and capture for visualization less frequently
      this.paramsToMesh(this.params);
      if (totalIterations % captureInterval === 0) {
        this.captureSnapshot();
      }

      if (verbose) {
        console.log(`Chunk complete: ${totalIterations}/${maxIterations}, energy=${currentEnergy.toExponential(3)}, reason: ${result.convergenceReason}`);
      }

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

      // Yield to browser
      await new Promise(resolve => setTimeout(resolve, 0));
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

  /**
   * Multi-resolution optimization following the SIGGRAPH 2018 paper strategy.
   *
   * Starts with a coarse mesh and progressively subdivides, optimizing at each level.
   * This prevents fragmentation by establishing large-scale structure first.
   *
   * Paper approach (Section 4.2):
   * - "minimize the energy on an initial coarse mesh to get the basic shape"
   * - "apply regular 4-1 subdivision to all triangles and continue minimizing"
   * - Use E^P (combinatorial) for coarse phase, E^λ (covariance) for fine phase
   *
   * @param baseMesh - The base mesh (typically subdivision level 0 or 1)
   * @param options - Multi-resolution optimization options
   */
  static async optimizeMultiResolution(
    baseMesh: SubdividedMesh,
    options: MultiResolutionOptions
  ): Promise<MultiResolutionResult> {
    const {
      startLevel = baseMesh.subdivisionLevel,
      targetLevel = startLevel + 1,
      iterationsPerLevel = 50,
      gradientTolerance = 1e-5,
      verbose = true,
      coarseEnergyType = 'combinatorial',
      fineEnergyType = 'covariance',
      useCompiled = false,
      onProgress,
    } = options;

    if (targetLevel < startLevel) {
      throw new Error('Target level must be >= start level');
    }

    const subdivisionLevels: number[] = [];
    const energiesPerLevel: number[] = [];
    let allHistory: TriangleMesh[] = [];
    let totalIterations = 0;
    let totalFunctionEvals = 0;

    // Start with the base mesh at the start level
    let currentMesh = baseMesh;

    // If we need to start at a different level, subdivide to get there
    while (currentMesh.subdivisionLevel < startLevel) {
      currentMesh = currentMesh.subdivide();
    }

    if (verbose) {
      console.log(`\n=== Multi-Resolution Optimization ===`);
      console.log(`Starting at level ${currentMesh.subdivisionLevel} (${currentMesh.mesh.vertices.length} vertices)`);
      console.log(`Target level: ${targetLevel}`);
      console.log(`Coarse energy: ${coarseEnergyType}, Fine energy: ${fineEnergyType}\n`);
    }

    // Optimize at each subdivision level
    for (let level = startLevel; level <= targetLevel; level++) {
      const isCoarsePhase = level < targetLevel;
      const energyType = isCoarsePhase ? coarseEnergyType : fineEnergyType;

      if (verbose) {
        console.log(`\n--- Level ${level} (${currentMesh.mesh.vertices.length} vertices, ${currentMesh.mesh.faces.length} faces) ---`);
        console.log(`Energy: ${energyType} (${isCoarsePhase ? 'coarse' : 'fine'} phase)`);
      }

      // Create optimizer for this level
      const optimizer = new DevelopableOptimizer(currentMesh.mesh);

      // Optimize at this level
      const result = await optimizer.optimizeAsync({
        maxIterations: iterationsPerLevel,
        gradientTolerance,
        verbose,
        energyType,
        useCompiled,
        chunkSize: 20,
        onProgress: onProgress ? (iter, energy) => onProgress(level, iter, energy) : undefined,
      });

      subdivisionLevels.push(level);
      energiesPerLevel.push(result.finalEnergy);
      totalIterations += result.iterations;
      totalFunctionEvals += result.functionEvaluations || 0;
      allHistory.push(...result.history);

      if (verbose) {
        console.log(`Level ${level} complete: ${result.iterations} iterations, energy=${result.finalEnergy.toExponential(3)}`);
        console.log(`Convergence: ${result.convergenceReason}`);
      }

      // Update the current mesh with optimized positions
      currentMesh.mesh = optimizer.mesh;

      // Subdivide for next level (if not at target)
      if (level < targetLevel) {
        if (verbose) {
          console.log(`Subdividing ${level} → ${level + 1}...`);
        }
        currentMesh = currentMesh.subdivide();
      }
    }

    if (verbose) {
      console.log(`\n=== Multi-Resolution Complete ===`);
      console.log(`Total iterations: ${totalIterations}`);
      console.log(`Total function evaluations: ${totalFunctionEvals}`);
      console.log(`Final level: ${currentMesh.subdivisionLevel}`);
      console.log(`Final vertices: ${currentMesh.mesh.vertices.length}`);
    }

    return {
      success: true,
      iterations: totalIterations,
      finalEnergy: energiesPerLevel[energiesPerLevel.length - 1],
      history: allHistory,
      convergenceReason: 'Multi-resolution optimization complete',
      functionEvaluations: totalFunctionEvals,
      subdivisionLevels,
      energiesPerLevel,
    };
  }
}
