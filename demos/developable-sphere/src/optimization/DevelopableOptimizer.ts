import { Value, V, Vec3, lbfgs, CompiledResiduals } from 'scalar-autograd';
import { TriangleMesh } from '../mesh/TriangleMesh';
import { DevelopableEnergy } from '../energy/DevelopableEnergy';
import { BoundingBoxEnergy } from '../energy/BoundingBoxEnergy';

export interface OptimizationOptions {
  maxIterations?: number;
  gradientTolerance?: number;
  verbose?: boolean;
  captureInterval?: number; // Save mesh every N iterations
  onProgress?: (iteration: number, energy: number, history: TriangleMesh[]) => void;
  chunkSize?: number; // Number of iterations per chunk (for async optimization)
  energyType?: 'variance' | 'boundingbox'; // Energy function to use
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
}

export class DevelopableOptimizer {
  private mesh: TriangleMesh;
  private history: TriangleMesh[] = [];
  private shouldStop = false;
  private compiled: ReturnType<typeof CompiledResiduals.compile> | null = null;
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
    } = options;

    // Convert mesh vertices to flat parameter array
    const params = this.meshToParams();

    // Choose energy function
    const EnergyClass = energyType === 'boundingbox' ? BoundingBoxEnergy : DevelopableEnergy;

    // Compile residuals for efficient gradient computation
    if (verbose) {
      console.log(`Using ${energyType} energy with compiled residuals...`);
    }

    const compiled = CompiledResiduals.compile(params, (p: Value[]) => {
      this.paramsToMesh(p);
      return EnergyClass.computeResiduals(this.mesh);
    });

    if (verbose) {
      console.log(`Compiled: ${compiled.kernelCount} kernels, ${compiled.kernelReuseFactor.toFixed(1)}x reuse`);
    }

    // Capture initial state
    this.captureSnapshot();

    // Run L-BFGS optimization
    const result = lbfgs(params, compiled, {
      maxIterations,
      gradientTolerance,
      verbose,
    });

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
    } = options;

    this.shouldStop = false;
    this.history = [];

    this.params = this.meshToParams();

    if (verbose) {
      console.log(`Using ${energyType} energy (non-compiled)...`);
    }

    let totalIterations = 0;
    let totalFunctionEvals = 0;
    let currentEnergy = DevelopableEnergy.compute(this.mesh).data;
    let lastResult: any = null;

    // Initial capture
    this.captureSnapshot();
    if (onProgress) {
      onProgress(0, currentEnergy, this.history);
    }

    // Non-compiled objective function
    const objectiveFn = (p: Value[]) => {
      this.paramsToMesh(p);
      return DevelopableEnergy.compute(this.mesh);
    };

    while (totalIterations < maxIterations && !this.shouldStop) {
      // Run a chunk of iterations
      const chunkIterations = Math.min(chunkSize, maxIterations - totalIterations);

      // Use non-compiled objective for this chunk
      const result = lbfgs(this.params, objectiveFn, {
        maxIterations: chunkIterations,
        gradientTolerance,
        verbose,
      });

      totalIterations += result.iterations;
      totalFunctionEvals += result.functionEvaluations;
      currentEnergy = result.finalCost;
      lastResult = result;

      // Update mesh with current params and capture for visualization
      this.paramsToMesh(this.params);
      if (totalIterations % captureInterval === 0) {
        this.captureSnapshot();
        if (onProgress) {
          onProgress(totalIterations, currentEnergy, this.history);
        }
      }

      if (verbose) {
        console.log(`Chunk complete: ${totalIterations}/${maxIterations}, energy=${currentEnergy.toExponential(3)}, reason: ${result.convergenceReason}`);
      }

      // Check if optimization converged or failed
      if (result.convergenceReason !== "Maximum iterations reached") {
        this.paramsToMesh(this.params);
        this.captureSnapshot();

        // Compute final gradient norm using compiled gradients
        const { J } = this.compiled!.evaluate(this.params);
        const gradientNorm = Math.sqrt(J[0].reduce((sum, g) => sum + g * g, 0));

        return {
          success: result.success,
          iterations: totalIterations,
          finalEnergy: currentEnergy,
          history: this.history,
          convergenceReason: result.convergenceReason,
          gradientNorm,
          functionEvaluations: totalFunctionEvals,
          kernelCount: this.compiled!.kernelCount,
          kernelReuseFactor: this.compiled!.kernelReuseFactor,
        };
      }

      // Yield to browser
      await new Promise(resolve => setTimeout(resolve, 0));
    }

    this.paramsToMesh(this.params);
    this.captureSnapshot();

    // Compute final gradient norm using compiled gradients
    const { J } = this.compiled!.evaluate(this.params);
    const gradientNorm = Math.sqrt(J[0].reduce((sum, g) => sum + g * g, 0));

    const convergenceReason = this.shouldStop
      ? "Stopped by user"
      : lastResult?.convergenceReason || "Maximum iterations reached";

    return {
      success: !this.shouldStop,
      iterations: totalIterations,
      finalEnergy: currentEnergy,
      history: this.history,
      convergenceReason,
      gradientNorm,
      functionEvaluations: totalFunctionEvals,
      kernelCount: this.compiled!.kernelCount,
      kernelReuseFactor: this.compiled!.kernelReuseFactor,
    };
  }
}
