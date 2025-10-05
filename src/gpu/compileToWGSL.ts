/**
 * Compile Value computation graphs to WebGPU WGSL shaders
 *
 * This module translates the JavaScript code generation from compileIndirectKernel
 * into WGSL (WebGPU Shading Language) for GPU execution.
 */

import { Value } from '../Value';

/**
 * Converts JavaScript Math operations to WGSL equivalents
 */
function jsToWGSL(jsCode: string): string {
  return jsCode
    // Math functions - WGSL has them as builtins without "Math."
    .replace(/Math\.sin/g, 'sin')
    .replace(/Math\.cos/g, 'cos')
    .replace(/Math\.tan/g, 'tan')
    .replace(/Math\.asin/g, 'asin')
    .replace(/Math\.acos/g, 'acos')
    .replace(/Math\.atan/g, 'atan')
    .replace(/Math\.exp/g, 'exp')
    .replace(/Math\.log/g, 'log')
    .replace(/Math\.sqrt/g, 'sqrt')
    .replace(/Math\.abs/g, 'abs')
    .replace(/Math\.floor/g, 'floor')
    .replace(/Math\.ceil/g, 'ceil')
    .replace(/Math\.round/g, 'round')
    .replace(/Math\.sign/g, 'sign')
    .replace(/Math\.min/g, 'min')
    .replace(/Math\.max/g, 'max')
    .replace(/Math\.pow/g, 'pow')
    // WGSL uses tanh, sinh, cosh directly
    .replace(/Math\.tanh/g, 'tanh');
}

/**
 * Compile a Value computation graph to WGSL compute shader (forward pass only)
 *
 * @param output - The output Value node
 * @returns WGSL shader source code
 */
export function compileToWGSL(output: Value): {
  wgslCode: string;
  graphInputs: Value[];
  nodeToVar: Map<Value, string>;
} {
  const visited = new Set<Value>();
  const topoOrder: Value[] = [];
  const nodeToVar = new Map<Value, string>();
  let varCounter = 0;

  // Build topological order (same as CPU compiler)
  function buildTopoOrder(node: Value) {
    if (visited.has(node)) return;
    visited.add(node);

    for (const child of (node as any).prev as Value[]) {
      buildTopoOrder(child);
    }

    topoOrder.push(node);
  }

  buildTopoOrder(output);

  // Collect graph inputs (leaf nodes that require gradients - excludes constants)
  const graphInputs: Value[] = [];
  for (const node of topoOrder) {
    const prev = (node as any).prev as Value[];
    // Only include leaf nodes that can receive gradients (not constants)
    if (prev.length === 0 && node.requiresGrad && !graphInputs.includes(node)) {
      graphInputs.push(node);
    }
  }

  // Assign variable names
  function getVarName(node: Value): string {
    if (!nodeToVar.has(node)) {
      nodeToVar.set(node, `v${varCounter++}`);
    }
    return nodeToVar.get(node)!;
  }

  // Generate WGSL variable declarations and computations
  const declarations: string[] = [];

  for (const node of topoOrder) {
    const prev = (node as any).prev as Value[];
    const varName = getVarName(node);

    if (prev.length === 0) {
      // Leaf node
      if (node.requiresGrad) {
        // Input parameter - load from buffer
        const inputIdx = graphInputs.indexOf(node);
        declarations.push(`  let ${varName} = inputs[thread_idx * ${graphInputs.length} + ${inputIdx}];`);
      } else {
        // Constant - inline as f32 literal
        const value = node.data;
        const wgslLiteral = Number.isInteger(value) ? `${value}.0` : String(value);
        declarations.push(`  let ${varName} = ${wgslLiteral};`);
      }
    } else {
      // Computed node - get JavaScript code and convert to WGSL
      const childCodes = prev.map(c => getVarName(c));
      const jsCode = node.getForwardCode(childCodes);
      const wgslCode = jsToWGSL(jsCode);
      declarations.push(`  let ${varName} = ${wgslCode};`);
    }
  }

  const outputVar = getVarName(output);

  // Generate complete WGSL shader
  const wgslCode = `
// Auto-generated WGSL compute shader
// Inputs: ${graphInputs.length} values per thread
// Output: 1 value per thread

@group(0) @binding(0) var<storage, read> inputs: array<f32>;
@group(0) @binding(1) var<storage, read_write> outputs: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let thread_idx = global_id.x;

  // Check bounds
  if (thread_idx >= arrayLength(&outputs)) {
    return;
  }

${declarations.join('\n')}

  // Write output
  outputs[thread_idx] = ${outputVar};
}
`;

  return {
    wgslCode,
    graphInputs,
    nodeToVar
  };
}

/**
 * Wrapper for executing a compiled WGSL kernel on GPU
 */
export class WGSLKernel {
  private device: GPUDevice;
  private pipeline: GPUComputePipeline;
  private graphInputs: Value[];
  private inputsPerThread: number;

  constructor(
    device: GPUDevice,
    wgslCode: string,
    graphInputs: Value[]
  ) {
    this.device = device;
    this.graphInputs = graphInputs;
    this.inputsPerThread = graphInputs.length;

    // Create shader module
    const shaderModule = device.createShaderModule({
      code: wgslCode
    });

    // Create compute pipeline
    this.pipeline = device.createComputePipeline({
      layout: 'auto',
      compute: {
        module: shaderModule,
        entryPoint: 'main'
      }
    });
  }

  /**
   * Execute kernel on a batch of inputs
   *
   * @param batchInputs - Flat array: [input0_val0, input0_val1, ..., input1_val0, input1_val1, ...]
   * @param batchSize - Number of instances to compute
   * @returns Array of output values
   */
  async execute(batchInputs: Float32Array, batchSize: number): Promise<Float32Array> {
    // Create input buffer
    const inputBuffer = this.device.createBuffer({
      size: batchInputs.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      mappedAtCreation: true
    });
    new Float32Array(inputBuffer.getMappedRange()).set(batchInputs);
    inputBuffer.unmap();

    // Create output buffer
    const outputBuffer = this.device.createBuffer({
      size: batchSize * 4, // 4 bytes per f32
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
    });

    // Create staging buffer for reading back
    const stagingBuffer = this.device.createBuffer({
      size: batchSize * 4,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
    });

    // Create bind group
    const bindGroup = this.device.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: inputBuffer } },
        { binding: 1, resource: { buffer: outputBuffer } }
      ]
    });

    // Execute compute shader
    const commandEncoder = this.device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(this.pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(Math.ceil(batchSize / 64)); // 64 threads per workgroup
    passEncoder.end();

    // Copy output to staging
    commandEncoder.copyBufferToBuffer(
      outputBuffer, 0,
      stagingBuffer, 0,
      batchSize * 4
    );

    // Submit and wait
    this.device.queue.submit([commandEncoder.finish()]);

    // Read results
    await stagingBuffer.mapAsync(GPUMapMode.READ);
    const result = new Float32Array(stagingBuffer.getMappedRange()).slice();
    stagingBuffer.unmap();

    // Cleanup
    inputBuffer.destroy();
    outputBuffer.destroy();
    stagingBuffer.destroy();

    return result;
  }
}
