/**
 * Hello GPU - Basic WebGPU functionality test
 * Computes squares of an array of numbers on GPU
 */

import { WebGPUContext } from '../../src/gpu/WebGPUContext';

describe('WebGPU Hello World', () => {
  let ctx: WebGPUContext;

  beforeAll(async () => {
    if (!WebGPUContext.isAvailable()) {
      console.warn('WebGPU not available - skipping GPU tests');
      return;
    }

    ctx = WebGPUContext.getInstance();
    await ctx.initialize();
  });

  afterAll(() => {
    WebGPUContext.reset();
  });

  it('should square an array of numbers on GPU', async () => {
    if (!WebGPUContext.isAvailable()) {
      console.warn('Skipping: WebGPU not available');
      return;
    }

    const device = ctx.device;

    // Input data: [1, 2, 3, 4]
    const input = new Float32Array([1, 2, 3, 4]);
    const expected = new Float32Array([1, 4, 9, 16]);

    // Create GPU buffers
    const inputBuffer = device.createBuffer({
      size: input.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      mappedAtCreation: true
    });
    new Float32Array(inputBuffer.getMappedRange()).set(input);
    inputBuffer.unmap();

    const outputBuffer = device.createBuffer({
      size: input.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
    });

    // Create staging buffer for reading results back to CPU
    const stagingBuffer = device.createBuffer({
      size: input.byteLength,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
    });

    // WGSL compute shader - squares each element
    const shaderCode = `
      @group(0) @binding(0) var<storage, read> input: array<f32>;
      @group(0) @binding(1) var<storage, read_write> output: array<f32>;

      @compute @workgroup_size(1)
      fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
        let idx = global_id.x;
        output[idx] = input[idx] * input[idx];
      }
    `;

    const shaderModule = device.createShaderModule({
      code: shaderCode
    });

    // Create compute pipeline
    const computePipeline = device.createComputePipeline({
      layout: 'auto',
      compute: {
        module: shaderModule,
        entryPoint: 'main'
      }
    });

    // Create bind group
    const bindGroup = device.createBindGroup({
      layout: computePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: inputBuffer } },
        { binding: 1, resource: { buffer: outputBuffer } }
      ]
    });

    // Execute compute shader
    const commandEncoder = device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(computePipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(input.length); // One thread per element
    passEncoder.end();

    // Copy output to staging buffer
    commandEncoder.copyBufferToBuffer(
      outputBuffer, 0,
      stagingBuffer, 0,
      input.byteLength
    );

    // Submit commands
    device.queue.submit([commandEncoder.finish()]);

    // Read results back
    await stagingBuffer.mapAsync(GPUMapMode.READ);
    const result = new Float32Array(stagingBuffer.getMappedRange()).slice();
    stagingBuffer.unmap();

    // Verify results
    console.log('Input:', Array.from(input));
    console.log('Output:', Array.from(result));
    console.log('Expected:', Array.from(expected));

    expect(result.length).toBe(expected.length);
    for (let i = 0; i < result.length; i++) {
      expect(result[i]).toBeCloseTo(expected[i], 5);
    }

    // Cleanup
    inputBuffer.destroy();
    outputBuffer.destroy();
    stagingBuffer.destroy();
  });

  it('should handle larger arrays efficiently', async () => {
    if (!WebGPUContext.isAvailable()) {
      console.warn('Skipping: WebGPU not available');
      return;
    }

    const device = ctx.device;

    // Test with 1000 elements
    const size = 1000;
    const input = new Float32Array(size);
    for (let i = 0; i < size; i++) {
      input[i] = i + 1;
    }

    const inputBuffer = device.createBuffer({
      size: input.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      mappedAtCreation: true
    });
    new Float32Array(inputBuffer.getMappedRange()).set(input);
    inputBuffer.unmap();

    const outputBuffer = device.createBuffer({
      size: input.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
    });

    const stagingBuffer = device.createBuffer({
      size: input.byteLength,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
    });

    const shaderCode = `
      @group(0) @binding(0) var<storage, read> input: array<f32>;
      @group(0) @binding(1) var<storage, read_write> output: array<f32>;

      @compute @workgroup_size(64)
      fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
        let idx = global_id.x;
        if (idx < arrayLength(&input)) {
          output[idx] = input[idx] * input[idx];
        }
      }
    `;

    const shaderModule = device.createShaderModule({ code: shaderCode });
    const computePipeline = device.createComputePipeline({
      layout: 'auto',
      compute: { module: shaderModule, entryPoint: 'main' }
    });

    const bindGroup = device.createBindGroup({
      layout: computePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: inputBuffer } },
        { binding: 1, resource: { buffer: outputBuffer } }
      ]
    });

    const commandEncoder = device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(computePipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(Math.ceil(size / 64)); // 64 threads per workgroup
    passEncoder.end();

    commandEncoder.copyBufferToBuffer(outputBuffer, 0, stagingBuffer, 0, input.byteLength);
    device.queue.submit([commandEncoder.finish()]);

    await stagingBuffer.mapAsync(GPUMapMode.READ);
    const result = new Float32Array(stagingBuffer.getMappedRange()).slice();
    stagingBuffer.unmap();

    // Verify a few samples
    expect(result[0]).toBeCloseTo(1, 5);    // 1^2 = 1
    expect(result[9]).toBeCloseTo(100, 5);  // 10^2 = 100
    expect(result[99]).toBeCloseTo(10000, 5); // 100^2 = 10000

    console.log(`Successfully computed squares of ${size} numbers on GPU`);

    inputBuffer.destroy();
    outputBuffer.destroy();
    stagingBuffer.destroy();
  });
});
