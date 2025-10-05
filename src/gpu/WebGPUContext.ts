/**
 * WebGPU Context - Singleton for managing GPU device and queue
 *
 * Provides centralized access to WebGPU adapter, device, and queue.
 * Handles initialization, error handling, and cleanup.
 */

import { create, globals } from 'webgpu';

// Assign WebGPU globals to global scope
Object.assign(globalThis, globals);

export class WebGPUContext {
  private static instance: WebGPUContext | null = null;

  private gpu: GPU;
  private _adapter: GPUAdapter | null = null;
  private _device: GPUDevice | null = null;

  private constructor() {
    // Create WebGPU navigator
    const navigator = { gpu: create([]) };
    this.gpu = navigator.gpu;
  }

  /**
   * Get the singleton instance
   */
  static getInstance(): WebGPUContext {
    if (!WebGPUContext.instance) {
      WebGPUContext.instance = new WebGPUContext();
    }
    return WebGPUContext.instance;
  }

  /**
   * Initialize WebGPU adapter and device
   * @param powerPreference - 'low-power' or 'high-performance'
   */
  async initialize(powerPreference: 'low-power' | 'high-performance' = 'high-performance'): Promise<void> {
    if (this._device) {
      return; // Already initialized
    }

    try {
      // Request adapter
      this._adapter = await this.gpu.requestAdapter({
        powerPreference
      });

      if (!this._adapter) {
        throw new Error('WebGPU: Failed to get GPU adapter');
      }

      // Request device
      this._device = await this._adapter.requestDevice();

      if (!this._device) {
        throw new Error('WebGPU: Failed to get GPU device');
      }

      // Log device info
      console.log('WebGPU initialized:', {
        adapter: 'Unknown adapter',
        limits: this._device.limits
      });

    } catch (error) {
      throw new Error(`WebGPU initialization failed: ${error}`);
    }
  }

  /**
   * Get the GPU adapter (initialized)
   */
  get adapter(): GPUAdapter {
    if (!this._adapter) {
      throw new Error('WebGPU not initialized. Call initialize() first.');
    }
    return this._adapter;
  }

  /**
   * Get the GPU device (initialized)
   */
  get device(): GPUDevice {
    if (!this._device) {
      throw new Error('WebGPU not initialized. Call initialize() first.');
    }
    return this._device;
  }

  /**
   * Get the GPU command queue
   */
  get queue(): GPUQueue {
    return this.device.queue;
  }

  /**
   * Check if WebGPU is available
   */
  static isAvailable(): boolean {
    try {
      const navigator = { gpu: create([]) };
      return !!navigator.gpu;
    } catch {
      return false;
    }
  }

  /**
   * Cleanup and destroy device
   */
  destroy(): void {
    if (this._device) {
      this._device.destroy();
      this._device = null;
      this._adapter = null;
    }
  }

  /**
   * Reset singleton (mainly for testing)
   */
  static reset(): void {
    if (WebGPUContext.instance) {
      WebGPUContext.instance.destroy();
      WebGPUContext.instance = null;
    }
  }
}
