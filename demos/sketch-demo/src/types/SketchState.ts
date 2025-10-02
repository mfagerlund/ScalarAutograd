/**
 * UI and application state types
 */

import type { Entity } from './Entities';
import type { Project } from './Project';
import { createDemoProject } from './Project';

/**
 * Selection state
 */
export interface SelectionState {
  selectedEntities: Set<Entity>;
  hoveredEntity: Entity | null;
}

/**
 * Tool modes for entity creation
 */
export enum ToolMode {
  Select = 'select',
  Point = 'point',
  Line = 'line',
  Circle = 'circle',
}

/**
 * Creation tool state (for multi-click operations like line/circle creation)
 */
export interface CreationState {
  mode: ToolMode;
  // For line creation: first point clicked
  lineStartPoint?: import('./Entities').Point;
  // For circle creation: center point clicked
  circleCenterPoint?: import('./Entities').Point;
  // Temporary mouse position for preview
  mouseX?: number;
  mouseY?: number;
}

/**
 * Solver status
 */
export enum SolverStatus {
  Idle = 'idle',
  Solving = 'solving',
  Success = 'success',
  Failed = 'failed',
  OverConstrained = 'over-constrained',
}

/**
 * Solver state and diagnostics
 */
export interface SolverState {
  isConverged: boolean;
  residual: number;
  iterations: number;
  error: string | null;
}

/**
 * View/camera state for pan and zoom
 */
export interface ViewState {
  offsetX: number;
  offsetY: number;
  zoom: number; // 1.0 = 100%, >1 = zoomed in
}

/**
 * Overall sketch state
 */
export interface SketchState {
  // Current project
  project: Project;

  // UI state
  selection: SelectionState;
  creation: CreationState;
  solver: SolverState;
  view: ViewState;

  // Settings
  showGrid: boolean;
  snapToGrid: boolean;
  gridSize: number;
}

/**
 * Initial sketch state factory
 */
export function createInitialState(): SketchState {
  return {
    project: createDemoProject(),

    selection: {
      selectedEntities: new Set(),
      hoveredEntity: null,
    },

    creation: {
      mode: ToolMode.Select,
    },

    solver: {
      isConverged: true,
      residual: 0,
      iterations: 0,
      error: null,
    },

    view: {
      offsetX: 0,
      offsetY: 0,
      zoom: 1.0,
    },

    showGrid: true,
    snapToGrid: false,
    gridSize: 20,
  };
}
