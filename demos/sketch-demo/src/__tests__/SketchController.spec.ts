/**
 * Tests for SketchController - state management operations
 */

import { beforeEach, describe, expect, it } from 'vitest';
import { SketchController } from '../SketchController';
import type { SketchState } from '../types';
import {
    createInitialState,
    LineConstraintType,
    ToolMode,
} from '../types';

describe('SketchController - User Workflow', () => {
  let controller: SketchController;
  let state: SketchState;

  beforeEach(() => {
    state = createInitialState();
    controller = new SketchController(state);
  });

  it('should simulate: select line → select point → delete point', () => {
    // Setup: Create two points and a line
    const p1 = controller.createPoint(0, 0);
    const p2 = controller.createPoint(100, 0);
    const line = controller.createLine(p1, p2);

    expect(state.project.points).toHaveLength(7); // 5 demo + 2 new
    expect(state.project.lines).toHaveLength(5); // 4 demo + 1 new

    // User selects the line
    controller.selectEntity(line);
    expect(controller.isSelected(line)).toBe(true);
    expect(controller.getSelectedEntities()).toHaveLength(1);

    // User selects p2 (replaces selection)
    controller.selectEntity(p2);
    expect(controller.isSelected(p2)).toBe(true);
    expect(controller.isSelected(line)).toBe(false); // line no longer selected
    expect(controller.getSelectedEntities()).toHaveLength(1);

    // User presses delete
    controller.deleteSelected();

    // Point is deleted, and line is cascade deleted
    expect(state.project.points).toHaveLength(6); // p2 removed
    expect(state.project.lines).toHaveLength(4); // line removed
    expect(controller.getSelectedEntities()).toHaveLength(0); // selection cleared
  });

  it('should simulate: multi-select with shift', () => {
    const p1 = controller.createPoint(0, 0);
    const p2 = controller.createPoint(100, 0);
    const p3 = controller.createPoint(50, 100);

    // Select p1
    controller.selectEntity(p1);
    expect(controller.getSelectedEntities()).toHaveLength(1);

    // Shift+select p2 (add to selection)
    controller.selectEntity(p2, true);
    expect(controller.getSelectedEntities()).toHaveLength(2);
    expect(controller.isSelected(p1)).toBe(true);
    expect(controller.isSelected(p2)).toBe(true);

    // Shift+select p3 (add to selection)
    controller.selectEntity(p3, true);
    expect(controller.getSelectedEntities()).toHaveLength(3);

    // Delete all selected
    controller.deleteSelected();
    expect(state.project.points).toHaveLength(5); // All 3 removed (5 demo points remain)
  });

  it('should simulate: click empty space to deselect', () => {
    const p1 = controller.createPoint(50, 50);
    controller.selectEntity(p1);
    expect(controller.isSelected(p1)).toBe(true);

    // User clicks empty space (no entity found)
    const entityAtEmptySpace = controller.findEntityAt(500, 500);
    if (!entityAtEmptySpace) {
      controller.clearSelection();
    }

    expect(controller.getSelectedEntities()).toHaveLength(0);
  });
});

describe('SketchController - Selection', () => {
  let controller: SketchController;

  beforeEach(() => {
    controller = new SketchController(createInitialState());
  });

  it('should select an entity', () => {
    const point = controller.createPoint(10, 20);

    controller.selectEntity(point);

    expect(controller.isSelected(point)).toBe(true);
    expect(controller.getSelectedEntities()).toContain(point);
  });

  it('should replace selection by default', () => {
    const p1 = controller.createPoint(10, 20);
    const p2 = controller.createPoint(30, 40);

    controller.selectEntity(p1);
    controller.selectEntity(p2); // Replaces p1

    expect(controller.isSelected(p1)).toBe(false);
    expect(controller.isSelected(p2)).toBe(true);
    expect(controller.getSelectedEntities()).toHaveLength(1);
  });

  it('should add to selection when addToSelection=true', () => {
    const p1 = controller.createPoint(10, 20);
    const p2 = controller.createPoint(30, 40);

    controller.selectEntity(p1);
    controller.selectEntity(p2, true); // Add to selection

    expect(controller.isSelected(p1)).toBe(true);
    expect(controller.isSelected(p2)).toBe(true);
    expect(controller.getSelectedEntities()).toHaveLength(2);
  });

  it('should deselect an entity', () => {
    const point = controller.createPoint(10, 20);
    controller.selectEntity(point);

    controller.deselectEntity(point);

    expect(controller.isSelected(point)).toBe(false);
  });

  it('should clear all selections', () => {
    const p1 = controller.createPoint(10, 20);
    const p2 = controller.createPoint(30, 40);
    controller.selectEntity(p1);
    controller.selectEntity(p2, true);

    controller.clearSelection();

    expect(controller.getSelectedEntities()).toHaveLength(0);
  });

  it('should set hovered entity', () => {
    const point = controller.createPoint(10, 20);

    controller.setHoveredEntity(point);

    expect(controller.state.selection.hoveredEntity).toBe(point);
  });
});

describe('SketchController - Entity Creation', () => {
  let controller: SketchController;

  beforeEach(() => {
    controller = new SketchController(createInitialState());
  });

  it('should create a point', () => {
    const point = controller.createPoint(100, 200, false, 'P1');

    expect(controller.state.project.points).toContain(point);
    expect(point.x).toBe(100);
    expect(point.y).toBe(200);
    expect(point.label).toBe('P1');
  });

  it('should create a line', () => {
    const p1 = controller.createPoint(0, 0);
    const p2 = controller.createPoint(100, 0);

    const line = controller.createLine(p1, p2, LineConstraintType.Horizontal, 100);

    expect(controller.state.project.lines).toContain(line);
    expect(line.start).toBe(p1);
    expect(line.end).toBe(p2);
    expect(line.constraintType).toBe(LineConstraintType.Horizontal);
    expect(line.fixedLength).toBe(100);
  });

  it('should create a circle', () => {
    const center = controller.createPoint(50, 50);

    const circle = controller.createCircle(center, 25);

    expect(controller.state.project.circles).toContain(circle);
    expect(circle.center).toBe(center);
    expect(circle.radius).toBe(25);
    expect(circle.fixedRadius).toBe(true);
  });
});

describe('SketchController - Entity Deletion', () => {
  let controller: SketchController;

  beforeEach(() => {
    controller = new SketchController(createInitialState());
  });

  it('should delete a standalone point', () => {
    const point = controller.createPoint(100, 100);
    const initialCount = controller.state.project.points.length;

    controller.deleteEntity(point);

    expect(controller.state.project.points).toHaveLength(initialCount - 1);
    expect(controller.state.project.points).not.toContain(point);
  });

  it('should cascade delete lines when point is deleted', () => {
    const p1 = controller.createPoint(0, 0);
    const p2 = controller.createPoint(100, 0);
    const p3 = controller.createPoint(50, 100);
    const line1 = controller.createLine(p1, p2);
    const line2 = controller.createLine(p2, p3);

    const initialLineCount = controller.state.project.lines.length;

    controller.deleteEntity(p2);

    // Both lines should be deleted
    expect(controller.state.project.lines).toHaveLength(initialLineCount - 2);
    expect(controller.state.project.lines).not.toContain(line1);
    expect(controller.state.project.lines).not.toContain(line2);
  });

  it('should cascade delete circles when center point is deleted', () => {
    const center = controller.createPoint(100, 100);
    const circle = controller.createCircle(center, 50);

    const initialCircleCount = controller.state.project.circles.length;

    controller.deleteEntity(center);

    expect(controller.state.project.circles).toHaveLength(initialCircleCount - 1);
    expect(controller.state.project.circles).not.toContain(circle);
  });

  it('should delete selected entities', () => {
    const p1 = controller.createPoint(0, 0);
    const p2 = controller.createPoint(100, 0);

    controller.selectEntity(p1);
    controller.selectEntity(p2, true);

    const initialCount = controller.state.project.points.length;
    controller.deleteSelected();

    expect(controller.state.project.points).toHaveLength(initialCount - 2);
    expect(controller.getSelectedEntities()).toHaveLength(0);
  });

  it('should remove deleted entity from selection', () => {
    const point = controller.createPoint(50, 50);
    controller.selectEntity(point);

    controller.deleteEntity(point);

    expect(controller.isSelected(point)).toBe(false);
  });
});

describe('SketchController - Entity Modification', () => {
  let controller: SketchController;

  beforeEach(() => {
    controller = new SketchController(createInitialState());
  });

  it('should update point position', () => {
    const point = controller.createPoint(0, 0);

    controller.updatePointPosition(point, 50, 75);

    expect(point.x).toBe(50);
    expect(point.y).toBe(75);
  });

  it('should not update pinned point position', () => {
    const point = controller.createPoint(0, 0, true); // pinned

    controller.updatePointPosition(point, 50, 75);

    expect(point.x).toBe(0); // Unchanged
    expect(point.y).toBe(0);
  });

  it('should toggle point pinned state', () => {
    const point = controller.createPoint(0, 0);

    controller.togglePointPinned(point);
    expect(point.pinned).toBe(true);

    controller.togglePointPinned(point);
    expect(point.pinned).toBe(false);
  });

  it('should update line constraint type', () => {
    const p1 = controller.createPoint(0, 0);
    const p2 = controller.createPoint(100, 0);
    const line = controller.createLine(p1, p2);

    controller.updateLineConstraint(line, LineConstraintType.Horizontal);

    expect(line.constraintType).toBe(LineConstraintType.Horizontal);
  });

  it('should update circle radius', () => {
    const center = controller.createPoint(50, 50);
    const circle = controller.createCircle(center, 25);

    controller.updateCircleRadius(circle, 50);

    expect(circle.radius).toBe(50);
  });
});

describe('SketchController - Multi-step Creation', () => {
  let controller: SketchController;

  beforeEach(() => {
    controller = new SketchController(createInitialState());
  });

  it('should handle line creation workflow', () => {
    const p1 = controller.createPoint(0, 0);
    const p2 = controller.createPoint(100, 0);

    // Start line creation
    controller.startLineCreation(p1);
    expect(controller.state.creation.lineStartPoint).toBe(p1);

    // Finish line creation
    const line = controller.finishLineCreation(p2);

    expect(line).not.toBeNull();
    expect(line!.start).toBe(p1);
    expect(line!.end).toBe(p2);
    expect(controller.state.creation.lineStartPoint).toBeUndefined();
  });

  it('should handle circle creation workflow', () => {
    const center = controller.createPoint(50, 50);

    // Start circle creation
    controller.startCircleCreation(center);
    expect(controller.state.creation.circleCenterPoint).toBe(center);

    // Finish circle creation
    const circle = controller.finishCircleCreation(30);

    expect(circle).not.toBeNull();
    expect(circle!.center).toBe(center);
    expect(circle!.radius).toBe(30);
    expect(controller.state.creation.circleCenterPoint).toBeUndefined();
  });

  it('should clear creation state when changing tools', () => {
    const p1 = controller.createPoint(0, 0);
    controller.startLineCreation(p1);

    controller.setToolMode(ToolMode.Circle);

    expect(controller.state.creation.lineStartPoint).toBeUndefined();
  });
});

describe('SketchController - Hit Testing', () => {
  let controller: SketchController;

  beforeEach(() => {
    controller = new SketchController(createInitialState());
  });

  it('should find point at position', () => {
    const point = controller.createPoint(500, 500);

    const found = controller.findEntityAt(502, 498, 5);

    expect(found).toBe(point);
  });

  it('should find line at position', () => {
    const p1 = controller.createPoint(0, 0);
    const p2 = controller.createPoint(100, 0);
    const line = controller.createLine(p1, p2);

    // Click near the middle of the line
    const found = controller.findEntityAt(50, 2, 5);

    expect(found).toBe(line);
  });

  it('should find circle at position', () => {
    const center = controller.createPoint(100, 100);
    const circle = controller.createCircle(center, 50);

    // Click near the circle perimeter
    const found = controller.findEntityAt(150, 100, 5);

    expect(found).toBe(circle);
  });

  it('should return null when nothing is at position', () => {
    const found = controller.findEntityAt(999, 999);

    expect(found).toBeNull();
  });

  it('should prioritize points over lines', () => {
    const p1 = controller.createPoint(0, 0);
    const p2 = controller.createPoint(100, 0);
    controller.createLine(p1, p2);

    // Click directly on p1 (which is also on the line)
    const found = controller.findEntityAt(0, 0);

    expect(found).toBe(p1); // Point, not line
  });
});

describe('SketchController - Query Helpers', () => {
  let controller: SketchController;

  beforeEach(() => {
    controller = new SketchController(createInitialState());
  });

  it('should get entities connected to a point', () => {
    const p1 = controller.createPoint(0, 0);
    const p2 = controller.createPoint(100, 0);
    const p3 = controller.createPoint(50, 100);

    const line1 = controller.createLine(p1, p2);
    const line2 = controller.createLine(p2, p3);
    const circle = controller.createCircle(p2, 20);

    const connected = controller.getConnectedEntities(p2);

    expect(connected).toHaveLength(3);
    expect(connected).toContain(line1);
    expect(connected).toContain(line2);
    expect(connected).toContain(circle);
  });
});
