/**
 * SketchController - Handles all state mutations for the sketch application
 * Provides intent-based methods for user actions
 */

import { SketchSolver } from './SketchSolver';
import type { Constraint } from './types/Constraints';
import { ConstraintType } from './types/Constraints';
import type { Circle, Entity, Line, Point } from './types/Entities';
import { isCircle, isLine, isPoint, LineConstraintType } from './types/Entities';
import type { SketchState } from './types/SketchState';
import { ToolMode } from './types/SketchState';

export class SketchController {
  private solver: SketchSolver;

  constructor(public state: SketchState) {
    this.solver = new SketchSolver({ tolerance: 1e-4, maxIterations: 200 });
  }

  /**
   * Solve constraints and update solver state.
   * Call this after any geometry changes.
   */
  solve(): void {
    const result = this.solver.solve(this.state.project);
    this.state.solver.isConverged = result.converged;
    this.state.solver.residual = result.residual;
    this.state.solver.iterations = result.iterations;
    this.state.solver.error = result.error;
  }

  // ============================================================================
  // SELECTION OPERATIONS
  // ============================================================================

  /**
   * Select an entity (replaces current selection unless shift is held)
   */
  selectEntity(entity: Entity, addToSelection = false): void {
    if (!addToSelection) {
      this.state.selection.selectedEntities.clear();
    }
    this.state.selection.selectedEntities.add(entity);
  }

  /**
   * Deselect an entity
   */
  deselectEntity(entity: Entity): void {
    this.state.selection.selectedEntities.delete(entity);
  }

  /**
   * Clear all selections
   */
  clearSelection(): void {
    this.state.selection.selectedEntities.clear();
  }

  /**
   * Set hovered entity (for hover effects)
   */
  setHoveredEntity(entity: Entity | null): void {
    this.state.selection.hoveredEntity = entity;
  }

  /**
   * Get all selected entities as array
   */
  getSelectedEntities(): Entity[] {
    return Array.from(this.state.selection.selectedEntities);
  }

  /**
   * Check if entity is selected
   */
  isSelected(entity: Entity): boolean {
    return this.state.selection.selectedEntities.has(entity);
  }

  // ============================================================================
  // ENTITY CREATION
  // ============================================================================

  /**
   * Create a new point at the given position
   */
  createPoint(x: number, y: number, pinned = false, label?: string): Point {
    const point: Point = { x, y, pinned, label };
    this.state.project.points.push(point);
    this.solve();
    return point;
  }

  /**
   * Create a line between two points
   */
  createLine(
    start: Point,
    end: Point,
    constraintType = LineConstraintType.Free,
    fixedLength?: number,
    label?: string
  ): Line {
    const line: Line = { start, end, constraintType, fixedLength, label };
    this.state.project.lines.push(line);
    this.solve();
    return line;
  }

  /**
   * Create a circle at a center point
   */
  createCircle(
    center: Point,
    radius: number,
    fixedRadius = true,
    label?: string
  ): Circle {
    const circle: Circle = { center, radius, fixedRadius, label };
    this.state.project.circles.push(circle);
    this.solve();
    return circle;
  }

  /**
   * Add a constraint to the project
   */
  addConstraint(constraint: Constraint): void {
    this.state.project.constraints.push(constraint);
    this.solve();
  }

  // ============================================================================
  // ENTITY DELETION (with cascade)
  // ============================================================================

  /**
   * Delete an entity and cascade delete dependent entities and constraints
   */
  deleteEntity(entity: Entity): void {
    if (isPoint(entity)) {
      this.deletePoint(entity);
    } else if (isLine(entity)) {
      this.deleteLine(entity);
    } else if (isCircle(entity)) {
      this.deleteCircle(entity);
    }

    // Remove from selection
    this.deselectEntity(entity);
  }

  /**
   * Delete all currently selected entities
   */
  deleteSelected(): void {
    const selected = this.getSelectedEntities();
    selected.forEach(entity => this.deleteEntity(entity));
    this.solve();
  }

  /**
   * Delete a point (cascades to lines, circles, and constraints)
   */
  private deletePoint(point: Point): void {
    // Find and delete all lines connected to this point
    const connectedLines = this.state.project.lines.filter(
      l => l.start === point || l.end === point
    );
    connectedLines.forEach(line => this.deleteLine(line));

    // Find and delete all circles centered on this point
    const connectedCircles = this.state.project.circles.filter(
      c => c.center === point
    );
    connectedCircles.forEach(circle => this.deleteCircle(circle));

    // Delete constraints involving this point
    this.deleteConstraintsInvolving(point);

    // Remove the point itself
    this.state.project.points = this.state.project.points.filter(p => p !== point);
  }

  /**
   * Delete a line (cascades to constraints)
   */
  private deleteLine(line: Line): void {
    // Delete constraints involving this line
    this.deleteConstraintsInvolving(line);

    // Remove the line itself
    this.state.project.lines = this.state.project.lines.filter(l => l !== line);
  }

  /**
   * Delete a circle (cascades to constraints)
   */
  private deleteCircle(circle: Circle): void {
    // Delete constraints involving this circle
    this.deleteConstraintsInvolving(circle);

    // Remove the circle itself
    this.state.project.circles = this.state.project.circles.filter(c => c !== circle);
  }

  /**
   * Delete all constraints involving a specific entity
   */
  private deleteConstraintsInvolving(entity: Entity): void {
    this.state.project.constraints = this.state.project.constraints.filter(c => {
      // Check if constraint references this entity
      switch (c.type) {
        case ConstraintType.Collinear:
        case ConstraintType.Parallel:
        case ConstraintType.Perpendicular:
          return c.line1 !== entity && c.line2 !== entity;
        case ConstraintType.Angle:
          return c.line1 !== entity && c.line2 !== entity;
        case ConstraintType.PointOnLine:
          return c.point !== entity && c.line !== entity;
        case ConstraintType.PointOnCircle:
          return c.point !== entity && c.circle !== entity;
        case ConstraintType.Tangent:
          return c.line !== entity && c.circle !== entity;
        case ConstraintType.RadialAlignment:
          return c.point1 !== entity && c.circle !== entity && c.point2 !== entity;
        default:
          return true;
      }
    });
  }

  // ============================================================================
  // ENTITY MODIFICATION
  // ============================================================================

  /**
   * Update point position (for dragging)
   */
  updatePointPosition(point: Point, x: number, y: number): void {
    if (!point.pinned) {
      point.x = x;
      point.y = y;
      try {
        this.solve();
      } catch (error) {
        console.error('Solver error when moving point:', error);
        console.log('Point:', point, 'Position:', x, y);
        console.log('Project state:', this.state.project);
      }
    }
  }

  /**
   * Toggle point pinned state
   */
  togglePointPinned(point: Point): void {
    point.pinned = !point.pinned;
  }

  /**
   * Update line constraint type
   */
  updateLineConstraint(line: Line, constraintType: LineConstraintType): void {
    line.constraintType = constraintType;
    this.solve();
  }

  /**
   * Update line fixed length
   */
  updateLineLength(line: Line, length: number | undefined): void {
    line.fixedLength = length;
    this.solve();
  }

  /**
   * Update circle radius
   */
  updateCircleRadius(circle: Circle, radius: number): void {
    circle.radius = radius;
    this.solve();
  }

  /**
   * Toggle circle fixed radius
   */
  toggleCircleFixedRadius(circle: Circle): void {
    circle.fixedRadius = !circle.fixedRadius;
  }

  // ============================================================================
  // TOOL MODE
  // ============================================================================

  /**
   * Set the current tool mode
   */
  setToolMode(mode: ToolMode): void {
    this.state.creation.mode = mode;
    // Clear creation state when switching tools
    this.state.creation.lineStartPoint = undefined;
    this.state.creation.circleCenterPoint = undefined;
    this.state.creation.mouseX = undefined;
    this.state.creation.mouseY = undefined;
  }

  /**
   * Get the current tool mode
   */
  getToolMode(): ToolMode {
    return this.state.creation.mode;
  }

  // ============================================================================
  // MULTI-STEP CREATION (Line and Circle tools)
  // ============================================================================

  /**
   * Start line creation (first point clicked)
   */
  startLineCreation(point: Point): void {
    this.state.creation.lineStartPoint = point;
  }

  /**
   * Finish line creation (second point clicked)
   */
  finishLineCreation(point: Point): Line | null {
    const start = this.state.creation.lineStartPoint;
    if (!start) return null;

    const line = this.createLine(start, point);

    // Clear creation state
    this.state.creation.lineStartPoint = undefined;

    return line;
  }

  /**
   * Start circle creation (center point clicked)
   */
  startCircleCreation(point: Point): void {
    this.state.creation.circleCenterPoint = point;
  }

  /**
   * Finish circle creation (radius determined)
   */
  finishCircleCreation(radius: number): Circle | null {
    const center = this.state.creation.circleCenterPoint;
    if (!center) return null;

    const circle = this.createCircle(center, radius, true);

    // Clear creation state
    this.state.creation.circleCenterPoint = undefined;

    return circle;
  }

  /**
   * Update preview mouse position (for rubber-band preview)
   */
  updatePreviewPosition(x: number, y: number): void {
    this.state.creation.mouseX = x;
    this.state.creation.mouseY = y;
  }

  /**
   * Clear preview (mouse left canvas, etc.)
   */
  clearPreview(): void {
    this.state.creation.mouseX = undefined;
    this.state.creation.mouseY = undefined;
  }

  // ============================================================================
  // QUERY HELPERS
  // ============================================================================

  /**
   * Find entity at a given position (for hit testing)
   * Returns the topmost entity at the position, or null
   */
  findEntityAt(x: number, y: number, tolerance = 8): Entity | null {
    // Check points first (highest priority)
    for (const point of this.state.project.points) {
      const dx = point.x - x;
      const dy = point.y - y;
      if (Math.sqrt(dx * dx + dy * dy) <= tolerance) {
        return point;
      }
    }

    // Check circles
    for (const circle of this.state.project.circles) {
      const dx = circle.center.x - x;
      const dy = circle.center.y - y;
      const dist = Math.sqrt(dx * dx + dy * dy);
      // Check if click is near the circle perimeter
      if (Math.abs(dist - circle.radius) <= tolerance) {
        return circle;
      }
    }

    // Check lines
    for (const line of this.state.project.lines) {
      const dist = this.distanceToLineSegment(x, y, line.start, line.end);
      if (dist <= tolerance) {
        return line;
      }
    }

    return null;
  }

  /**
   * Calculate distance from point to line segment
   */
  private distanceToLineSegment(px: number, py: number, p1: Point, p2: Point): number {
    const dx = p2.x - p1.x;
    const dy = p2.y - p1.y;
    const lengthSquared = dx * dx + dy * dy;

    if (lengthSquared === 0) {
      // Line segment is a point
      const dpx = px - p1.x;
      const dpy = py - p1.y;
      return Math.sqrt(dpx * dpx + dpy * dpy);
    }

    // Calculate projection of point onto line segment
    let t = ((px - p1.x) * dx + (py - p1.y) * dy) / lengthSquared;
    t = Math.max(0, Math.min(1, t)); // Clamp to [0, 1]

    const closestX = p1.x + t * dx;
    const closestY = p1.y + t * dy;

    const distX = px - closestX;
    const distY = py - closestY;
    return Math.sqrt(distX * distX + distY * distY);
  }

  /**
   * Get all entities connected to a point
   */
  getConnectedEntities(point: Point): Entity[] {
    const connected: Entity[] = [];

    // Find lines connected to this point
    this.state.project.lines.forEach(line => {
      if (line.start === point || line.end === point) {
        connected.push(line);
      }
    });

    // Find circles centered on this point
    this.state.project.circles.forEach(circle => {
      if (circle.center === point) {
        connected.push(circle);
      }
    });

    return connected;
  }
}
