import { useRef, useEffect, useState } from 'react';
import { SketchController } from './SketchController';
import type { SketchState } from './types/SketchState';
import { createInitialState } from './types/SketchState';
import type { Entity, Point, Line, Circle } from './types/Entities';
import { ToolMode } from './types/SketchState';
import { isPoint, isLine, isCircle } from './types/Entities';
import { ConstraintType } from './types/Constraints';
import { createDemoProject } from './types/Project';
import './App.css';

function App() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [controller] = useState(() => {
    const initialState = createInitialState();
    // Replace empty project with demo project
    initialState.project = createDemoProject();
    return new SketchController(initialState);
  });

  const [, forceUpdate] = useState({});
  const [mousePos, setMousePos] = useState({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState(false);
  const [draggedPoint, setDraggedPoint] = useState<Point | null>(null);
  const lastSizeRef = useRef({ width: 0, height: 0 });

  // Render function with smooth animations
  const render = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    // Update canvas size if container size changed
    const rect = canvas.getBoundingClientRect();
    if (rect.width !== lastSizeRef.current.width || rect.height !== lastSizeRef.current.height) {
      canvas.width = rect.width;
      canvas.height = rect.height;
      lastSizeRef.current = { width: rect.width, height: rect.height };
    }

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Clear with subtle grid
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Draw subtle grid
    ctx.strokeStyle = '#f0f0f0';
    ctx.lineWidth = 1;
    const gridSize = 20;
    for (let x = 0; x < canvas.width; x += gridSize) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, canvas.height);
      ctx.stroke();
    }
    for (let y = 0; y < canvas.height; y += gridSize) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(canvas.width, y);
      ctx.stroke();
    }

    const state = controller.state;

    // Draw lines
    for (const line of state.project.lines) {
      const isSelected = controller.isSelected(line);
      const isHovered = state.selection.hoveredEntity === line;

      // Hover glow
      if (isHovered && !isSelected) {
        ctx.strokeStyle = '#3498db';
        ctx.lineWidth = 8;
        ctx.setLineDash([]);
        ctx.globalAlpha = 0.3;
        ctx.beginPath();
        ctx.moveTo(line.start.x, line.start.y);
        ctx.lineTo(line.end.x, line.end.y);
        ctx.stroke();
        ctx.globalAlpha = 1;
      }

      // Line stroke
      ctx.strokeStyle = isSelected ? '#3498db' : (isHovered ? '#3498db' : '#2c3e50');
      ctx.lineWidth = isHovered ? 3 : 2;
      ctx.setLineDash([]);
      ctx.beginPath();
      ctx.moveTo(line.start.x, line.start.y);
      ctx.lineTo(line.end.x, line.end.y);
      ctx.stroke();

      // Selection highlight
      if (isSelected) {
        ctx.strokeStyle = '#3498db';
        ctx.lineWidth = 4;
        ctx.setLineDash([8, 4]);
        ctx.globalAlpha = 0.5;
        ctx.beginPath();
        ctx.moveTo(line.start.x, line.start.y);
        ctx.lineTo(line.end.x, line.end.y);
        ctx.stroke();
        ctx.globalAlpha = 1;
        ctx.setLineDash([]);
      }
    }

    // Draw circles
    for (const circle of state.project.circles) {
      const isSelected = controller.isSelected(circle);
      const isHovered = state.selection.hoveredEntity === circle;

      ctx.strokeStyle = isSelected ? '#3498db' : '#2c3e50';
      ctx.lineWidth = isHovered ? 3 : 2;
      ctx.setLineDash([]);
      ctx.beginPath();
      ctx.arc(circle.center.x, circle.center.y, circle.radius, 0, Math.PI * 2);
      ctx.stroke();

      // Selection highlight
      if (isSelected) {
        ctx.strokeStyle = '#3498db';
        ctx.lineWidth = 4;
        ctx.setLineDash([8, 4]);
        ctx.globalAlpha = 0.5;
        ctx.beginPath();
        ctx.arc(circle.center.x, circle.center.y, circle.radius, 0, Math.PI * 2);
        ctx.stroke();
        ctx.globalAlpha = 1;
        ctx.setLineDash([]);
      }
    }

    // Draw preview for line/circle creation
    if (state.creation.lineStartPoint && state.creation.mouseX !== undefined) {
      ctx.strokeStyle = '#3498db';
      ctx.lineWidth = 2;
      ctx.setLineDash([5, 5]);
      ctx.globalAlpha = 0.6;
      ctx.beginPath();
      ctx.moveTo(state.creation.lineStartPoint.x, state.creation.lineStartPoint.y);
      ctx.lineTo(state.creation.mouseX, state.creation.mouseY!);
      ctx.stroke();
      ctx.globalAlpha = 1;
      ctx.setLineDash([]);
    }

    if (state.creation.circleCenterPoint && state.creation.mouseX !== undefined) {
      const dx = state.creation.mouseX - state.creation.circleCenterPoint.x;
      const dy = state.creation.mouseY! - state.creation.circleCenterPoint.y;
      const radius = Math.sqrt(dx * dx + dy * dy);

      ctx.strokeStyle = '#3498db';
      ctx.lineWidth = 2;
      ctx.setLineDash([5, 5]);
      ctx.globalAlpha = 0.6;
      ctx.beginPath();
      ctx.arc(state.creation.circleCenterPoint.x, state.creation.circleCenterPoint.y, radius, 0, Math.PI * 2);
      ctx.stroke();
      ctx.globalAlpha = 1;
      ctx.setLineDash([]);
    }

    // Draw snap indicator (when hovering near a point)
    if (state.creation.mouseX !== undefined && state.creation.mouseY !== undefined) {
      const pointSnap = snapToPoint(state.creation.mouseX, state.creation.mouseY);
      if (pointSnap) {
        ctx.strokeStyle = '#3498db';
        ctx.lineWidth = 2;
        ctx.setLineDash([]);
        ctx.globalAlpha = 0.5;
        ctx.beginPath();
        ctx.arc(pointSnap.x, pointSnap.y, 12, 0, Math.PI * 2);
        ctx.stroke();
        ctx.globalAlpha = 1;
      }
    }

    // Draw points (on top of everything)
    for (const point of state.project.points) {
      const isSelected = controller.isSelected(point);
      const isHovered = state.selection.hoveredEntity === point;

      // Outer ring
      if (isSelected || isHovered) {
        ctx.fillStyle = isSelected ? '#3498db' : '#95a5a6';
        ctx.beginPath();
        ctx.arc(point.x, point.y, 8, 0, Math.PI * 2);
        ctx.fill();
      }

      // Main point
      ctx.fillStyle = point.pinned ? '#e74c3c' : '#2c3e50';
      ctx.beginPath();
      ctx.arc(point.x, point.y, 5, 0, Math.PI * 2);
      ctx.fill();

      // Pinned indicator
      if (point.pinned) {
        ctx.fillStyle = '#ffffff';
        ctx.font = '8px FontAwesome';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText('\uf08d', point.x, point.y); // thumbtack icon
      }
    }
  };

  // Animation loop
  useEffect(() => {
    let animationId: number;
    const animate = () => {
      render();
      animationId = requestAnimationFrame(animate);
    };
    animate();
    return () => cancelAnimationFrame(animationId);
  }, [controller]);

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Tool shortcuts
      if (e.key === 'v' || e.key === 'V') {
        controller.setToolMode(ToolMode.Select);
        forceUpdate({});
      } else if (e.key === 'p' || e.key === 'P') {
        controller.setToolMode(ToolMode.Point);
        forceUpdate({});
      } else if (e.key === 'l' || e.key === 'L') {
        controller.setToolMode(ToolMode.Line);
        forceUpdate({});
      } else if (e.key === 'c' || e.key === 'C') {
        controller.setToolMode(ToolMode.Circle);
        forceUpdate({});
      } else if (e.key === 'Delete' || e.key === 'Backspace') {
        controller.deleteSelected();
        forceUpdate({});
      } else if (e.key === 'Escape') {
        controller.clearSelection();
        controller.setToolMode(ToolMode.Select);
        forceUpdate({});
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [controller]);

  // Snap helper functions
  const snapToGrid = (x: number, y: number, gridSize = 20): { x: number; y: number } => {
    return {
      x: Math.round(x / gridSize) * gridSize,
      y: Math.round(y / gridSize) * gridSize,
    };
  };

  const snapToPoint = (x: number, y: number, snapDistance = 10): { x: number; y: number } | null => {
    for (const point of controller.state.project.points) {
      const dx = point.x - x;
      const dy = point.y - y;
      const dist = Math.sqrt(dx * dx + dy * dy);
      if (dist <= snapDistance) {
        return { x: point.x, y: point.y };
      }
    }
    return null;
  };

  const getSnappedPosition = (x: number, y: number, enableSnap = true): { x: number; y: number } => {
    if (!enableSnap) return { x, y };

    // Only snap to existing points (not grid)
    const pointSnap = snapToPoint(x, y);
    if (pointSnap) return pointSnap;

    // No grid snapping by default
    return { x, y };
  };

  // Canvas interaction handlers
  const handleMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const rect = canvasRef.current!.getBoundingClientRect();
    let x = e.clientX - rect.left;
    let y = e.clientY - rect.top;

    // Apply snapping unless Shift is held (Shift = multi-select or no-snap)
    const enableSnap = !e.shiftKey || controller.getToolMode() !== ToolMode.Select;
    if (enableSnap) {
      const snapped = getSnappedPosition(x, y);
      x = snapped.x;
      y = snapped.y;
    }

    const entity = controller.findEntityAt(x, y);
    const mode = controller.getToolMode();

    if (mode === ToolMode.Select) {
      if (entity) {
        if (isPoint(entity) && !entity.pinned) {
          setIsDragging(true);
          setDraggedPoint(entity);
        }
        controller.selectEntity(entity, e.shiftKey);
      } else {
        controller.clearSelection();
      }
    } else if (mode === ToolMode.Point) {
      controller.createPoint(x, y, false);
    } else if (mode === ToolMode.Line) {
      if (controller.state.creation.lineStartPoint) {
        const endPoint = (entity && isPoint(entity)) ? entity : controller.createPoint(x, y);
        controller.finishLineCreation(endPoint);
      } else {
        const startPoint = (entity && isPoint(entity)) ? entity : controller.createPoint(x, y);
        controller.startLineCreation(startPoint);
      }
    } else if (mode === ToolMode.Circle) {
      if (controller.state.creation.circleCenterPoint) {
        const dx = x - controller.state.creation.circleCenterPoint.x;
        const dy = y - controller.state.creation.circleCenterPoint.y;
        const radius = Math.sqrt(dx * dx + dy * dy);
        controller.finishCircleCreation(radius);
      } else {
        const centerPoint = (entity && isPoint(entity)) ? entity : controller.createPoint(x, y);
        controller.startCircleCreation(centerPoint);
      }
    }

    forceUpdate({});
  };

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const rect = canvasRef.current!.getBoundingClientRect();
    let x = e.clientX - rect.left;
    let y = e.clientY - rect.top;

    // Apply snapping unless Shift is held
    const enableSnap = !e.shiftKey;
    const snapped = enableSnap ? getSnappedPosition(x, y) : { x, y };

    setMousePos(snapped);
    controller.updatePreviewPosition(snapped.x, snapped.y);

    if (isDragging && draggedPoint) {
      controller.updatePointPosition(draggedPoint, snapped.x, snapped.y);
      forceUpdate({});
    } else {
      const entity = controller.findEntityAt(x, y);
      controller.setHoveredEntity(entity);
      forceUpdate({});
    }
  };

  const handleMouseUp = () => {
    setIsDragging(false);
    setDraggedPoint(null);
  };

  const handleMouseLeave = () => {
    controller.clearPreview();
    controller.setHoveredEntity(null);
    setIsDragging(false);
    setDraggedPoint(null);
    forceUpdate({});
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100vh' }}>
      {/* Header */}
      <div style={{
        background: '#2c3e50',
        color: 'white',
        padding: '12px 20px',
        display: 'flex',
        alignItems: 'center',
        gap: '12px',
        boxShadow: '0 2px 4px rgba(0,0,0,0.1)'
      }}>
        <i className="fa-solid fa-ruler-combined"></i>
        <h1 style={{ fontSize: '18px', fontWeight: 500, margin: 0 }}>Sketch Solver Demo</h1>
      </div>

      {/* Toolbar */}
      <div style={{
        background: 'white',
        borderBottom: '1px solid #ddd',
        padding: '8px 12px',
        display: 'flex',
        gap: '4px'
      }}>
        {/* Drawing Tools */}
        <button
          className={`tool-btn ${controller.getToolMode() === ToolMode.Select ? 'active' : ''}`}
          onClick={() => { controller.setToolMode(ToolMode.Select); forceUpdate({}); }}
          title="Select (V)"
        >
          <i className="fa-solid fa-arrow-pointer"></i>
        </button>
        <button
          className={`tool-btn ${controller.getToolMode() === ToolMode.Point ? 'active' : ''}`}
          onClick={() => { controller.setToolMode(ToolMode.Point); forceUpdate({}); }}
          title="Point (P)"
        >
          <i className="fa-solid fa-circle"></i>
        </button>
        <button
          className={`tool-btn ${controller.getToolMode() === ToolMode.Line ? 'active' : ''}`}
          onClick={() => { controller.setToolMode(ToolMode.Line); forceUpdate({}); }}
          title="Line (L)"
        >
          <i className="fa-solid fa-minus"></i>
        </button>
        <button
          className={`tool-btn ${controller.getToolMode() === ToolMode.Circle ? 'active' : ''}`}
          onClick={() => { controller.setToolMode(ToolMode.Circle); forceUpdate({}); }}
          title="Circle (C)"
        >
          <i className="fa-regular fa-circle"></i>
        </button>

        {/* Divider */}
        <div style={{ width: '1px', height: '24px', background: '#ddd', margin: '0 8px' }}></div>

        {/* Constraint Tools */}
        {(() => {
          const selected = controller.getSelectedEntities();
          const selectedLines = selected.filter(isLine);
          const selectedPoints = selected.filter(isPoint);
          const selectedCircles = selected.filter(isCircle);

          return (
            <>
              <button
                className="tool-btn"
                disabled={selectedLines.length !== 2}
                onClick={() => {
                  if (selectedLines.length === 2) {
                    controller.addConstraint({
                      type: ConstraintType.Parallel,
                      line1: selectedLines[0],
                      line2: selectedLines[1],
                    });
                    forceUpdate({});
                  }
                }}
                title="Parallel (requires 2 lines)"
              >
                <i className="fa-solid fa-equals"></i>
              </button>
              <button
                className="tool-btn"
                disabled={selectedLines.length !== 2}
                onClick={() => {
                  if (selectedLines.length === 2) {
                    controller.addConstraint({
                      type: ConstraintType.Perpendicular,
                      line1: selectedLines[0],
                      line2: selectedLines[1],
                    });
                    forceUpdate({});
                  }
                }}
                title="Perpendicular (requires 2 lines)"
              >
                <i className="fa-solid fa-plus" style={{ transform: 'rotate(45deg)' }}></i>
              </button>
              <button
                className="tool-btn"
                disabled={selectedLines.length !== 2}
                onClick={() => {
                  if (selectedLines.length === 2) {
                    const angle = prompt('Enter angle in degrees:', '90');
                    if (angle) {
                      controller.addConstraint({
                        type: ConstraintType.Angle,
                        line1: selectedLines[0],
                        line2: selectedLines[1],
                        angleDegrees: parseFloat(angle),
                      });
                      forceUpdate({});
                    }
                  }
                }}
                title="Angle (requires 2 lines)"
              >
                <i className="fa-solid fa-angle-right"></i>
              </button>
              <button
                className="tool-btn"
                disabled={selectedPoints.length !== 1 || selectedLines.length !== 1}
                onClick={() => {
                  if (selectedPoints.length === 1 && selectedLines.length === 1) {
                    controller.addConstraint({
                      type: ConstraintType.PointOnLine,
                      point: selectedPoints[0],
                      line: selectedLines[0],
                    });
                    forceUpdate({});
                  }
                }}
                title="Point on Line"
              >
                <i className="fa-solid fa-circle-dot"></i>
              </button>
              <button
                className="tool-btn"
                disabled={selectedPoints.length !== 1 || selectedCircles.length !== 1}
                onClick={() => {
                  if (selectedPoints.length === 1 && selectedCircles.length === 1) {
                    controller.addConstraint({
                      type: ConstraintType.PointOnCircle,
                      point: selectedPoints[0],
                      circle: selectedCircles[0],
                    });
                    forceUpdate({});
                  }
                }}
                title="Point on Circle"
              >
                <i className="fa-solid fa-record-vinyl"></i>
              </button>
              <button
                className="tool-btn"
                disabled={selectedLines.length !== 2}
                onClick={() => {
                  if (selectedLines.length === 2) {
                    controller.addConstraint({
                      type: ConstraintType.Collinear,
                      line1: selectedLines[0],
                      line2: selectedLines[1],
                    });
                    forceUpdate({});
                  }
                }}
                title="Collinear"
              >
                <i className="fa-solid fa-ellipsis"></i>
              </button>
              <button
                className="tool-btn"
                disabled={selectedLines.length !== 1 || selectedCircles.length !== 1}
                onClick={() => {
                  if (selectedLines.length === 1 && selectedCircles.length === 1) {
                    controller.addConstraint({
                      type: ConstraintType.Tangent,
                      line: selectedLines[0],
                      circle: selectedCircles[0],
                    });
                    forceUpdate({});
                  }
                }}
                title="Tangent"
              >
                <i className="fa-solid fa-hand-point-up"></i>
              </button>
              <button
                className="tool-btn"
                disabled={selectedPoints.length !== 2 || selectedCircles.length !== 1}
                onClick={() => {
                  if (selectedPoints.length === 2 && selectedCircles.length === 1) {
                    controller.addConstraint({
                      type: ConstraintType.RadialAlignment,
                      point1: selectedPoints[0],
                      point2: selectedPoints[1],
                      circle: selectedCircles[0],
                    });
                    forceUpdate({});
                  }
                }}
                title="Radial Alignment"
              >
                <i className="fa-solid fa-bullseye"></i>
              </button>
            </>
          );
        })()}
      </div>

      {/* Main Content */}
      <div style={{ flex: 1, display: 'flex', overflow: 'hidden' }}>
        <canvas
          ref={canvasRef}
          style={{ flex: 1, cursor: 'crosshair' }}
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onMouseLeave={handleMouseLeave}
        />
      </div>

      {/* Status Bar */}
      <div style={{
        background: '#2c3e50',
        color: 'white',
        padding: '8px 20px',
        display: 'flex',
        alignItems: 'center',
        gap: '24px',
        fontSize: '13px',
        borderTop: '1px solid #34495e'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <i className={`fa-solid ${controller.state.solver.isConverged ? 'fa-circle-check' : 'fa-circle-xmark'}`}
             style={{ color: controller.state.solver.isConverged ? '#2ecc71' : '#e74c3c' }}></i>
          <span>{controller.state.solver.isConverged ? 'Converged' : 'Not Converged'}</span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <i className="fa-solid fa-wave-square"></i>
          <span>Residual: {controller.state.solver.residual.toExponential(2)}</span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <i className="fa-solid fa-rotate"></i>
          <span>Iterations: {controller.state.solver.iterations}</span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginLeft: 'auto' }}>
          <i className="fa-solid fa-crosshairs"></i>
          <span>({Math.round(mousePos.x)}, {Math.round(mousePos.y)})</span>
        </div>
      </div>
    </div>
  );
}

export default App;
