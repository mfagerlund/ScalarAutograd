/**
 * ConstraintToolbar - Context-aware constraint creation panel
 * Shows available constraints based on current selection
 */

import React from 'react';
import type { SketchController } from '../SketchController';
import { ConstraintType } from '../types/Constraints';
import type { Circle, Entity, Line, Point } from '../types/Entities';
import { isCircle, isLine, isPoint, LineConstraintType } from '../types/Entities';

interface ConstraintToolbarProps {
  controller: SketchController;
  selectedEntities: Entity[];
  onConstraintAdded?: () => void;
}

export const ConstraintToolbar: React.FC<ConstraintToolbarProps> = ({
  controller,
  selectedEntities,
  onConstraintAdded,
}) => {
  // Categorize selected entities
  const selectedPoints = selectedEntities.filter(isPoint) as Point[];
  const selectedLines = selectedEntities.filter(isLine) as Line[];
  const selectedCircles = selectedEntities.filter(isCircle) as Circle[];

  /**
   * Add a constraint and notify parent
   */
  const addConstraint = (constraint: any) => {
    controller.addConstraint(constraint);
    onConstraintAdded?.();
  };

  /**
   * Render constraint button
   */
  const ConstraintButton: React.FC<{
    label: string;
    onClick: () => void;
    description?: string;
  }> = ({ label, onClick, description }) => (
    <button
      onClick={onClick}
      className="constraint-btn"
      title={description}
    >
      {label}
    </button>
  );

  // No selection - show help text
  if (selectedEntities.length === 0) {
    return (
      <div className="constraint-toolbar">
        <h3>Constraints</h3>
        <p className="help-text">
          Select entities to add constraints
        </p>
      </div>
    );
  }

  return (
    <div className="constraint-toolbar">
      <h3>Constraints</h3>
      <div className="selection-info">
        {selectedPoints.length > 0 && <span>{selectedPoints.length} point{selectedPoints.length > 1 ? 's' : ''}</span>}
        {selectedLines.length > 0 && <span>{selectedLines.length} line{selectedLines.length > 1 ? 's' : ''}</span>}
        {selectedCircles.length > 0 && <span>{selectedCircles.length} circle{selectedCircles.length > 1 ? 's' : ''}</span>}
      </div>

      <div className="constraint-buttons">
        {/* Fusion 360 Order: Coincident, Collinear, Concentric, Midpoint, Fix/Unfix, Equal, Parallel, Perpendicular, Horizontal/Vertical, Tangent, Curvature, Smooth, Symmetry */}

        {/* COINCIDENT - Point + Point, Point + Line, Point + Circle */}
        {selectedPoints.length === 2 && selectedLines.length === 0 && selectedCircles.length === 0 && (
          <ConstraintButton
            label="⚬ Coincident"
            description="Make points share same position"
            onClick={() => {
              addConstraint({
                type: ConstraintType.Coincident,
                point1: selectedPoints[0],
                point2: selectedPoints[1],
              });
            }}
          />
        )}

        {selectedPoints.length === 1 && selectedLines.length === 1 && selectedCircles.length === 0 && (
          <>
            <ConstraintButton
              label="⚬ Coincident"
              description="Constrain point to lie on line"
              onClick={() => {
                addConstraint({
                  type: ConstraintType.PointOnLine,
                  point: selectedPoints[0],
                  line: selectedLines[0],
                });
              }}
            />
            <ConstraintButton
              label="⊣ Midpoint"
              description="Make point the midpoint of line"
              onClick={() => {
                addConstraint({
                  type: ConstraintType.Midpoint,
                  point: selectedPoints[0],
                  line: selectedLines[0],
                });
              }}
            />
          </>
        )}

        {selectedPoints.length === 1 && selectedCircles.length === 1 && selectedLines.length === 0 && (
          <ConstraintButton
            label="⚬ Coincident"
            description="Constrain point to lie on circle"
            onClick={() => {
              addConstraint({
                type: ConstraintType.PointOnCircle,
                point: selectedPoints[0],
                circle: selectedCircles[0],
              });
            }}
          />
        )}

        {/* COLLINEAR - 2 Lines */}
        {selectedLines.length === 2 && selectedCircles.length === 0 && (
          <ConstraintButton
            label="⫽ Collinear"
            description="Make lines collinear"
            onClick={() => {
              addConstraint({
                type: ConstraintType.Collinear,
                line1: selectedLines[0],
                line2: selectedLines[1],
              });
            }}
          />
        )}

        {/* CONCENTRIC - 2 Circles */}
        {selectedCircles.length === 2 && selectedLines.length === 0 && (
          <ConstraintButton
            label="◎ Concentric"
            description="Make circles share same center"
            onClick={() => {
              addConstraint({
                type: ConstraintType.Concentric,
                circle1: selectedCircles[0],
                circle2: selectedCircles[1],
              });
            }}
          />
        )}

        {/* EQUAL - Multiple Lines or Multiple Circles */}
        {selectedLines.length >= 2 && selectedCircles.length === 0 && (
          <ConstraintButton
            label="= Equal"
            description="Make all selected lines equal length"
            onClick={() => {
              addConstraint({
                type: ConstraintType.EqualLength,
                lines: selectedLines,
              });
            }}
          />
        )}

        {selectedCircles.length >= 2 && selectedLines.length === 0 && (
          <ConstraintButton
            label="= Equal"
            description="Make all selected circles equal radius"
            onClick={() => {
              addConstraint({
                type: ConstraintType.EqualRadius,
                circles: selectedCircles,
              });
            }}
          />
        )}

        {/* PARALLEL - 2 Lines */}
        {selectedLines.length === 2 && selectedCircles.length === 0 && (
          <ConstraintButton
            label="∥ Parallel"
            description="Make lines parallel"
            onClick={() => {
              addConstraint({
                type: ConstraintType.Parallel,
                line1: selectedLines[0],
                line2: selectedLines[1],
              });
            }}
          />
        )}

        {/* PERPENDICULAR - 2 Lines */}
        {selectedLines.length === 2 && selectedCircles.length === 0 && (
          <ConstraintButton
            label="⊥ Perpendicular"
            description="Make lines perpendicular"
            onClick={() => {
              addConstraint({
                type: ConstraintType.Perpendicular,
                line1: selectedLines[0],
                line2: selectedLines[1],
              });
            }}
          />
        )}

        {/* HORIZONTAL/VERTICAL - Single Line */}
        {selectedLines.length === 1 && selectedPoints.length === 0 && selectedCircles.length === 0 && (
          <ConstraintButton
            label="⫿ Horizontal/Vertical"
            description="Make line horizontal or vertical"
            onClick={() => {
              const line = selectedLines[0];
              const dx = Math.abs(line.end.x - line.start.x);
              const dy = Math.abs(line.end.y - line.start.y);

              // Auto-detect which constraint to apply based on current orientation
              const shouldBeHorizontal = dx > dy;
              line.constraintType = shouldBeHorizontal ? LineConstraintType.Horizontal : LineConstraintType.Vertical;
              onConstraintAdded?.();
            }}
          />
        )}

        {/* TANGENT - Line + Circle OR Circle + Circle */}
        {selectedLines.length === 1 && selectedCircles.length === 1 && (
          <ConstraintButton
            label="⟂ Tangent"
            description="Make line tangent to circle"
            onClick={() => {
              addConstraint({
                type: ConstraintType.Tangent,
                line: selectedLines[0],
                circle: selectedCircles[0],
              });
            }}
          />
        )}

        {selectedCircles.length === 2 && selectedLines.length === 0 && (
          <ConstraintButton
            label="⟂ Tangent"
            description="Make circles tangent to each other"
            onClick={() => {
              addConstraint({
                type: ConstraintType.CirclesTangent,
                circle1: selectedCircles[0],
                circle2: selectedCircles[1],
              });
            }}
          />
        )}

        {/* SYMMETRY - 2 Points + Line */}
        {selectedPoints.length === 2 && selectedLines.length === 1 && selectedCircles.length === 0 && (
          <ConstraintButton
            label="⇌ Symmetry"
            description="Make points symmetric about line"
            onClick={() => {
              addConstraint({
                type: ConstraintType.Symmetry,
                point1: selectedPoints[0],
                point2: selectedPoints[1],
                symmetryLine: selectedLines[0],
              });
            }}
          />
        )}

        {/* ANGLE - 2 Lines (at end since less common) */}
        {selectedLines.length === 2 && selectedCircles.length === 0 && (
          <ConstraintButton
            label="∠ Angle..."
            description="Set angle between lines"
            onClick={() => {
              const angleStr = prompt('Enter angle in degrees:', '90');
              if (angleStr) {
                const angle = parseFloat(angleStr);
                if (!isNaN(angle)) {
                  addConstraint({
                    type: ConstraintType.Angle,
                    line1: selectedLines[0],
                    line2: selectedLines[1],
                    angleDegrees: angle,
                  });
                }
              }
            }}
          />
        )}
      </div>

      {/* Show help text if no constraints are available */}
      {selectedEntities.length > 0 && (
        <div className="constraint-buttons">
          {/* Check if any constraints were rendered */}
          {document.querySelectorAll('.constraint-btn').length === 0 && (
            <p className="help-text">
              No constraints available for this selection.
              <br />
              <br />Try selecting:
              <br />• 2 points → Coincident
              <br />• Point + line → Coincident, Midpoint
              <br />• Point + circle → Coincident
              <br />• 1 line → Horizontal/Vertical
              <br />• 2 lines → Collinear, Parallel, Perpendicular, Equal, Angle
              <br />• 2+ lines → Equal
              <br />• 2 circles → Concentric, Tangent, Equal
              <br />• 2+ circles → Equal
              <br />• Line + circle → Tangent
              <br />• 2 points + line → Symmetry
            </p>
          )}
        </div>
      )}
    </div>
  );
};
