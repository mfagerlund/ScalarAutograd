/**
 * ConstraintToolbar - Context-aware constraint creation panel
 * Shows available constraints based on current selection
 */

import React from 'react';
import type { Entity, Point, Line, Circle } from '../types/Entities';
import { isPoint, isLine, isCircle } from '../types/Entities';
import type { SketchController } from '../SketchController';
import { ConstraintType } from '../types/Constraints';

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
        {/* LINE + LINE CONSTRAINTS */}
        {selectedLines.length === 2 && (
          <>
            <ConstraintButton
              label="Parallel"
              description="Make lines parallel"
              onClick={() => {
                addConstraint({
                  type: ConstraintType.Parallel,
                  line1: selectedLines[0],
                  line2: selectedLines[1],
                });
              }}
            />
            <ConstraintButton
              label="Perpendicular"
              description="Make lines perpendicular"
              onClick={() => {
                addConstraint({
                  type: ConstraintType.Perpendicular,
                  line1: selectedLines[0],
                  line2: selectedLines[1],
                });
              }}
            />
            <ConstraintButton
              label="Collinear"
              description="Make lines collinear"
              onClick={() => {
                addConstraint({
                  type: ConstraintType.Collinear,
                  line1: selectedLines[0],
                  line2: selectedLines[1],
                });
              }}
            />
            <ConstraintButton
              label="Angle..."
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
                      angle,
                    });
                  }
                }
              }}
            />
          </>
        )}

        {/* POINT + LINE CONSTRAINTS */}
        {selectedPoints.length === 1 && selectedLines.length === 1 && (
          <ConstraintButton
            label="Point on Line"
            description="Constrain point to lie on line"
            onClick={() => {
              addConstraint({
                type: ConstraintType.PointOnLine,
                point: selectedPoints[0],
                line: selectedLines[0],
              });
            }}
          />
        )}

        {/* POINT + CIRCLE CONSTRAINTS */}
        {selectedPoints.length === 1 && selectedCircles.length === 1 && (
          <ConstraintButton
            label="Point on Circle"
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

        {/* LINE + CIRCLE CONSTRAINTS */}
        {selectedLines.length === 1 && selectedCircles.length === 1 && (
          <ConstraintButton
            label="Tangent"
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

        {/* POINT + POINT + CIRCLE CONSTRAINTS (Radial Alignment) */}
        {selectedPoints.length === 2 && selectedCircles.length === 1 && (
          <ConstraintButton
            label="Radial Alignment"
            description="Align two points radially on circle"
            onClick={() => {
              addConstraint({
                type: ConstraintType.RadialAlignment,
                point1: selectedPoints[0],
                point2: selectedPoints[1],
                circle: selectedCircles[0],
              });
            }}
          />
        )}
      </div>

      {/* Show message if no constraints available for current selection */}
      {selectedLines.length !== 2 &&
        !(selectedPoints.length === 1 && selectedLines.length === 1) &&
        !(selectedPoints.length === 1 && selectedCircles.length === 1) &&
        !(selectedLines.length === 1 && selectedCircles.length === 1) &&
        !(selectedPoints.length === 2 && selectedCircles.length === 1) && (
          <p className="help-text">
            No constraints available for this selection.
            <br />
            Try selecting:
            <br />• 2 lines (parallel, perpendicular, angle)
            <br />• 1 point + 1 line (point on line)
            <br />• 1 point + 1 circle (point on circle)
            <br />• 1 line + 1 circle (tangent)
            <br />• 2 points + 1 circle (radial alignment)
          </p>
        )}
    </div>
  );
};
