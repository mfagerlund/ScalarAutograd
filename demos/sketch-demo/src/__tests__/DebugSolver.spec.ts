import { describe, it } from 'vitest';
import { SketchSolver } from '../SketchSolver';
import { testLog } from '../../../../test/testUtils';
import type {
    Line,
    Point,
    Project,
} from '../types';
import { LineConstraintType } from '../types';

describe('Debug Solver', () => {
  it('debug fixed length', () => {
    const solver = new SketchSolver({ tolerance: 1e-4, maxIterations: 200 });

    const p1: Point = { x: 0, y: 0, pinned: true };
    const p2: Point = { x: 50, y: 0 }; // Wrong length initially

    const line: Line = {
      start: p1,
      end: p2,
      constraintType: LineConstraintType.Free,
      fixedLength: 100, // Should be 100 units long
    };

    const project: Project = {
      name: 'Test',
      points: [p1, p2],
      lines: [line],
      circles: [],
      constraints: [],
    };

    const result = solver.solve(project);

    testLog('Fixed length result:', result);
    testLog('p2 after solve:', p2);
    testLog('Line length:', Math.hypot(p2.x - p1.x, p2.y - p1.y));
  });
});
