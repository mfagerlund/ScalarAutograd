import { useState, useCallback, useEffect, useRef, useMemo } from 'react';
import { DemoCanvas } from '../../components/DemoCanvas';
import type { DemoProps } from '../types';
import { AdamW, SGD, Optimizer, V, Value } from '../../../../src';

interface Point {
  x: Value;
  y: Value;
}

interface RopeState {
  points: Point[];
  iteration: number;
  loss: number;
  converged: boolean;
  pointCount: number;
}

const BOX_SIZE = 280;
const BOX_PADDING = 30;
const INITIAL_POINTS = 8;
const MAX_POINTS = 50;
const MIN_DISTANCE = 15;
const NEIGHBOR_DISTANCE = 25;
const GRID_CELL_SIZE = MIN_DISTANCE * 2;
const EPSILON = 0.01;

function createSpatialGrid(points: Point[], boxSize: number) {
  const grid = new Map<string, number[]>();
  const cellSize = GRID_CELL_SIZE;

  points.forEach((p, idx) => {
    const cellX = Math.floor(p.x.data / cellSize);
    const cellY = Math.floor(p.y.data / cellSize);
    const key = `${cellX},${cellY}`;

    if (!grid.has(key)) {
      grid.set(key, []);
    }
    grid.get(key)!.push(idx);
  });

  return { grid, cellSize };
}

function getNearbyPoints(grid: Map<string, number[]>, x: number, y: number, cellSize: number): number[] {
  const nearby: number[] = [];
  const cellX = Math.floor(x / cellSize);
  const cellY = Math.floor(y / cellSize);

  for (let dx = -1; dx <= 1; dx++) {
    for (let dy = -1; dy <= 1; dy++) {
      const key = `${cellX + dx},${cellY + dy}`;
      const cell = grid.get(key);
      if (cell) {
        nearby.push(...cell);
      }
    }
  }

  return nearby;
}

function lineSegmentDistance(
  p1x: Value, p1y: Value,
  p2x: Value, p2y: Value,
  p3x: Value, p3y: Value,
  p4x: Value, p4y: Value
): Value {
  const dx1 = V.sub(p2x, p1x);
  const dy1 = V.sub(p2y, p1y);
  const dx2 = V.sub(p4x, p3x);
  const dy2 = V.sub(p4y, p3y);

  const len1Sq = V.add(V.square(dx1), V.square(dy1));
  const len2Sq = V.add(V.square(dx2), V.square(dy2));

  const dx3 = V.sub(p3x, p1x);
  const dy3 = V.sub(p3y, p1y);

  const t1Num = V.add(V.mul(dx3, dx1), V.mul(dy3, dy1));
  const t1 = V.div(t1Num, V.add(len1Sq, V.C(EPSILON)));
  const t1Clamped = V.max(V.C(0), V.min(V.C(1), t1));

  const closest1x = V.add(p1x, V.mul(t1Clamped, dx1));
  const closest1y = V.add(p1y, V.mul(t1Clamped, dy1));

  const t2Num = V.add(V.mul(V.sub(closest1x, p3x), dx2), V.mul(V.sub(closest1y, p3y), dy2));
  const t2 = V.div(t2Num, V.add(len2Sq, V.C(EPSILON)));
  const t2Clamped = V.max(V.C(0), V.min(V.C(1), t2));

  const closest2x = V.add(p3x, V.mul(t2Clamped, dx2));
  const closest2y = V.add(p3y, V.mul(t2Clamped, dy2));

  const distSq = V.add(
    V.square(V.sub(closest1x, closest2x)),
    V.square(V.sub(closest1y, closest2y))
  );

  return V.sqrt(V.add(distSq, V.C(EPSILON)));
}

function computeLoss(points: Point[], boxSize: number): Value {
  const losses: Value[] = [];

  for (let i = 0; i < points.length; i++) {
    const next = (i + 1) % points.length;
    const dx = V.sub(points[next].x, points[i].x);
    const dy = V.sub(points[next].y, points[i].y);
    const dist = V.sqrt(V.add(V.square(dx), V.square(dy)));
    const neighborDiff = V.sub(dist, V.C(NEIGHBOR_DISTANCE));
    losses.push(V.mul(V.C(2.0), V.square(neighborDiff)));
  }

  for (let i = 0; i < points.length; i++) {
    for (let j = i + 1; j < points.length; j++) {
      const diff = Math.min(Math.abs(i - j), points.length - Math.abs(i - j));
      if (diff <= 1) continue;

      const dx = V.sub(points[i].x, points[j].x);
      const dy = V.sub(points[i].y, points[j].y);
      const distSq = V.add(V.square(dx), V.square(dy));
      const dist = V.sqrt(V.add(distSq, V.C(EPSILON)));

      const penetration = V.sub(V.C(MIN_DISTANCE), dist);
      const penalty = V.mul(V.C(5.0), V.square(V.max(V.C(0), penetration)));
      losses.push(penalty);
    }
  }

  for (let i = 0; i < points.length; i++) {
    const next_i = (i + 1) % points.length;
    for (let j = i + 2; j < points.length; j++) {
      const next_j = (j + 1) % points.length;

      const diff = Math.min(Math.abs(i - j), points.length - Math.abs(i - j));
      if (diff <= 1) continue;

      const segDist = lineSegmentDistance(
        points[i].x, points[i].y,
        points[next_i].x, points[next_i].y,
        points[j].x, points[j].y,
        points[next_j].x, points[next_j].y
      );

      const crossingPenalty = V.sub(V.C(MIN_DISTANCE * 0.8), segDist);
      const penalty = V.mul(V.C(3.0), V.square(V.max(V.C(0), crossingPenalty)));
      losses.push(penalty);
    }
  }

  for (let i = 0; i < points.length; i++) {
    const xMin = V.max(V.C(0), V.sub(V.C(5), points[i].x));
    const xMax = V.max(V.C(0), V.sub(points[i].x, V.C(boxSize - 5)));
    const yMin = V.max(V.C(0), V.sub(V.C(5), points[i].y));
    const yMax = V.max(V.C(0), V.sub(points[i].y, V.C(boxSize - 5)));

    losses.push(V.mul(V.C(10.0), V.square(xMin)));
    losses.push(V.mul(V.C(10.0), V.square(xMax)));
    losses.push(V.mul(V.C(10.0), V.square(yMin)));
    losses.push(V.mul(V.C(10.0), V.square(yMax)));
  }

  return V.mean(losses);
}

function createInitialPoints(count: number, boxSize: number, prefix: string, radius: number): Point[] {
  const points: Point[] = [];
  const centerX = boxSize / 2;
  const centerY = boxSize / 2;

  for (let i = 0; i < count; i++) {
    const t = i / count;
    const angle = t * Math.PI * 2;

    points.push({
      x: V.W(centerX + Math.cos(angle) * radius, `${prefix}_x_${i}`),
      y: V.W(centerY + Math.sin(angle) * radius, `${prefix}_y_${i}`)
    });
  }

  return points;
}

function addPointToRope(points: Point[], boxSize: number, prefix: string): Point[] {
  if (points.length >= MAX_POINTS) return points;

  const insertIdx = Math.floor(Math.random() * points.length);

  const p1 = points[insertIdx];
  const p2 = points[(insertIdx + 1) % points.length];
  const midX = (p1.x.data + p2.x.data) / 2;
  const midY = (p1.y.data + p2.y.data) / 2;

  const newPoints = [...points];
  newPoints.splice(insertIdx + 1, 0, {
    x: V.W(midX, `${prefix}_x_${points.length}`),
    y: V.W(midY, `${prefix}_y_${points.length}`)
  });

  return newPoints;
}

export function SelfAvoidingRope({ width, height, onMetrics }: DemoProps) {
  const [adamState, setAdamState] = useState<RopeState | null>(null);
  const [nlsState, setNlsState] = useState<RopeState | null>(null);
  const [nlsStepSize, setNlsStepSize] = useState(0.3);
  const [initialRadius, setInitialRadius] = useState(50);
  const [addSpeed, setAddSpeed] = useState(1.0);
  const [optimizerType, setOptimizerType] = useState<'SGD' | 'Adam'>('Adam');
  const [sgdLearningRate, setSgdLearningRate] = useState(0.5);

  const adamStateRef = useRef<RopeState | null>(null);
  const nlsStateRef = useRef<RopeState | null>(null);
  const adamOptimizerRef = useRef<Optimizer | null>(null);
  const frameCountRef = useRef(0);
  const animationRef = useRef<number>();
  const nlsStepSizeRef = useRef(nlsStepSize);
  const addSpeedRef = useRef(addSpeed);
  const optimizerTypeRef = useRef(optimizerType);
  const sgdLearningRateRef = useRef(sgdLearningRate);

  useEffect(() => {
    nlsStepSizeRef.current = nlsStepSize;
  }, [nlsStepSize]);

  useEffect(() => {
    addSpeedRef.current = addSpeed;
  }, [addSpeed]);

  useEffect(() => {
    optimizerTypeRef.current = optimizerType;
  }, [optimizerType]);

  useEffect(() => {
    sgdLearningRateRef.current = sgdLearningRate;
  }, [sgdLearningRate]);

  const initializeStates = useCallback(() => {
    const adamPoints = createInitialPoints(INITIAL_POINTS, BOX_SIZE, 'adam', initialRadius);
    const nlsPoints = createInitialPoints(INITIAL_POINTS, BOX_SIZE, 'nls', initialRadius);

    const newAdamState: RopeState = {
      points: adamPoints,
      iteration: 0,
      loss: 0,
      converged: false,
      pointCount: INITIAL_POINTS
    };

    const newNlsState: RopeState = {
      points: nlsPoints,
      iteration: 0,
      loss: 0,
      converged: false,
      pointCount: INITIAL_POINTS
    };

    setAdamState(newAdamState);
    setNlsState(newNlsState);
    adamStateRef.current = newAdamState;
    nlsStateRef.current = newNlsState;

    const allParams = [...adamPoints.map(p => p.x), ...adamPoints.map(p => p.y)];
    if (optimizerType === 'Adam') {
      adamOptimizerRef.current = new AdamW(allParams, { learningRate: 0.01 });
    } else {
      adamOptimizerRef.current = new SGD(allParams, { learningRate: sgdLearningRate });
    }
    frameCountRef.current = 0;
  }, [initialRadius, optimizerType, sgdLearningRate]);

  useEffect(() => {
    initializeStates();
  }, [initializeStates]);

  useEffect(() => {
    const animate = () => {
      const adamState = adamStateRef.current;

      if (adamState) {
        for (let step = 0; step < 3; step++) {
          const loss = computeLoss(adamState.points, BOX_SIZE);

          if (isNaN(loss.data) || !isFinite(loss.data)) {
            console.error('Loss is NaN or infinite, resetting');
            initializeStates();
            return;
          }

          adamOptimizerRef.current!.zeroGrad();
          loss.backward();
          adamOptimizerRef.current!.step();

          adamState.iteration++;
          adamState.loss = loss.data;
          adamState.converged = loss.data < 0.01;
        }

        setAdamState({ ...adamState });
      }

      const nlsState = nlsStateRef.current;
      if (nlsState) {
        function residuals(params: Value[]) {
          const n = params.length / 2;
          const points: Point[] = [];
          for (let i = 0; i < n; i++) {
            points.push({ x: params[i], y: params[n + i] });
          }

          const losses: Value[] = [];

          for (let i = 0; i < points.length; i++) {
            const next = (i + 1) % points.length;
            const dx = V.sub(points[next].x, points[i].x);
            const dy = V.sub(points[next].y, points[i].y);
            const dist = V.sqrt(V.add(V.square(dx), V.square(dy)));
            const neighborDiff = V.sub(dist, V.C(NEIGHBOR_DISTANCE));
            losses.push(V.mul(V.C(Math.SQRT2), neighborDiff));
          }

          for (let i = 0; i < points.length; i++) {
            for (let j = i + 1; j < points.length; j++) {
              const diff = Math.min(Math.abs(i - j), points.length - Math.abs(i - j));
              if (diff <= 1) continue;

              const dx = V.sub(points[i].x, points[j].x);
              const dy = V.sub(points[i].y, points[j].y);
              const distSq = V.add(V.square(dx), V.square(dy));
              const dist = V.sqrt(V.add(distSq, V.C(EPSILON)));

              const penetration = V.sub(V.C(MIN_DISTANCE), dist);
              const penalty = V.mul(V.C(Math.sqrt(5.0)), V.max(V.C(0), penetration));
              losses.push(penalty);
            }
          }

          for (let i = 0; i < points.length; i++) {
            const next_i = (i + 1) % points.length;
            for (let j = i + 2; j < points.length; j++) {
              const next_j = (j + 1) % points.length;

              const diff = Math.min(Math.abs(i - j), points.length - Math.abs(i - j));
              if (diff <= 1) continue;

              const segDist = lineSegmentDistance(
                points[i].x, points[i].y,
                points[next_i].x, points[next_i].y,
                points[j].x, points[j].y,
                points[next_j].x, points[next_j].y
              );

              const crossingPenalty = V.sub(V.C(MIN_DISTANCE * 0.8), segDist);
              const penalty = V.mul(V.C(Math.sqrt(3.0)), V.max(V.C(0), crossingPenalty));
              losses.push(penalty);
            }
          }

          for (let i = 0; i < points.length; i++) {
            const xMin = V.mul(V.C(Math.sqrt(10.0)), V.max(V.C(0), V.sub(V.C(5), points[i].x)));
            const xMax = V.mul(V.C(Math.sqrt(10.0)), V.max(V.C(0), V.sub(points[i].x, V.C(BOX_SIZE - 5))));
            const yMin = V.mul(V.C(Math.sqrt(10.0)), V.max(V.C(0), V.sub(V.C(5), points[i].y)));
            const yMax = V.mul(V.C(Math.sqrt(10.0)), V.max(V.C(0), V.sub(points[i].y, V.C(BOX_SIZE - 5))));

            losses.push(xMin);
            losses.push(xMax);
            losses.push(yMin);
            losses.push(yMax);
          }

          return losses;
        }

        const allParams = [...nlsState.points.map(p => p.x), ...nlsState.points.map(p => p.y)];
        const oldValues = allParams.map(p => p.data);

        const result = V.nonlinearLeastSquares(allParams, residuals, {
          maxIterations: 200,
          costTolerance: 1e-8,
          paramTolerance: 1e-8,
          gradientTolerance: 1e-8
        });

        const stepSize = nlsStepSizeRef.current;
        let totalChange = 0;
        for (let i = 0; i < allParams.length; i++) {
          const newValue = allParams[i].data;
          const interpolated = oldValues[i] + (newValue - oldValues[i]) * stepSize;
          totalChange += Math.abs(interpolated - oldValues[i]);
          allParams[i].data = interpolated;
        }

        if (frameCountRef.current % 60 === 0) {
          console.log(`NLS: iter=${result.iterations}, loss=${result.finalCost.toFixed(6)}, change=${totalChange.toFixed(4)}, stepSize=${stepSize}`);
        }

        nlsState.iteration += result.iterations;
        nlsState.loss = result.finalCost;
        nlsState.converged = result.success;

        nlsStateRef.current = nlsState;
        setNlsState({ ...nlsState });
      }

      frameCountRef.current++;

      const pointsAddInterval = Math.round(60 * addSpeedRef.current);
      if (frameCountRef.current % pointsAddInterval === 0) {
        if (adamState && adamState.points.length < MAX_POINTS) {
          const newAdamPoints = addPointToRope(adamState.points, BOX_SIZE, 'adam');
          adamState.points = newAdamPoints;
          adamState.pointCount = newAdamPoints.length;

          const allParams = [...newAdamPoints.map(p => p.x), ...newAdamPoints.map(p => p.y)];
          if (optimizerTypeRef.current === 'Adam') {
            adamOptimizerRef.current = new AdamW(allParams, { learningRate: 0.01 });
          } else {
            adamOptimizerRef.current = new SGD(allParams, { learningRate: sgdLearningRateRef.current });
          }

          adamStateRef.current = adamState;
          setAdamState({ ...adamState });
        }

        if (nlsState && nlsState.points.length < MAX_POINTS) {
          const newNlsPoints = addPointToRope(nlsState.points, BOX_SIZE, 'nls');
          nlsState.points = newNlsPoints;
          nlsState.pointCount = newNlsPoints.length;

          nlsStateRef.current = nlsState;
          setNlsState({ ...nlsState });
        }
      }

      animationRef.current = requestAnimationFrame(animate);
    };

    animationRef.current = requestAnimationFrame(animate);

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, []);

  const render = useCallback((ctx: CanvasRenderingContext2D) => {
    ctx.fillStyle = '#0f172a';
    ctx.fillRect(0, 0, width, height);

    const drawRope = (state: RopeState | null, offsetX: number, color: string, label: string) => {
      if (!state) return;

      ctx.save();
      ctx.translate(offsetX, BOX_PADDING);

      ctx.strokeStyle = '#1e293b';
      ctx.lineWidth = 2;
      ctx.strokeRect(0, 0, BOX_SIZE, BOX_SIZE);

      ctx.fillStyle = color;
      ctx.font = '14px sans-serif';
      ctx.fillText(label, 10, -10);

      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.lineCap = 'round';
      ctx.lineJoin = 'round';
      ctx.globalAlpha = 0.6;

      ctx.beginPath();
      state.points.forEach((p, i) => {
        if (i === 0) {
          ctx.moveTo(p.x.data, p.y.data);
        } else {
          ctx.lineTo(p.x.data, p.y.data);
        }
      });
      ctx.closePath();
      ctx.stroke();

      ctx.globalAlpha = 1;

      state.points.forEach((p, i) => {
        ctx.beginPath();
        ctx.arc(p.x.data, p.y.data, 4, 0, Math.PI * 2);
        ctx.fillStyle = color;
        ctx.fill();
      });

      ctx.fillStyle = '#f1f5f9';
      ctx.font = '11px monospace';
      ctx.fillText(`Points: ${state.pointCount}`, 10, BOX_SIZE - 30);
      ctx.fillText(`Iter: ${state.iteration}`, 10, BOX_SIZE - 15);
      ctx.fillText(`Loss: ${state.loss.toFixed(4)}`, 10, BOX_SIZE - 0);

      ctx.restore();
    };

    const leftX = (width - BOX_SIZE * 2 - 40) / 2;
    const rightX = leftX + BOX_SIZE + 40;

    drawRope(nlsState, leftX, '#f59e0b', 'NLS');
    drawRope(adamState, rightX, '#6366f1', optimizerType);

  }, [width, height, adamState, nlsState, optimizerType]);

  return (
    <div className="demo-container">
      <div className="demo-header">
        <h2>Self-Avoiding Rope</h2>
        <p style={{ fontSize: '0.875rem', color: 'var(--text)', opacity: 0.7 }}>
          A growing closed loop that avoids crossing itself.
        </p>
      </div>

      <div style={{
        display: 'grid',
        gridTemplateColumns: '1fr 1fr 1fr',
        gap: '1rem',
        marginBottom: '1rem',
        maxWidth: '800px',
        margin: '0 auto 1rem auto'
      }}>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.25rem' }}>
          <label style={{ fontSize: '0.75rem', color: 'var(--text)', opacity: 0.8 }}>
            NLS Step Size: {nlsStepSize.toFixed(2)}
          </label>
          <input
            type="range"
            min="0.05"
            max="1.0"
            step="0.05"
            value={nlsStepSize}
            onChange={(e) => setNlsStepSize(Number(e.target.value))}
            style={{ width: '100%' }}
          />
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.25rem' }}>
          <label style={{ fontSize: '0.75rem', color: 'var(--text)', opacity: 0.8 }}>
            SGD LR: {sgdLearningRate.toFixed(2)}
          </label>
          <input
            type="range"
            min="0.01"
            max="2.0"
            step="0.01"
            value={sgdLearningRate}
            onChange={(e) => setSgdLearningRate(Number(e.target.value))}
            style={{ width: '100%' }}
          />
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.25rem' }}>
          <label style={{ fontSize: '0.75rem', color: 'var(--text)', opacity: 0.8 }}>
            Add Speed: {addSpeed.toFixed(2)}s
          </label>
          <input
            type="range"
            min="0.1"
            max="1.0"
            step="0.1"
            value={addSpeed}
            onChange={(e) => setAddSpeed(Number(e.target.value))}
            style={{ width: '100%' }}
          />
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.25rem' }}>
          <label style={{ fontSize: '0.75rem', color: 'var(--text)', opacity: 0.8 }}>
            Initial Radius: {initialRadius}
          </label>
          <input
            type="range"
            min="20"
            max="80"
            step="5"
            value={initialRadius}
            onChange={(e) => setInitialRadius(Number(e.target.value))}
            style={{ width: '100%' }}
          />
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.25rem' }}>
          <label style={{ fontSize: '0.75rem', color: 'var(--text)', opacity: 0.8 }}>
            Optimizer
          </label>
          <div style={{ display: 'flex', gap: '0.5rem' }}>
            <button
              onClick={() => setOptimizerType('SGD')}
              style={{
                flex: 1,
                padding: '0.5rem',
                background: optimizerType === 'SGD' ? 'var(--primary)' : 'var(--surface)',
                color: 'white',
                border: '1px solid var(--primary)',
                borderRadius: '0.25rem',
                cursor: 'pointer',
                fontSize: '0.75rem'
              }}
            >
              SGD
            </button>
            <button
              onClick={() => setOptimizerType('Adam')}
              style={{
                flex: 1,
                padding: '0.5rem',
                background: optimizerType === 'Adam' ? 'var(--primary)' : 'var(--surface)',
                color: 'white',
                border: '1px solid var(--primary)',
                borderRadius: '0.25rem',
                cursor: 'pointer',
                fontSize: '0.75rem'
              }}
            >
              Adam
            </button>
          </div>
        </div>

        <div style={{ display: 'flex', alignItems: 'flex-end' }}>
          <button
            onClick={initializeStates}
            style={{
              padding: '0.5rem 1rem',
              background: 'var(--primary)',
              color: 'white',
              border: 'none',
              borderRadius: '0.25rem',
              cursor: 'pointer',
              fontSize: '0.875rem',
              width: '100%'
            }}
          >
            Reset
          </button>
        </div>
      </div>

      <DemoCanvas width={width} height={height} onRender={render} />
    </div>
  );
}
