import { useState, useCallback, useEffect, useRef } from 'react';
import { DemoCanvas } from '../../components/DemoCanvas';
import { DemoControls } from '../../components/DemoControls';
import { MetricsDisplay } from '../../components/MetricsDisplay';
import type { OptimizationMetrics } from '../../components/MetricsDisplay';
import type { DemoProps } from '../types';
import { V } from '../../../../V';
import { SGD, Adam, AdamW, Optimizer } from '../../../../Optimizers';

interface Point {
  x: ReturnType<typeof V.W>;
  y: ReturnType<typeof V.W>;
}

interface AnchorPoint {
  index: number;
  trueX: number;
  trueY: number;
}

const NUM_POINTS = 20;
const TARGET_DISTANCE = 50; // Distance constraint between adjacent points
const ANCHOR_INDICES = [0, 7, 14]; // Three anchor points

const ANCHOR_HOMES: AnchorPoint[] = ANCHOR_INDICES.map(i => {
  const angle = (i / NUM_POINTS) * 2 * Math.PI;
  return {
    index: i,
    trueX: 150 * Math.cos(angle),
    trueY: 150 * Math.sin(angle),
  };
});

function createPoints(): Point[] {
  // Randomize positions each time to avoid local minima
  return Array.from({ length: NUM_POINTS }, (_, i) => ({
    x: V.W((Math.sin(i * 2.5) * 150) + (Math.random() - 0.5) * 100, `x${i}`),
    y: V.W((Math.cos(i * 2.5) * 150) + (Math.random() - 0.5) * 100, `y${i}`),
  }));
}

function computeLoss(points: Point[], anchors: AnchorPoint[]) {
  const losses = [];

  // 1. Equal distance constraint between adjacent points (n and n+1)
  for (let i = 0; i < points.length; i++) {
    const p1 = points[i];
    const p2 = points[(i + 1) % points.length];
    const dx = V.sub(p1.x, p2.x);
    const dy = V.sub(p1.y, p2.y);
    const dist = V.sqrt(V.add(V.square(dx), V.square(dy)));
    const error = V.sub(dist, TARGET_DISTANCE);
    losses.push(V.square(error));
  }

  // 2. Distance to opposite point constraint
  const expectedRadius = (NUM_POINTS * TARGET_DISTANCE) / (2 * Math.PI);
  const oppositeSteps = Math.floor(NUM_POINTS / 2);
  const oppositeAngle = (Math.PI * oppositeSteps) / NUM_POINTS;
  const expectedOppositeDistance = 2 * expectedRadius * Math.sin(oppositeAngle);

  for (let i = 0; i < points.length; i++) {
    const p1 = points[i];
    const p2 = points[(i + oppositeSteps) % points.length];
    const dx = V.sub(p1.x, p2.x);
    const dy = V.sub(p1.y, p2.y);
    const dist = V.sqrt(V.add(V.square(dx), V.square(dy)));
    const error = V.sub(dist, expectedOppositeDistance);
    losses.push(V.square(error));
  }

  // 3. Distance from centroid constraint (small weight to help convergence)
  const cx = V.mean(points.map(p => p.x));
  const cy = V.mean(points.map(p => p.y));

  for (let i = 0; i < points.length; i++) {
    const p = points[i];
    const dx = V.sub(p.x, cx);
    const dy = V.sub(p.y, cy);
    const dist = V.sqrt(V.add(V.square(dx), V.square(dy)));
    const error = V.sub(dist, expectedRadius);
    losses.push(V.mul(V.square(error), 0.1)); // Small weight
  }

  return V.mean(losses);
}

export function CircleFormation({ width, height, onMetrics }: DemoProps) {
  const [points, setPoints] = useState<Point[]>(createPoints);
  const [isPlaying, setIsPlaying] = useState(false);
  const [speed, setSpeed] = useState(() => {
    const saved = localStorage.getItem('circleFormation_speed');
    return saved ? parseFloat(saved) : 1;
  });
  const [iteration, setIteration] = useState(0);
  const [loss, setLoss] = useState(0);
  const [optimizer, setOptimizer] = useState<Optimizer | null>(null);
  const [optimizerType, setOptimizerType] = useState<'sgd' | 'adam' | 'adamw' | 'lm'>('lm');

  // Learning rates from localStorage
  const [sgdLR, setSgdLR] = useState(() => {
    const saved = localStorage.getItem('circleFormation_sgdLR');
    return saved ? parseFloat(saved) : 5.0;
  });
  const [adamLR, setAdamLR] = useState(() => {
    const saved = localStorage.getItem('circleFormation_adamLR');
    return saved ? parseFloat(saved) : 1.5;
  });
  const [adamwLR, setAdamwLR] = useState(() => {
    const saved = localStorage.getItem('circleFormation_adamwLR');
    return saved ? parseFloat(saved) : 2.5;
  });

  const animationRef = useRef<number>();

  const centerX = width / 2;
  const centerY = height / 2;

  // Save settings to localStorage
  useEffect(() => {
    localStorage.setItem('circleFormation_speed', speed.toString());
  }, [speed]);

  useEffect(() => {
    localStorage.setItem('circleFormation_sgdLR', sgdLR.toString());
  }, [sgdLR]);

  useEffect(() => {
    localStorage.setItem('circleFormation_adamLR', adamLR.toString());
  }, [adamLR]);

  useEffect(() => {
    localStorage.setItem('circleFormation_adamwLR', adamwLR.toString());
  }, [adamwLR]);

  // Initialize optimizer
  useEffect(() => {
    if (optimizerType === 'lm') {
      setOptimizer(null); // LM doesn't use iterative optimizer
      return;
    }

    const params = points.flatMap(p => [p.x, p.y]);
    let opt: Optimizer;

    switch (optimizerType) {
      case 'sgd':
        opt = new SGD(params, { learningRate: sgdLR });
        break;
      case 'adam':
        opt = new Adam(params, { learningRate: adamLR });
        break;
      case 'adamw':
        opt = new AdamW(params, { learningRate: adamwLR, weightDecay: 0.001 });
        break;
    }

    setOptimizer(opt);
  }, [optimizerType, points, sgdLR, adamLR, adamwLR]);

  // Run LM optimization
  const runLM = useCallback(() => {
    setIsPlaying(true);

    // Compute residuals for LM
    const computeResiduals = (params: ReturnType<typeof V.W>[]) => {
      const residuals = [];

      // Reconstruct points from params
      const pts: Point[] = [];
      for (let i = 0; i < NUM_POINTS; i++) {
        pts.push({
          x: params[i * 2],
          y: params[i * 2 + 1],
        });
      }

      // 1. Equal distance constraints (n and n+1)
      for (let i = 0; i < pts.length; i++) {
        const p1 = pts[i];
        const p2 = pts[(i + 1) % pts.length];
        const dx = V.sub(p1.x, p2.x);
        const dy = V.sub(p1.y, p2.y);
        const dist = V.sqrt(V.add(V.square(dx), V.square(dy)));
        residuals.push(V.sub(dist, TARGET_DISTANCE));
      }

      // 2. Distance to opposite point constraint
      const expectedRadius = (NUM_POINTS * TARGET_DISTANCE) / (2 * Math.PI);
      const oppositeSteps = Math.floor(NUM_POINTS / 2);
      const oppositeAngle = (Math.PI * oppositeSteps) / NUM_POINTS;
      const expectedOppositeDistance = 2 * expectedRadius * Math.sin(oppositeAngle);

      for (let i = 0; i < pts.length; i++) {
        const p1 = pts[i];
        const p2 = pts[(i + oppositeSteps) % pts.length];
        const dx = V.sub(p1.x, p2.x);
        const dy = V.sub(p1.y, p2.y);
        const dist = V.sqrt(V.add(V.square(dx), V.square(dy)));
        residuals.push(V.sub(dist, expectedOppositeDistance));
      }

      // 3. Distance from centroid constraint (single residual for radius variance)
      const cx = V.mean(pts.map(p => p.x));
      const cy = V.mean(pts.map(p => p.y));

      let radiusVariance = V.C(0);
      for (let i = 0; i < pts.length; i++) {
        const p = pts[i];
        const dx = V.sub(p.x, cx);
        const dy = V.sub(p.y, cy);
        const dist = V.sqrt(V.add(V.square(dx), V.square(dy)));
        const error = V.sub(dist, expectedRadius);
        radiusVariance = V.add(radiusVariance, V.square(error));
      }
      const rmsRadiusError = V.sqrt(V.div(radiusVariance, NUM_POINTS));
      residuals.push(V.mul(rmsRadiusError, Math.sqrt(0.1))); // Single residual

      return residuals;
    };

    const params = points.flatMap(p => [p.x, p.y]);

    const result = V.nonlinearLeastSquares(params, computeResiduals, {
      maxIterations: 100,
      costTolerance: 1e-6,
      verbose: true, // Enable to see what's happening
    });

    // Update points with result
    for (let i = 0; i < NUM_POINTS; i++) {
      points[i].x.data = params[i * 2].data;
      points[i].y.data = params[i * 2 + 1].data;
    }

    setPoints([...points]);
    setIteration(result.iterations);
    setLoss(result.finalCost);
    setIsPlaying(false);

    if (onMetrics) {
      onMetrics({
        loss: result.finalCost,
        iteration: result.iterations,
        converged: result.converged,
      });
    }
  }, [points, onMetrics]);

  // Handle play for LM
  useEffect(() => {
    if (isPlaying && optimizerType === 'lm') {
      runLM();
    }
  }, [isPlaying, optimizerType, runLM]);

  // Optimization loop for gradient-based optimizers
  useEffect(() => {
    if (!isPlaying || !optimizer || optimizerType === 'lm') {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
      return;
    }

    let lastTime = performance.now();

    const animate = (currentTime: number) => {
      const deltaTime = currentTime - lastTime;

      if (deltaTime >= 16) { // ~60fps
        for (let step = 0; step < speed; step++) {
          optimizer.zeroGrad();
          const l = computeLoss(points, ANCHOR_HOMES);
          l.backward();
          optimizer.step();

          setLoss(l.data);
          setIteration(i => i + 1);

          if (onMetrics) {
            onMetrics({
              loss: l.data,
              iteration: iteration + step,
              converged: l.data < 1.0,
            });
          }
        }
        setPoints([...points]); // Trigger re-render
        lastTime = currentTime;
      }

      animationRef.current = requestAnimationFrame(animate);
    };

    animationRef.current = requestAnimationFrame(animate);

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [isPlaying, optimizer, speed, points, iteration, onMetrics]);

  const handlePlay = useCallback(() => {
    // Always restart with new points
    const newPoints = createPoints();
    setPoints(newPoints);
    setIteration(0);
    setLoss(0);

    // Reinitialize optimizer with new points
    if (optimizerType !== 'lm') {
      const params = newPoints.flatMap(p => [p.x, p.y]);
      let opt: Optimizer;

      switch (optimizerType) {
        case 'sgd':
          opt = new SGD(params, { learningRate: sgdLR });
          break;
        case 'adam':
          opt = new Adam(params, { learningRate: adamLR });
          break;
        case 'adamw':
          opt = new AdamW(params, { learningRate: adamwLR, weightDecay: 0.001 });
          break;
      }

      setOptimizer(opt);
    }

    // Start playing
    requestAnimationFrame(() => {
      setIsPlaying(true);
    });
  }, [optimizerType, sgdLR, adamLR, adamwLR]);

  const handleOptimizerChange = useCallback((newType: 'sgd' | 'adam' | 'adamw' | 'lm') => {
    // Reset points to initial state
    const newPoints = createPoints();
    setPoints(newPoints);
    setIteration(0);
    setLoss(0);
    setOptimizerType(newType);

    // Automatically start optimizing with new optimizer
    setTimeout(() => setIsPlaying(true), 100);
  }, []);

  const render = useCallback((ctx: CanvasRenderingContext2D) => {
    // Clear canvas
    ctx.fillStyle = '#0f172a';
    ctx.fillRect(0, 0, width, height);

    // Compute centroid
    const avgX = points.reduce((sum, p) => sum + p.x.data, 0) / points.length;
    const avgY = points.reduce((sum, p) => sum + p.y.data, 0) / points.length;
    const computedCenterX = centerX + avgX;
    const computedCenterY = centerY + avgY;

    // Compute average radius
    const radii = points.map(p => {
      const dx = p.x.data - avgX;
      const dy = p.y.data - avgY;
      return Math.sqrt(dx * dx + dy * dy);
    });
    const avgRadius = radii.reduce((sum, r) => sum + r, 0) / radii.length;

    // Expected radius for target configuration
    const expectedRadius = (NUM_POINTS * TARGET_DISTANCE) / (2 * Math.PI);

    // Draw target circle (where points should end up)
    ctx.strokeStyle = '#475569';
    ctx.lineWidth = 3;
    ctx.setLineDash([10, 5]);
    ctx.beginPath();
    ctx.arc(computedCenterX, computedCenterY, expectedRadius, 0, Math.PI * 2);
    ctx.stroke();
    ctx.setLineDash([]);

    // Draw connections between adjacent points
    ctx.strokeStyle = '#6366f1';
    ctx.lineWidth = 1;
    ctx.globalAlpha = 0.4;
    for (let i = 0; i < points.length; i++) {
      const p1 = points[i];
      const p2 = points[(i + 1) % points.length];

      const x1 = centerX + p1.x.data;
      const y1 = centerY + p1.y.data;
      const x2 = centerX + p2.x.data;
      const y2 = centerY + p2.y.data;

      // Calculate actual distance vs target
      const dx = p2.x.data - p1.x.data;
      const dy = p2.y.data - p1.y.data;
      const dist = Math.sqrt(dx * dx + dy * dy);
      const error = Math.abs(dist - TARGET_DISTANCE);
      const errorRatio = Math.min(error / 20, 1);

      // Color based on constraint satisfaction
      const r = Math.floor(99 + errorRatio * 140);
      const g = Math.floor(102 - errorRatio * 30);
      const b = Math.floor(241 - errorRatio * 100);

      ctx.strokeStyle = `rgba(${r}, ${g}, ${b}, 0.6)`;
      ctx.lineWidth = 2;

      ctx.beginPath();
      ctx.moveTo(x1, y1);
      ctx.lineTo(x2, y2);
      ctx.stroke();
    }
    ctx.globalAlpha = 1;

    // Draw centroid
    ctx.fillStyle = '#f59e0b';
    ctx.globalAlpha = 0.5;
    ctx.beginPath();
    ctx.arc(computedCenterX, computedCenterY, 6, 0, Math.PI * 2);
    ctx.fill();
    ctx.globalAlpha = 1;

    // Draw points
    points.forEach((p) => {
      const x = centerX + p.x.data;
      const y = centerY + p.y.data;

      // Calculate distance from centroid
      const dx = p.x.data - avgX;
      const dy = p.y.data - avgY;
      const distFromCenter = Math.sqrt(dx * dx + dy * dy);
      const radiusError = Math.abs(distFromCenter - avgRadius);
      const isOnCircle = radiusError < 3;

      // Color based on how circular it is
      ctx.fillStyle = isOnCircle ? '#10b981' : '#6366f1';
      ctx.beginPath();
      ctx.arc(x, y, 5, 0, Math.PI * 2);
      ctx.fill();

      if (isOnCircle) {
        ctx.shadowColor = '#10b981';
        ctx.shadowBlur = 10;
        ctx.fill();
        ctx.shadowBlur = 0;
      }
    });
  }, [points, width, height, centerX, centerY]);

  return (
    <div className="demo-container">
      <div className="demo-header">
        <h2>Constrained Formation</h2>
        <div className="optimizer-selector">
          <button
            className={optimizerType === 'lm' ? 'active' : ''}
            onClick={() => handleOptimizerChange('lm')}
          >
            L-M
          </button>
          <button
            className={optimizerType === 'adam' ? 'active' : ''}
            onClick={() => handleOptimizerChange('adam')}
          >
            Adam
          </button>
          <button
            className={optimizerType === 'adamw' ? 'active' : ''}
            onClick={() => handleOptimizerChange('adamw')}
          >
            AdamW
          </button>
          <button
            className={optimizerType === 'sgd' ? 'active' : ''}
            onClick={() => handleOptimizerChange('sgd')}
          >
            SGD
          </button>
        </div>
      </div>

      <DemoCanvas width={width} height={height} onRender={render} />

      <div className="demo-sidebar">
        <DemoControls
          isPlaying={isPlaying}
          onPlayPause={handlePlay}
          speed={speed}
          onSpeedChange={setSpeed}
        />

        {optimizerType !== 'lm' && (
          <div style={{
            padding: '1rem',
            background: 'var(--background)',
            borderRadius: '0.5rem',
            display: 'flex',
            flexDirection: 'column',
            gap: '0.5rem',
          }}>
            <label style={{ fontSize: '0.875rem', color: 'var(--text)', opacity: 0.8 }}>
              Learning Rate
            </label>
            <input
              type="range"
              min="0.01"
              max="10"
              step="0.01"
              value={
                optimizerType === 'sgd' ? sgdLR :
                optimizerType === 'adam' ? adamLR :
                adamwLR
              }
              onChange={(e) => {
                const value = parseFloat(e.target.value);
                if (optimizerType === 'sgd') setSgdLR(value);
                else if (optimizerType === 'adam') setAdamLR(value);
                else setAdamwLR(value);
              }}
              style={{ width: '100%' }}
            />
            <div style={{
              fontSize: '1rem',
              fontWeight: 600,
              color: 'var(--text)',
              fontFamily: 'monospace',
            }}>
              {
                optimizerType === 'sgd' ? sgdLR.toFixed(2) :
                optimizerType === 'adam' ? adamLR.toFixed(2) :
                adamwLR.toFixed(2)
              }
            </div>
          </div>
        )}

        <MetricsDisplay
          metrics={{
            loss,
            iteration,
            converged: loss < 1.0,
          }}
        />
      </div>
    </div>
  );
}
