import { useState, useCallback, useEffect } from 'react';
import { DemoCanvas } from '../../components/DemoCanvas';
import { DemoControls } from '../../components/DemoControls';
import { MetricsDisplay } from '../../components/MetricsDisplay';
import type { OptimizationMetrics } from '../../components/MetricsDisplay';
import type { DemoProps } from '../types';
import { V } from '../../../../V';
import type { Value } from '../../../../Value';

interface Point {
  x: number;
  y: number;
}

const TRUE_CENTER_X = 0;
const TRUE_CENTER_Y = 0;
const TRUE_RADIUS = 150;
const NUM_POINTS = 50;
const NOISE_LEVEL = 8;

function generateNoisyCircleData(): Point[] {
  const points: Point[] = [];
  for (let i = 0; i < NUM_POINTS; i++) {
    const angle = (i / NUM_POINTS) * 2 * Math.PI;
    const noise_x = (Math.random() - 0.5) * NOISE_LEVEL;
    const noise_y = (Math.random() - 0.5) * NOISE_LEVEL;
    points.push({
      x: TRUE_CENTER_X + TRUE_RADIUS * Math.cos(angle) + noise_x,
      y: TRUE_CENTER_Y + TRUE_RADIUS * Math.sin(angle) + noise_y,
    });
  }
  return points;
}

export function CircleFitting({ width, height, onMetrics }: DemoProps) {
  const [dataPoints] = useState<Point[]>(generateNoisyCircleData);
  const [centerX, setCenterX] = useState(0);
  const [centerY, setCenterY] = useState(0);
  const [radius, setRadius] = useState(100);
  const [isRunning, setIsRunning] = useState(false);
  const [iteration, setIteration] = useState(0);
  const [loss, setLoss] = useState(0);
  const [converged, setConverged] = useState(false);
  const [speed] = useState(1);

  const canvasCenterX = width / 2;
  const canvasCenterY = height / 2;

  const runOptimization = useCallback(() => {
    setIsRunning(true);
    setConverged(false);

    // Create differentiable parameters
    const params = [
      V.W(centerX, 'cx'),
      V.W(centerY, 'cy'),
      V.W(radius, 'r')
    ];

    // Run nonlinear least squares
    const result = V.nonlinearLeastSquares(
      params,
      ([cx, cy, r]: Value[]) => {
        // Compute residual for each data point (distance from circle)
        return dataPoints.map(p => {
          const dx = V.sub(p.x, cx);
          const dy = V.sub(p.y, cy);
          const dist = V.sqrt(V.add(V.square(dx), V.square(dy)));
          return V.sub(dist, r);
        });
      },
      {
        maxIterations: 100,
        costTolerance: 1e-6,
        verbose: false,
      }
    );

    // Update state with results
    setCenterX(params[0].data);
    setCenterY(params[1].data);
    setRadius(params[2].data);
    setIteration(result.iterations);
    setLoss(result.finalCost);
    setConverged(result.converged);
    setIsRunning(false);

    if (onMetrics) {
      onMetrics({
        loss: result.finalCost,
        iteration: result.iterations,
        converged: result.converged,
      });
    }
  }, [centerX, centerY, radius, dataPoints, onMetrics]);

  const handleReset = useCallback(() => {
    // Random initial guess
    setCenterX(Math.random() * 100 - 50);
    setCenterY(Math.random() * 100 - 50);
    setRadius(100 + Math.random() * 100);
    setIteration(0);
    setLoss(0);
    setConverged(false);
    setIsRunning(false);
  }, []);

  const handlePlayPause = useCallback(() => {
    if (!isRunning && !converged) {
      runOptimization();
    }
  }, [isRunning, converged, runOptimization]);

  // Auto-run on mount
  useEffect(() => {
    handleReset();
  }, []);

  const render = useCallback((ctx: CanvasRenderingContext2D) => {
    // Clear canvas
    ctx.fillStyle = '#0f172a';
    ctx.fillRect(0, 0, width, height);

    // Draw true circle (ground truth) in dark gray
    ctx.strokeStyle = '#1e293b';
    ctx.lineWidth = 2;
    ctx.setLineDash([5, 5]);
    ctx.beginPath();
    ctx.arc(
      canvasCenterX + TRUE_CENTER_X,
      canvasCenterY + TRUE_CENTER_Y,
      TRUE_RADIUS,
      0,
      Math.PI * 2
    );
    ctx.stroke();
    ctx.setLineDash([]);

    // Draw fitted circle
    const fittedCx = canvasCenterX + centerX;
    const fittedCy = canvasCenterY + centerY;

    ctx.strokeStyle = converged ? '#10b981' : '#6366f1';
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.arc(fittedCx, fittedCy, radius, 0, Math.PI * 2);
    ctx.stroke();

    // Draw fitted center point
    ctx.fillStyle = converged ? '#10b981' : '#6366f1';
    ctx.beginPath();
    ctx.arc(fittedCx, fittedCy, 5, 0, Math.PI * 2);
    ctx.fill();

    // Draw data points
    dataPoints.forEach(p => {
      const px = canvasCenterX + p.x;
      const py = canvasCenterY + p.y;

      // Calculate error for this point
      const dx = p.x - centerX;
      const dy = p.y - centerY;
      const dist = Math.sqrt(dx * dx + dy * dy);
      const error = Math.abs(dist - radius);

      // Color based on residual
      const errorRatio = Math.min(error / 20, 1);
      const r = Math.floor(99 + errorRatio * (239 - 99));
      const g = Math.floor(102 - errorRatio * 58);
      const b = Math.floor(241 - errorRatio * 173);

      ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
      ctx.beginPath();
      ctx.arc(px, py, 4, 0, Math.PI * 2);
      ctx.fill();

      // Draw residual line
      if (!converged) {
        ctx.strokeStyle = `rgba(${r}, ${g}, ${b}, 0.3)`;
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(px, py);

        // Project point onto fitted circle
        const angle = Math.atan2(p.y - centerY, p.x - centerX);
        const projX = canvasCenterX + centerX + radius * Math.cos(angle);
        const projY = canvasCenterY + centerY + radius * Math.sin(angle);

        ctx.lineTo(projX, projY);
        ctx.stroke();
      }
    });

    // Draw legend
    ctx.fillStyle = '#f1f5f9';
    ctx.font = '12px monospace';
    ctx.fillText('--- True Circle', 20, 30);
    ctx.fillText('— Fitted Circle', 20, 50);

    // Draw fitted parameters
    ctx.fillStyle = '#f1f5f9';
    ctx.font = '11px monospace';
    ctx.fillText(`Center: (${centerX.toFixed(1)}, ${centerY.toFixed(1)})`, width - 180, 30);
    ctx.fillText(`Radius: ${radius.toFixed(1)}`, width - 180, 50);
    ctx.fillText(`Error: ${Math.abs(radius - TRUE_RADIUS).toFixed(2)}`, width - 180, 70);
  }, [dataPoints, centerX, centerY, radius, converged, width, height, canvasCenterX, canvasCenterY]);

  return (
    <div className="demo-container">
      <div className="demo-header">
        <h2>Circle Fitting (Levenberg-Marquardt)</h2>
        <div className="demo-info">
          <span style={{ fontSize: '0.875rem', color: '#94a3b8' }}>
            {NUM_POINTS} noisy points • True radius: {TRUE_RADIUS}px
          </span>
        </div>
      </div>

      <DemoCanvas width={width} height={height} onRender={render} />

      <div className="demo-sidebar">
        <DemoControls
          isPlaying={isRunning}
          onPlayPause={handlePlayPause}
          onReset={handleReset}
          speed={speed}
          onSpeedChange={() => {}}
        />

        <MetricsDisplay
          metrics={{
            loss,
            iteration,
            converged,
          }}
        />

        {converged && (
          <div style={{
            padding: '1rem',
            background: '#10b981',
            color: 'white',
            borderRadius: '0.5rem',
            textAlign: 'center',
            fontWeight: 600,
          }}>
            ✓ Converged in {iteration} iterations using Levenberg-Marquardt!
            <div style={{ fontSize: '0.875rem', marginTop: '0.5rem', opacity: 0.9 }}>
              Radius error: {Math.abs(radius - TRUE_RADIUS).toFixed(3)}px
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
