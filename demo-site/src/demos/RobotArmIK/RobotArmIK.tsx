import { useState, useCallback, useEffect, useRef, useMemo } from 'react';
import { DemoCanvas } from '../../components/DemoCanvas';
import type { DemoProps } from '../types';
import { V } from '../../../../src/V';
import { Adam } from '../../../../src/Optimizers';
import { Value } from '../../../../src/Value';

interface ArmSegment {
  length: number;
}

interface RobotArm {
  angles: Value[];
  segments: ArmSegment[];
  iteration: number;
  loss: number;
  converged: boolean;
  failed: boolean;
  convergenceReason: string;
  endEffector: { x: number; y: number };
}

const NUM_JOINTS = 6;
const BASE_SEGMENT_LENGTH = 45;

function createSegments(): ArmSegment[] {
  return Array.from({ length: NUM_JOINTS }, (_, i) => ({
    length: BASE_SEGMENT_LENGTH - i * 3
  }));
}

function forwardKinematics(angles: Value[], segments: ArmSegment[]): { x: Value; y: Value } {
  let x = V.C(0);
  let y = V.C(0);
  let cumulativeAngle = V.C(0);

  for (let i = 0; i < segments.length; i++) {
    cumulativeAngle = V.add(cumulativeAngle, angles[i]);
    const segmentX = V.mul(V.C(segments[i].length), V.cos(cumulativeAngle));
    const segmentY = V.mul(V.C(segments[i].length), V.sin(cumulativeAngle));
    x = V.add(x, segmentX);
    y = V.add(y, segmentY);
  }

  return { x, y };
}

function forwardKinematicsRaw(angles: number[], segments: ArmSegment[]): { x: number; y: number } {
  let x = 0;
  let y = 0;
  let cumulativeAngle = 0;

  for (let i = 0; i < segments.length; i++) {
    cumulativeAngle += angles[i];
    x += segments[i].length * Math.cos(cumulativeAngle);
    y += segments[i].length * Math.sin(cumulativeAngle);
  }

  return { x, y };
}

export function RobotArmIK({ width, height, onMetrics }: DemoProps) {
  const segments = useMemo(() => createSegments(), []);

  const [targetX, setTargetX] = useState(() => {
    const saved = localStorage.getItem('robotArmIK_targetX');
    return saved ? parseFloat(saved) : 150;
  });

  const [targetY, setTargetY] = useState(() => {
    const saved = localStorage.getItem('robotArmIK_targetY');
    return saved ? parseFloat(saved) : 100;
  });

  const [adamArm, setAdamArm] = useState<RobotArm | null>(null);
  const [nlsArm, setNlsArm] = useState<RobotArm | null>(null);
  const [nlsMaxIterations, setNlsMaxIterations] = useState(200);
  const [nlsCostTolerance, setNlsCostTolerance] = useState(1e-8);
  const [nlsParamTolerance, setNlsParamTolerance] = useState(1e-8);
  const [nlsGradientTolerance, setNlsGradientTolerance] = useState(1e-8);
  const [nlsInitialDamping, setNlsInitialDamping] = useState(1e-3);
  const [nlsDampingIncrease, setNlsDampingIncrease] = useState(10);
  const [nlsDampingDecrease, setNlsDampingDecrease] = useState(10);
  const [nlsLineSearchSteps, setNlsLineSearchSteps] = useState(10);
  const [nlsAdaptiveDamping, setNlsAdaptiveDamping] = useState(true);

  const adamOptimizerRef = useRef<Adam | null>(null);
  const animationRef = useRef<number>();
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const adamArmRef = useRef<RobotArm | null>(null);
  const nlsArmRef = useRef<RobotArm | null>(null);
  const targetXRef = useRef(targetX);
  const targetYRef = useRef(targetY);
  const adamWasConvergedRef = useRef(false);

  const centerX = width / 2;
  const centerY = height / 2;

  const runNLS = useCallback((arm: RobotArm) => {
    function residuals(params: Value[]) {
      const endEffector = forwardKinematics(params, segments);
      return [
        V.sub(endEffector.x, V.C(targetXRef.current)),
        V.sub(endEffector.y, V.C(targetYRef.current))
      ];
    }

    const result = V.nonlinearLeastSquares(arm.angles, residuals, {
      maxIterations: nlsMaxIterations,
      costTolerance: nlsCostTolerance,
      paramTolerance: nlsParamTolerance,
      gradientTolerance: nlsGradientTolerance,
      initialDamping: nlsInitialDamping,
      dampingIncreaseFactor: nlsDampingIncrease,
      dampingDecreaseFactor: nlsDampingDecrease,
      lineSearchSteps: nlsLineSearchSteps,
      adaptiveDamping: nlsAdaptiveDamping
    });

    const endEffector = forwardKinematicsRaw(
      arm.angles.map(a => a.data),
      segments
    );

    setNlsArm({
      ...arm,
      iteration: result.iterations,
      loss: result.finalCost,
      converged: result.success,
      failed: !result.success,
      convergenceReason: result.convergenceReason,
      endEffector
    });
  }, [segments, nlsMaxIterations, nlsCostTolerance, nlsParamTolerance, nlsGradientTolerance, nlsInitialDamping, nlsDampingIncrease, nlsDampingDecrease, nlsLineSearchSteps, nlsAdaptiveDamping]);

  const resetNLS = useCallback(() => {
    setNlsMaxIterations(200);
    setNlsCostTolerance(1e-8);
    setNlsParamTolerance(1e-8);
    setNlsGradientTolerance(1e-8);
    setNlsInitialDamping(1e-3);
    setNlsDampingIncrease(10);
    setNlsDampingDecrease(10);
    setNlsLineSearchSteps(10);
    setNlsAdaptiveDamping(true);
  }, []);

  const recordPosition = useCallback(() => {
    if (!nlsArm) return;

    const angles = nlsArm.angles.map(a => a.data);
    const testCase = {
      initialAngles: angles,
      targetX: targetX,
      targetY: targetY,
      segments: segments.map(s => s.length)
    };

    console.log('Test case:', JSON.stringify(testCase, null, 2));
    alert('Test case logged to console (F12)');
  }, [nlsArm, targetX, targetY, segments]);

  useEffect(() => {
    targetXRef.current = targetX;
    localStorage.setItem('robotArmIK_targetX', targetX.toString());
    if (nlsArmRef.current) {
      runNLS(nlsArmRef.current);
    }
    if (adamArmRef.current) {
      adamArmRef.current.iteration = 0;
      adamWasConvergedRef.current = false;
    }
  }, [targetX]);

  useEffect(() => {
    targetYRef.current = targetY;
    localStorage.setItem('robotArmIK_targetY', targetY.toString());
    if (nlsArmRef.current) {
      runNLS(nlsArmRef.current);
    }
    if (adamArmRef.current) {
      adamArmRef.current.iteration = 0;
      adamWasConvergedRef.current = false;
    }
  }, [targetY]);

  useEffect(() => {
    const initialAngles = segments.map((_, i) => 0.1 + i * 0.05);

    const adamAngles = initialAngles.map((a, i) => V.W(a, `adam_angle_${i}`));
    const nlsAngles = initialAngles.map((a, i) => V.W(a, `nls_angle_${i}`));

    const adamEndEffector = forwardKinematicsRaw(initialAngles, segments);
    const nlsEndEffector = forwardKinematicsRaw(initialAngles, segments);

    const newAdamArm = {
      angles: adamAngles,
      segments,
      iteration: 0,
      loss: 0,
      converged: false,
      failed: false,
      convergenceReason: '',
      endEffector: adamEndEffector
    };

    const newNlsArm = {
      angles: nlsAngles,
      segments,
      iteration: 0,
      loss: 0,
      converged: false,
      failed: false,
      convergenceReason: '',
      endEffector: nlsEndEffector
    };

    setAdamArm(newAdamArm);
    setNlsArm(newNlsArm);
    adamArmRef.current = newAdamArm;
    nlsArmRef.current = newNlsArm;

    adamOptimizerRef.current = new Adam(adamAngles, { learningRate: 0.1 });

    setTimeout(() => {
      runNLS(newNlsArm);
    }, 50);
  }, [segments]);

  useEffect(() => {
    if (nlsArmRef.current) {
      runNLS(nlsArmRef.current);
    }
  }, [nlsMaxIterations, nlsCostTolerance, nlsParamTolerance, nlsGradientTolerance, nlsInitialDamping, nlsDampingIncrease, nlsDampingDecrease, nlsLineSearchSteps, nlsAdaptiveDamping, runNLS]);

  useEffect(() => {
    const animate = () => {
      const arm = adamArmRef.current;
      if (arm) {
        function residuals(params: Value[]) {
          const endEffector = forwardKinematics(params, segments);
          return [
            V.sub(endEffector.x, V.C(targetXRef.current)),
            V.sub(endEffector.y, V.C(targetYRef.current))
          ];
        }

        const res = residuals(arm.angles);
        const cost = res.reduce((sum, r) => sum + r.data * r.data, 0);
        const converged = cost < 1e-8;

        if (!converged) {
          if (adamWasConvergedRef.current) {
            arm.iteration = 0;
          }

          const loss = V.mean(res.map(r => V.square(r)));
          adamOptimizerRef.current!.zeroGrad();
          loss.backward();
          adamOptimizerRef.current!.step();

          arm.iteration++;
        }

        arm.loss = cost;
        arm.converged = converged;
        adamWasConvergedRef.current = converged;

        const endEffector = forwardKinematicsRaw(
          arm.angles.map(a => a.data),
          segments
        );
        arm.endEffector = endEffector;

        setAdamArm({ ...arm });
      }

      animationRef.current = requestAnimationFrame(animate);
    };

    animationRef.current = requestAnimationFrame(animate);

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [segments]);

  const handleMouseDown = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = e.currentTarget;
    const rect = canvas.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;

    const targetCanvasX = centerX + targetX;
    const targetCanvasY = centerY + targetY;

    const dist = Math.sqrt(Math.pow(mouseX - targetCanvasX, 2) + Math.pow(mouseY - targetCanvasY, 2));

    if (dist < 15) {
      setIsDragging(true);
    }
  }, [centerX, centerY, targetX, targetY]);

  const handleMouseMove = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!isDragging) return;

    const canvas = e.currentTarget;
    const rect = canvas.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;

    const newX = mouseX - centerX;
    const newY = mouseY - centerY;

    setTargetX(newX);
    setTargetY(newY);
  }, [isDragging, centerX, centerY]);

  const handleMouseUp = useCallback(() => {
    setIsDragging(false);
  }, []);

  const render = useCallback((ctx: CanvasRenderingContext2D) => {
    ctx.fillStyle = '#0f172a';
    ctx.fillRect(0, 0, width, height);

    const drawArm = (arm: RobotArm | null, color: string) => {
      if (!arm) return;

      ctx.save();
      ctx.translate(centerX, centerY);

      ctx.beginPath();
      ctx.arc(0, 0, 8, 0, Math.PI * 2);
      ctx.fillStyle = '#475569';
      ctx.fill();

      ctx.globalAlpha = 0.5;

      let x = 0;
      let y = 0;
      let cumulativeAngle = 0;

      for (let i = 0; i < arm.segments.length; i++) {
        const angle = arm.angles[i].data;
        cumulativeAngle += angle;

        const nextX = x + arm.segments[i].length * Math.cos(cumulativeAngle);
        const nextY = y + arm.segments[i].length * Math.sin(cumulativeAngle);

        ctx.beginPath();
        ctx.moveTo(x, y);
        ctx.lineTo(nextX, nextY);
        ctx.strokeStyle = color;
        ctx.lineWidth = 6;
        ctx.lineCap = 'round';
        ctx.stroke();

        ctx.beginPath();
        ctx.arc(nextX, nextY, 5, 0, Math.PI * 2);
        ctx.fillStyle = color;
        ctx.fill();

        x = nextX;
        y = nextY;
      }

      ctx.globalAlpha = 1;

      const distToTarget = Math.sqrt(
        Math.pow(x - targetX, 2) + Math.pow(y - targetY, 2)
      );

      if (distToTarget < 3) {
        ctx.beginPath();
        ctx.arc(x, y, 15, 0, Math.PI * 2);
        ctx.fillStyle = color;
        ctx.globalAlpha = 0.3;
        ctx.fill();
        ctx.globalAlpha = 1;

        ctx.shadowColor = color;
        ctx.shadowBlur = 20;
        ctx.beginPath();
        ctx.arc(x, y, 8, 0, Math.PI * 2);
        ctx.fillStyle = color;
        ctx.fill();
        ctx.shadowBlur = 0;
      }

      ctx.restore();
    };

    drawArm(nlsArm, '#f59e0b');
    drawArm(adamArm, '#6366f1');

    ctx.beginPath();
    ctx.arc(centerX + targetX, centerY + targetY, 8, 0, Math.PI * 2);
    ctx.fillStyle = '#10b981';
    ctx.fill();
    ctx.strokeStyle = '#10b981';
    ctx.lineWidth = 2;
    ctx.stroke();

    ctx.fillStyle = '#10b981';
    ctx.font = '12px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('TARGET', centerX + targetX, centerY + targetY - 15);
  }, [width, height, centerX, centerY, targetX, targetY, adamArm, nlsArm]);

  return (
    <div className="demo-container">
      <div className="demo-header">
        <h2>Robot Arm Inverse Kinematics</h2>
        <p style={{ fontSize: '0.875rem', color: 'var(--text)', opacity: 0.7 }}>
          Two {NUM_JOINTS}-joint arms racing to reach the target. Drag the target to move it.
        </p>
      </div>

      <div
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
        style={{ cursor: isDragging ? 'grabbing' : 'default' }}
      >
        <DemoCanvas width={width} height={height} onRender={render} />
      </div>

      <div className="demo-sidebar">
        <div style={{
          padding: '1rem',
          background: 'var(--background)',
          borderRadius: '0.5rem',
          marginBottom: '1rem'
        }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.75rem' }}>
            <h3 style={{
              fontSize: '0.875rem',
              color: 'var(--text)',
              opacity: 0.8,
              margin: 0
            }}>
              NLS Parameters
            </h3>
            <div style={{ display: 'flex', gap: '0.5rem' }}>
              <button
                onClick={recordPosition}
                style={{
                  fontSize: '0.75rem',
                  padding: '0.25rem 0.5rem',
                  background: 'var(--surface)',
                  color: 'var(--text)',
                  border: '1px solid var(--border)',
                  borderRadius: '0.25rem',
                  cursor: 'pointer'
                }}
              >
                Record
              </button>
              <button
                onClick={resetNLS}
                style={{
                  fontSize: '0.75rem',
                  padding: '0.25rem 0.5rem',
                  background: 'var(--surface)',
                  color: 'var(--text)',
                  border: '1px solid var(--border)',
                  borderRadius: '0.25rem',
                  cursor: 'pointer'
                }}
              >
                Reset
              </button>
            </div>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.75rem' }}>
            <div>
              <label style={{ fontSize: '0.75rem', color: 'var(--text)', opacity: 0.7 }}>
                Max Iterations: {nlsMaxIterations}
              </label>
              <input
                type="range"
                min="10"
                max="500"
                step="10"
                value={nlsMaxIterations}
                onChange={(e) => setNlsMaxIterations(Number(e.target.value))}
                style={{ width: '100%', marginTop: '0.25rem' }}
              />
            </div>

            <div>
              <label style={{ fontSize: '0.75rem', color: 'var(--text)', opacity: 0.7 }}>
                Line Search: {nlsLineSearchSteps}
              </label>
              <input
                type="range"
                min="1"
                max="20"
                step="1"
                value={nlsLineSearchSteps}
                onChange={(e) => setNlsLineSearchSteps(Number(e.target.value))}
                style={{ width: '100%', marginTop: '0.25rem' }}
              />
            </div>

            <div>
              <label style={{ fontSize: '0.75rem', color: 'var(--text)', opacity: 0.7 }}>
                Cost Tol: {nlsCostTolerance.toExponential(1)}
              </label>
              <input
                type="range"
                min="-12"
                max="-4"
                step="0.5"
                value={Math.log10(nlsCostTolerance)}
                onChange={(e) => setNlsCostTolerance(Math.pow(10, Number(e.target.value)))}
                style={{ width: '100%', marginTop: '0.25rem' }}
              />
            </div>

            <div>
              <label style={{ fontSize: '0.75rem', color: 'var(--text)', opacity: 0.7 }}>
                Param Tol: {nlsParamTolerance.toExponential(1)}
              </label>
              <input
                type="range"
                min="-12"
                max="-4"
                step="0.5"
                value={Math.log10(nlsParamTolerance)}
                onChange={(e) => setNlsParamTolerance(Math.pow(10, Number(e.target.value)))}
                style={{ width: '100%', marginTop: '0.25rem' }}
              />
            </div>

            <div>
              <label style={{ fontSize: '0.75rem', color: 'var(--text)', opacity: 0.7 }}>
                Grad Tol: {nlsGradientTolerance.toExponential(1)}
              </label>
              <input
                type="range"
                min="-12"
                max="-4"
                step="0.5"
                value={Math.log10(nlsGradientTolerance)}
                onChange={(e) => setNlsGradientTolerance(Math.pow(10, Number(e.target.value)))}
                style={{ width: '100%', marginTop: '0.25rem' }}
              />
            </div>

            <div>
              <label style={{ fontSize: '0.75rem', color: 'var(--text)', opacity: 0.7 }}>
                Init Damp: {nlsInitialDamping.toExponential(1)}
              </label>
              <input
                type="range"
                min="-6"
                max="0"
                step="0.5"
                value={Math.log10(nlsInitialDamping)}
                onChange={(e) => setNlsInitialDamping(Math.pow(10, Number(e.target.value)))}
                style={{ width: '100%', marginTop: '0.25rem' }}
              />
            </div>

            <div>
              <label style={{ fontSize: '0.75rem', color: 'var(--text)', opacity: 0.7 }}>
                Damp Inc: {nlsDampingIncrease}
              </label>
              <input
                type="range"
                min="2"
                max="20"
                step="1"
                value={nlsDampingIncrease}
                onChange={(e) => setNlsDampingIncrease(Number(e.target.value))}
                style={{ width: '100%', marginTop: '0.25rem' }}
              />
            </div>

            <div>
              <label style={{ fontSize: '0.75rem', color: 'var(--text)', opacity: 0.7 }}>
                Damp Dec: {nlsDampingDecrease}
              </label>
              <input
                type="range"
                min="2"
                max="20"
                step="1"
                value={nlsDampingDecrease}
                onChange={(e) => setNlsDampingDecrease(Number(e.target.value))}
                style={{ width: '100%', marginTop: '0.25rem' }}
              />
            </div>

            <div style={{ gridColumn: '1 / -1' }}>
              <label style={{ fontSize: '0.75rem', color: 'var(--text)', opacity: 0.7, display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <input
                  type="checkbox"
                  checked={nlsAdaptiveDamping}
                  onChange={(e) => setNlsAdaptiveDamping(e.target.checked)}
                />
                Adaptive Damping
              </label>
            </div>
          </div>
        </div>

        {adamArm && nlsArm && (
          <div style={{
            padding: '1rem',
            background: 'var(--background)',
            borderRadius: '0.5rem',
          }}>
            <h3 style={{
              fontSize: '0.875rem',
              marginBottom: '0.75rem',
              color: 'var(--text)',
              opacity: 0.8
            }}>
              Comparison
            </h3>

            <div style={{
              display: 'flex',
              flexDirection: 'column',
              gap: '0.5rem',
              fontSize: '0.875rem',
              fontFamily: 'monospace'
            }}>
              <div style={{ color: '#f59e0b' }}>
                <strong>NLS:</strong> {nlsArm.iteration} iter
                {nlsArm.converged && ' [OK]'}
                {nlsArm.failed && ' [FAIL]'}
                {nlsArm.convergenceReason && (
                  <div style={{ fontSize: '0.75rem', opacity: 0.7, marginTop: '0.25rem' }}>
                    {nlsArm.convergenceReason}
                  </div>
                )}
              </div>
              <div style={{ color: '#6366f1' }}>
                <strong>Adam:</strong> {adamArm.iteration} iter
                {adamArm.converged && ' [OK]'}
              </div>

              {nlsArm.converged && adamArm.converged && (
                <div style={{
                  marginTop: '0.5rem',
                  paddingTop: '0.5rem',
                  borderTop: '1px solid var(--surface)',
                  color: 'var(--text)'
                }}>
                  Winner: {nlsArm.iteration < adamArm.iteration ? 'NLS' : 'Adam'} by {Math.abs(nlsArm.iteration - adamArm.iteration)} iterations
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
