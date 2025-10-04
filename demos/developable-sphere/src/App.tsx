import { useEffect, useRef, useState } from 'react';
import { IcoSphere } from './mesh/IcoSphere';
import { DevelopableOptimizer } from './optimization/DevelopableOptimizer';
import { DevelopableEnergy } from './energy/DevelopableEnergy';
import { MeshRenderer } from './visualization/MeshRenderer';
import { TriangleMesh } from './mesh/TriangleMesh';
import './App.css';

export default function App() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [renderer, setRenderer] = useState<MeshRenderer | null>(null);

  const [isOptimizing, setIsOptimizing] = useState(false);
  const [progress, setProgress] = useState({ iteration: 0, energy: 0 });
  const [currentMesh, setCurrentMesh] = useState<TriangleMesh | null>(null);
  const [history, setHistory] = useState<TriangleMesh[]>([]);
  const [currentFrame, setCurrentFrame] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [convergenceInfo, setConvergenceInfo] = useState<{
    reason: string;
    gradientNorm?: number;
    functionEvals?: number;
  } | null>(null);

  const [subdivisions, setSubdivisions] = useState(3);
  const [maxIterations, setMaxIterations] = useState(50);
  const [energyType, setEnergyType] = useState<'variance' | 'boundingbox'>('variance');

  const [metrics, setMetrics] = useState({
    hingeVertices: 0,
    seamVertices: 0,
    developableRatio: 0,
    averageEnergy: 0,
    totalVertices: 0,
    functionEvals: 0,
    kernelCount: 0,
    kernelReuse: 0,
  });

  // Initialize renderer and sphere
  useEffect(() => {
    if (!canvasRef.current) return;

    const r = new MeshRenderer(canvasRef.current);
    setRenderer(r);

    // Create initial sphere
    const sphere = IcoSphere.generate(subdivisions, 1.0);
    setCurrentMesh(sphere);

    const classification = DevelopableEnergy.classifyVertices(sphere);
    r.updateMesh(sphere, classification);
    r.render();

    updateMetrics(sphere);

    // Handle window resize
    const handleResize = () => {
      if (canvasRef.current) {
        const width = canvasRef.current.clientWidth;
        const height = canvasRef.current.clientHeight;
        r.resize(width, height);
      }
    };

    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  const updateMetrics = (mesh: TriangleMesh) => {
    const classification = DevelopableEnergy.classifyVertices(mesh);
    const energyValue = DevelopableEnergy.compute(mesh);
    const totalEnergy = energyValue.data;
    const avgEnergy = totalEnergy / mesh.vertices.length;

    setMetrics((prev) => ({
      ...prev,
      hingeVertices: classification.hingeVertices.length,
      seamVertices: classification.seamVertices.length,
      developableRatio: classification.hingeVertices.length / mesh.vertices.length,
      averageEnergy: avgEnergy,
      totalVertices: mesh.vertices.length,
    }));
  };

  const optimizerRef = useRef<DevelopableOptimizer | null>(null);
  const meshHistoryRef = useRef<TriangleMesh[]>([]);

  const handleOptimize = async () => {
    if (!renderer || !currentMesh) return;

    setIsOptimizing(true);
    setHistory([]);
    setProgress({ iteration: 0, energy: 0 });
    setConvergenceInfo(null);
    meshHistoryRef.current = [];

    // Create fresh sphere
    const sphere = IcoSphere.generate(subdivisions, 1.0);
    const optimizer = new DevelopableOptimizer(sphere);
    optimizerRef.current = optimizer;

    // Run async optimization (non-blocking)
    const result = await optimizer.optimizeAsync({
      maxIterations,
      gradientTolerance: 1e-8, // Relaxed from 1e-5 to allow more iterations
      verbose: true,
      captureInterval: Math.max(1, Math.floor(maxIterations / 20)),
      chunkSize: 5, // Process 5 iterations at a time
      energyType, // Pass energy type to optimizer
      onProgress: (iteration, energy, history) => {
        setProgress({ iteration, energy });

        // Update visualization in real-time
        if (history && history.length > 0) {
          const latestMesh = history[history.length - 1];
          const classification = DevelopableEnergy.classifyVertices(latestMesh);
          renderer.updateMesh(latestMesh, classification);
          renderer.render();
          updateMetrics(latestMesh);
          setCurrentMesh(latestMesh);
        }
      },
    });

    optimizerRef.current = null;
    setIsOptimizing(false);
    setHistory(result.history);
    setCurrentFrame(result.history.length - 1);
    setCurrentMesh(result.history[result.history.length - 1]);

    // Store convergence information
    setConvergenceInfo({
      reason: result.convergenceReason,
      gradientNorm: result.gradientNorm,
      functionEvals: result.functionEvaluations,
    });

    // Update function evaluations and kernel metrics
    setMetrics((prev) => ({
      ...prev,
      functionEvals: result.functionEvaluations || 0,
      kernelCount: result.kernelCount || 0,
      kernelReuse: result.kernelReuseFactor || 0,
    }));

    // Update visualization
    const finalMesh = result.history[result.history.length - 1];
    const classification = DevelopableEnergy.classifyVertices(finalMesh);
    renderer.updateMesh(finalMesh, classification);
    renderer.render();

    updateMetrics(finalMesh);
  };

  const handleStop = () => {
    if (optimizerRef.current) {
      optimizerRef.current.stop();
    }
  };

  const handleReset = () => {
    if (!renderer) return;

    const sphere = IcoSphere.generate(subdivisions, 1.0);
    setCurrentMesh(sphere);
    setHistory([]);
    setCurrentFrame(0);
    setIsPlaying(false);
    setConvergenceInfo(null);

    const classification = DevelopableEnergy.classifyVertices(sphere);
    renderer.updateMesh(sphere, classification);
    renderer.render();

    updateMetrics(sphere);
    setMetrics((prev) => ({ ...prev, functionEvals: 0, kernelCount: 0, kernelReuse: 0 }));
  };

  const handleFrameChange = (frame: number) => {
    if (!renderer || history.length === 0) return;

    const frameIdx = Math.max(0, Math.min(frame, history.length - 1));
    setCurrentFrame(frameIdx);

    const mesh = history[frameIdx];
    const classification = DevelopableEnergy.classifyVertices(mesh);
    renderer.updateMesh(mesh, classification);
    renderer.render();

    updateMetrics(mesh);
  };

  // Animation playback
  useEffect(() => {
    if (!isPlaying || history.length === 0) return;

    const interval = setInterval(() => {
      setCurrentFrame((prev) => {
        const next = (prev + 1) % history.length;
        handleFrameChange(next);
        return next;
      });
    }, 100);

    return () => clearInterval(interval);
  }, [isPlaying, history]);

  return (
    <div className="app">
      <div className="main-content">
        <div className="canvas-container">
          <canvas ref={canvasRef} width={800} height={600} />

          {/* Title Overlay */}
          <div style={{
            position: 'absolute',
            top: '10px',
            right: '10px',
            background: 'rgba(0, 0, 0, 0.5)',
            color: 'white',
            padding: '8px 12px',
            borderRadius: '4px',
            fontSize: '14px',
            fontWeight: 'bold',
            pointerEvents: 'none'
          }}>
            Developable Sphere Demo
          </div>

          {/* Status Overlay - Top Left */}
          <div style={{
            position: 'absolute',
            top: '10px',
            left: '10px',
            background: 'rgba(0, 0, 0, 0.7)',
            color: 'white',
            padding: '12px',
            borderRadius: '4px',
            fontSize: '12px',
            pointerEvents: 'none',
            minWidth: '200px'
          }}>
            {isOptimizing && (
              <>
                <div><strong>Optimizing...</strong></div>
                <div>Step: {progress.iteration} / {maxIterations}</div>
                <div>Energy: {progress.energy.toExponential(3)}</div>
              </>
            )}
            {convergenceInfo && !isOptimizing && (
              <>
                <div><strong>{convergenceInfo.reason}</strong></div>
                {convergenceInfo.gradientNorm !== undefined && (
                  <div>Grad: {convergenceInfo.gradientNorm.toExponential(3)}</div>
                )}
                {convergenceInfo.functionEvals !== undefined && (
                  <div>F-evals: {convergenceInfo.functionEvals}</div>
                )}
              </>
            )}
            {!isOptimizing && !convergenceInfo && (
              <div><strong>Ready</strong></div>
            )}
          </div>
        </div>

        <div className="controls-panel">

          <div className="metrics-section">
            <h3>Metrics</h3>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '4px', fontSize: '13px' }}>
              <div><span className="hinge-color">●</span> Hinges: {metrics.hingeVertices}</div>
              <div><span className="seam-color">●</span> Seams: {metrics.seamVertices}</div>
              <div>Developable: {(metrics.developableRatio * 100).toFixed(1)}%</div>
              <div>Vertices: {metrics.totalVertices}</div>
              <div>Avg Energy: {metrics.averageEnergy.toExponential(2)}</div>
              <div>F-evals: {metrics.functionEvals}</div>
              <div>Kernels: {metrics.kernelCount}</div>
              <div>Reuse: {metrics.kernelReuse.toFixed(1)}x</div>
            </div>
          </div>

          <div className="control-section">
            <h3>Settings</h3>

            <label>
              Energy Function
              <select
                value={energyType}
                onChange={(e) => setEnergyType(e.target.value as 'variance' | 'boundingbox')}
                disabled={isOptimizing}
              >
                <option value="boundingbox">Bounding Box</option>
                <option value="variance">Variance</option>
              </select>
            </label>

            <label>
              Subdivisions: {subdivisions}
              <input
                type="range"
                min="2"
                max="6"
                value={subdivisions}
                onChange={(e) => setSubdivisions(parseInt(e.target.value))}
                disabled={isOptimizing}
              />
              <span className="hint">
                {subdivisions === 2 && '162 verts'}
                {subdivisions === 3 && '642 verts'}
                {subdivisions === 4 && '2562 verts'}
                {subdivisions === 5 && '10242 verts'}
                {subdivisions === 6 && '40962 verts'}
              </span>
            </label>

            <label>
              Max Iterations: {maxIterations}
              <input
                type="range"
                min="20"
                max="200"
                step="10"
                value={maxIterations}
                onChange={(e) => setMaxIterations(parseInt(e.target.value))}
                disabled={isOptimizing}
              />
            </label>
          </div>

          <div className="control-section">
            <h3>Actions</h3>
            <button onClick={handleOptimize} disabled={isOptimizing} className="primary-button">
              {isOptimizing ? `Optimizing... (${progress.iteration})` : 'Run Optimization'}
            </button>
            {isOptimizing && (
              <button onClick={handleStop} className="secondary-button">
                Stop
              </button>
            )}
            <button onClick={handleReset} disabled={isOptimizing}>
              Reset to Sphere
            </button>
          </div>

          {history.length > 0 && (
            <div className="control-section">
              <h3>Animation</h3>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <button
                  onClick={() => setIsPlaying(!isPlaying)}
                  style={{ padding: '4px 8px', fontSize: '12px' }}
                >
                  {isPlaying ? '⏸' : '▶'}
                </button>
                <span style={{ fontSize: '12px', minWidth: '60px' }}>
                  {currentFrame + 1} / {history.length}
                </span>
                <input
                  type="range"
                  min="0"
                  max={history.length - 1}
                  value={currentFrame}
                  onChange={(e) => {
                    setIsPlaying(false);
                    handleFrameChange(parseInt(e.target.value));
                  }}
                  style={{ flex: 1 }}
                />
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
