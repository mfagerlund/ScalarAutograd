import { useEffect, useRef, useState } from 'react';
import { IcoSphere } from './mesh/IcoSphere';
import { DevelopableOptimizer } from './optimization/DevelopableOptimizer';
import { SubdividedMesh } from './mesh/SubdividedMesh';
import { CurvatureClassifier } from './energy/CurvatureClassifier';
import { MeshRenderer } from './visualization/MeshRenderer';
import { TriangleMesh } from './mesh/TriangleMesh';
import { EnergyRegistry } from './energy/EnergyRegistry';
import './App.css';

// Import all energies to ensure they register
import './energy/DevelopableEnergy';
import './energy/CovarianceEnergy';
import './energy/StochasticBimodalEnergy';
import './energy/RidgeBasedEnergy';
import './energy/AlignmentBimodalEnergy';
import './energy/BoundingBoxEnergy';
import './energy/CombinatorialEnergy';
import './energy/ContiguousBimodalEnergy';
import './energy/EigenProxyEnergy';
import './energy/StochasticCovarianceEnergy';
import './energy/GreatCircleEnergy';
import './energy/GreatCircleEnergyEx';
import './energy/DifferentiablePlaneAlignment';

const VERSION = 'v0';

export default function App() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [renderer, setRenderer] = useState<MeshRenderer | null>(null);

  const [isOptimizing, setIsOptimizing] = useState(false);
  const [isCompiling, setIsCompiling] = useState(false);
  const [progress, setProgress] = useState({ iteration: 0, energy: 0 });
  const [currentMesh, setCurrentMesh] = useState<TriangleMesh | null>(null);
  const [history, setHistory] = useState<TriangleMesh[]>([]);
  const [currentFrame, setCurrentFrame] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [convergenceInfo, setConvergenceInfo] = useState<{
    reason: string;
    iterations: number;
    timeElapsed: number;
    developableBefore: number;
    developableAfter: number;
    regionsBefore: number;
    regionsAfter: number;
    functionEvals?: number;
    compilationTime?: number;
    numFunctions?: number;
  } | null>(null);

  const [subdivisions, setSubdivisions] = useState(3);
  const [maxIterations, setMaxIterations] = useState(50);
  const [energyType, setEnergyType] = useState<string>(EnergyRegistry.getNames()[0] || 'bimodal');
  const [useCompiled, setUseCompiled] = useState(false);
  const [optimizer, setOptimizer] = useState<'lbfgs' | 'leastsquares'>('lbfgs');
  const [iterationsPerSecond, setIterationsPerSecond] = useState<number>(0);

  // Multi-resolution settings
  const [useMultiRes, setUseMultiRes] = useState(false);
  const [multiResStartLevel, setMultiResStartLevel] = useState(0);
  const [multiResTargetLevel, setMultiResTargetLevel] = useState(2);

  const [metrics, setMetrics] = useState({
    hingeVertices: 0,
    seamVertices: 0,
    developableRatio: 0,
    averageEnergy: 0,
    totalVertices: 0,
    functionEvals: 0,
    kernelCount: 0,
    kernelReuse: 0,
    numRegions: 0,
  });

  // Initialize renderer and sphere
  useEffect(() => {
    if (!canvasRef.current) return;

    const r = new MeshRenderer(canvasRef.current);
    setRenderer(r);

    // Create initial sphere
    const sphere = IcoSphere.generate(subdivisions, 1.0);
    setCurrentMesh(sphere);

    const classification = CurvatureClassifier.classifyVertices(sphere);
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
    // Use curvature for objective visualization (not energy-dependent)
    const classification = CurvatureClassifier.classifyVertices(mesh);
    const stats = CurvatureClassifier.getCurvatureStats(mesh);
    const avgCurvature = stats.mean;
    const numRegions = CurvatureClassifier.countDevelopableRegions(mesh);

    setMetrics((prev) => ({
      ...prev,
      hingeVertices: classification.hingeVertices.length,
      seamVertices: classification.seamVertices.length,
      developableRatio: classification.hingeVertices.length / mesh.vertices.length,
      averageEnergy: avgCurvature,
      totalVertices: mesh.vertices.length,
      numRegions,
    }));
  };

  const optimizerRef = useRef<DevelopableOptimizer | null>(null);
  const meshHistoryRef = useRef<TriangleMesh[]>([]);

  const handleOptimize = async () => {
    if (!renderer || !currentMesh || isOptimizing) return;

    // Set state and give UI a chance to update
    setIsOptimizing(true);
    setIsCompiling(useCompiled && !useMultiRes); // Multi-res handles compilation per level
    setHistory([]);
    setProgress({ iteration: 0, energy: 0 });
    setConvergenceInfo(null);
    setIterationsPerSecond(0);
    meshHistoryRef.current = [];

    // Yield to browser to update UI (0ms = yield to event loop)
    await new Promise(resolve => setTimeout(resolve, 0));

    const optimizationStartTime = Date.now();

    try {
      let result;
      let sphere: TriangleMesh;
      let developableBefore: number;
      let regionsBefore: number;

      if (useMultiRes) {
        // ========================================
        // MULTI-RESOLUTION OPTIMIZATION
        // ========================================
        // Create base mesh at start level
        const baseMesh = IcoSphere.generate(multiResStartLevel, 1.0);
        const subdividedBase = SubdividedMesh.fromMesh(baseMesh);

        // Capture initial developability
        const initialClassification = CurvatureClassifier.classifyVertices(baseMesh);
        developableBefore = initialClassification.hingeVertices.length / baseMesh.vertices.length;
        regionsBefore = CurvatureClassifier.countDevelopableRegions(baseMesh);

        console.log(`\n🚀 Starting Multi-Resolution Optimization`);
        console.log(`   Levels: ${multiResStartLevel} → ${multiResTargetLevel}`);
        console.log(`   Coarse energy: combinatorial, Fine energy: covariance`);

        // Run multi-resolution optimization
        result = await DevelopableOptimizer.optimizeMultiResolution(subdividedBase, {
          startLevel: multiResStartLevel,
          targetLevel: multiResTargetLevel,
          iterationsPerLevel: maxIterations,
          gradientTolerance: 1e-8,
          verbose: true,
          coarseEnergyType: 'combinatorial',
          fineEnergyType: 'covariance',
          useCompiled: useCompiled,
          onProgress: (_level, iteration, energy) => {
            setProgress({ iteration, energy });
            const elapsed = (Date.now() - optimizationStartTime) / 1000;
            if (elapsed > 0) {
              setIterationsPerSecond(iteration / elapsed);
            }
          },
        });

        sphere = result.history[result.history.length - 1];

      } else {
        // ========================================
        // SINGLE-LEVEL OPTIMIZATION (original)
        // ========================================
        // Create fresh sphere
        sphere = IcoSphere.generate(subdivisions, 1.0);

        // Capture initial developability
        const initialClassification = CurvatureClassifier.classifyVertices(sphere);
        developableBefore = initialClassification.hingeVertices.length / sphere.vertices.length;
        regionsBefore = CurvatureClassifier.countDevelopableRegions(sphere);

        // Immediately render the fresh sphere
        renderer.updateMesh(sphere, initialClassification);
        renderer.render();
        updateMetrics(sphere);
        setCurrentMesh(sphere);

        const opt = new DevelopableOptimizer(sphere);
        optimizerRef.current = opt;

        // Run async optimization (non-blocking)
        result = await opt.optimizeAsync({
          maxIterations,
          gradientTolerance: 1e-8, // Relaxed from 1e-5 to allow more iterations
          verbose: true,
          captureInterval: Math.max(1, Math.floor(maxIterations / 20)),
          chunkSize: 5, // Process 5 iterations at a time
          energyType, // Pass energy type to optimizer
          useCompiled, // Use compiled gradients
          optimizer, // Pass optimizer selection
          onProgress: (iteration, energy, history) => {
            // End compilation phase when optimization starts
            if (isCompiling) {
              setIsCompiling(false);
            }

            // Always update progress numbers
            setProgress({ iteration, energy });

            // Calculate iterations per second
            const elapsed = (Date.now() - optimizationStartTime) / 1000;
            if (elapsed > 0) {
              setIterationsPerSecond(iteration / elapsed);
            }

            // Update visualization when new mesh is captured
            if (history && history.length > 0) {
              const latestMesh = history[history.length - 1];
              const classification = CurvatureClassifier.classifyVertices(latestMesh);
              renderer.updateMesh(latestMesh, classification);
              renderer.render();
              updateMetrics(latestMesh);
              setCurrentMesh(latestMesh);
            }
          },
        });
      }

      // Update function evaluations and kernel metrics
      setMetrics((prev) => ({
        ...prev,
        functionEvals: result.functionEvaluations || 0,
        kernelCount: result.kernelCount || 0,
        kernelReuse: result.kernelReuseFactor || 0,
      }));

      // Update visualization
      const finalMesh = result.history[result.history.length - 1];
      const finalClassification = CurvatureClassifier.classifyVertices(finalMesh);
      const developableAfter = finalClassification.hingeVertices.length / finalMesh.vertices.length;
      const regionsAfter = CurvatureClassifier.countDevelopableRegions(finalMesh);

      renderer.updateMesh(finalMesh, finalClassification);
      renderer.render();

      updateMetrics(finalMesh);

      setHistory(result.history);
      setCurrentFrame(result.history.length - 1);
      setCurrentMesh(finalMesh);

      // Calculate elapsed time
      const timeElapsed = (Date.now() - optimizationStartTime) / 1000;

      // Store convergence information with before/after comparison
      setConvergenceInfo({
        reason: result.convergenceReason,
        iterations: result.iterations,
        timeElapsed,
        developableBefore,
        developableAfter,
        regionsBefore,
        regionsAfter,
        functionEvals: result.functionEvaluations,
        compilationTime: result.compilationTime,
        numFunctions: result.numFunctions,
      });

      // Log comprehensive summary
      console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
      console.log('OPTIMIZATION COMPLETE');
      console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
      if (useMultiRes) {
        console.log(`Mode: Multi-Resolution (${multiResStartLevel} → ${multiResTargetLevel})`);
        console.log(`Energy: E^P (combinatorial) → E^λ (covariance)`);
      } else {
        console.log(`Energy Function: ${energyType}`);
        console.log(`Mode: ${useCompiled ? 'Compiled' : 'Non-compiled'}`);
      }
      if (useCompiled && result.compilationTime !== undefined) {
        console.log('');
        console.log('COMPILATION:');
        console.log(`  Time: ${result.compilationTime.toFixed(2)}s`);
        console.log(`  Functions: ${result.numFunctions}`);
        console.log(`  Kernels: ${result.kernelCount}`);
        console.log(`  Reuse: ${result.kernelReuseFactor?.toFixed(1)}x`);
      }
      console.log('');
      console.log('OPTIMIZATION:');
      console.log(`  Iterations: ${result.iterations} ${useMultiRes ? '(total across all levels)' : `/ ${maxIterations}`}`);
      console.log(`  Time Elapsed: ${timeElapsed.toFixed(2)}s`);
      console.log(`  Speed: ${(result.iterations / timeElapsed).toFixed(1)} it/s`);
      console.log(`  Function Evals: ${result.functionEvaluations}`);
      console.log('');
      console.log('DEVELOPABILITY (Objective Curvature Threshold):');
      console.log(`  Before: ${(developableBefore * 100).toFixed(2)}% (${regionsBefore} regions)`);
      console.log(`  After:  ${(developableAfter * 100).toFixed(2)}% (${regionsAfter} regions)`);
      console.log(`  Change: ${((developableAfter - developableBefore) * 100).toFixed(2)}% ${developableAfter > developableBefore ? '✓' : '✗'}`);
      console.log(`  Region Quality: ${(developableAfter * 100 / Math.max(1, regionsAfter)).toFixed(1)}% per region`);
      console.log('');
      console.log(`Convergence: ${result.convergenceReason}`);
      console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');

      // Alert user if optimization completed very quickly (likely failed)
      const elapsed = (Date.now() - optimizationStartTime) / 1000;
      if (result.iterations < 3 && elapsed < 1) {
        alert(`Optimization ended quickly after ${result.iterations} iterations.\nReason: ${result.convergenceReason}\n\nThis often means the initial mesh already has very low energy. Try a different energy function or add perturbation.`);
      }
    } finally {
      // Always clean up state
      optimizerRef.current = null;
      setIsOptimizing(false);
    }
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

    const classification = CurvatureClassifier.classifyVertices(sphere);
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
    const classification = CurvatureClassifier.classifyVertices(mesh);
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
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '4px' }}>
                  <div className="spinner" style={{
                    width: '12px',
                    height: '12px',
                    border: '2px solid rgba(255,255,255,0.3)',
                    borderTop: '2px solid white',
                    borderRadius: '50%',
                    animation: 'spin 0.8s linear infinite'
                  }} />
                  <strong>{isCompiling ? 'Compiling...' : 'Optimizing...'}</strong>
                </div>
                {isCompiling ? (
                  <>
                    <div>Building computation graph...</div>
                    <div style={{ fontSize: '11px', opacity: 0.7, marginTop: '2px' }}>
                      This may take a moment for large meshes
                    </div>
                  </>
                ) : (
                  <>
                    <div>Step: {progress.iteration} / {maxIterations}</div>
                    <div>Energy: {progress.energy.toExponential(3)}</div>
                    <div>Speed: {iterationsPerSecond.toFixed(1)} it/s</div>
                    <div style={{
                      marginTop: '6px',
                      background: 'rgba(255,255,255,0.2)',
                      borderRadius: '2px',
                      height: '4px',
                      overflow: 'hidden'
                    }}>
                      <div style={{
                        width: `${(progress.iteration / maxIterations) * 100}%`,
                        height: '100%',
                        background: '#4CAF50',
                        transition: 'width 0.3s'
                      }} />
                    </div>
                  </>
                )}
              </>
            )}
            {convergenceInfo && !isOptimizing && (
              <div
                onClick={() => {
                  const report = `Optimization Report
Energy Function: ${energyType}
Subdivisions: ${subdivisions}
Max Iterations: ${maxIterations}
Compiled: ${useCompiled}
${useCompiled && convergenceInfo.compilationTime !== undefined ? `
Compilation:
- Time: ${convergenceInfo.compilationTime.toFixed(2)}s
- Functions: ${convergenceInfo.numFunctions}
- Kernels: ${metrics.kernelCount}
- Reuse: ${metrics.kernelReuse.toFixed(1)}x
` : ''}
Optimization:
- Iterations: ${convergenceInfo.iterations} / ${maxIterations}
- Time Elapsed: ${convergenceInfo.timeElapsed.toFixed(2)}s
- Speed: ${(convergenceInfo.iterations / convergenceInfo.timeElapsed).toFixed(1)} it/s
- Function Evals: ${convergenceInfo.functionEvals}

Developability (Objective - Curvature Threshold):
- Before: ${(convergenceInfo.developableBefore * 100).toFixed(2)}% (${convergenceInfo.regionsBefore} regions)
- After: ${(convergenceInfo.developableAfter * 100).toFixed(2)}% (${convergenceInfo.regionsAfter} regions)
- Change: ${((convergenceInfo.developableAfter - convergenceInfo.developableBefore) * 100).toFixed(2)}%

Convergence: ${convergenceInfo.reason}`;

                  navigator.clipboard.writeText(report).then(() => {
                    console.log('Report copied to clipboard!');
                  });
                }}
                style={{
                  cursor: 'pointer',
                  pointerEvents: 'auto',
                  padding: '4px',
                  margin: '-4px',
                  borderRadius: '4px',
                  transition: 'background 0.2s'
                }}
                onMouseEnter={(e) => e.currentTarget.style.background = 'rgba(255,255,255,0.1)'}
                onMouseLeave={(e) => e.currentTarget.style.background = 'transparent'}
                title="Click to copy report to clipboard"
              >
                <div><strong>Optimization Complete</strong> <span style={{ fontSize: '10px', opacity: 0.7 }}>(click to copy)</span></div>
                {useCompiled && convergenceInfo.compilationTime !== undefined && (
                  <div style={{ fontSize: '11px', marginTop: '4px', opacity: 0.8 }}>
                    Compiled {convergenceInfo.numFunctions} fn → {metrics.kernelCount} kernels ({metrics.kernelReuse.toFixed(1)}x) in {convergenceInfo.compilationTime.toFixed(1)}s
                  </div>
                )}
                <div>{convergenceInfo.iterations} iterations in {convergenceInfo.timeElapsed.toFixed(1)}s</div>
                <div style={{ marginTop: '6px', borderTop: '1px solid rgba(255,255,255,0.2)', paddingTop: '6px' }}>
                  <div>Before: {(convergenceInfo.developableBefore * 100).toFixed(1)}%</div>
                  <div>After: {(convergenceInfo.developableAfter * 100).toFixed(1)}%</div>
                  <div style={{ color: convergenceInfo.developableAfter > convergenceInfo.developableBefore ? '#4CAF50' : '#f44336' }}>
                    Change: {((convergenceInfo.developableAfter - convergenceInfo.developableBefore) * 100).toFixed(1)}%
                  </div>
                </div>
                <div style={{ marginTop: '4px', fontSize: '11px', opacity: 0.7 }}>
                  {convergenceInfo.reason}
                </div>
              </div>
            )}
            {!isOptimizing && !convergenceInfo && (
              <div><strong>Ready</strong></div>
            )}
          </div>

          {/* CSS for spinner animation */}
          <style>{`
            @keyframes spin {
              0% { transform: rotate(0deg); }
              100% { transform: rotate(360deg); }
            }
          `}</style>
        </div>

        <div className="controls-panel">

          <div className="metrics-section">
            <h3>Developable Sphere Demo {VERSION}</h3>
            <h4 style={{ margin: '8px 0', fontSize: '14px', fontWeight: 'normal', opacity: 0.8 }}>Metrics</h4>

            <label style={{ marginBottom: '8px' }}>
              Optimizer
              <select
                value={optimizer}
                onChange={(e) => setOptimizer(e.target.value as 'lbfgs' | 'leastsquares')}
                disabled={isOptimizing}
                style={{ fontFamily: 'monospace', fontSize: '12px' }}
              >
                <option value="lbfgs">L-BFGS (Quasi-Newton)</option>
                <option value="leastsquares">Levenberg-Marquardt (Least Squares)</option>
              </select>
            </label>

            <label style={{ marginBottom: '8px' }}>
              Energy Function
              <select
                value={energyType}
                onChange={(e) => setEnergyType(e.target.value)}
                disabled={isOptimizing}
                style={{ fontFamily: 'monospace', fontSize: '12px' }}
              >
                {EnergyRegistry.getAll().map((energy) => {
                  // Extract class name from the energy object
                  const className = energy.constructor?.name || energy.name;
                  return (
                    <option key={energy.name} value={energy.name}>
                      {energy.name} ({className})
                    </option>
                  );
                })}
              </select>
            </label>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '4px', fontSize: '13px' }}>
              <div><span className="hinge-color">●</span> Hinges: {metrics.hingeVertices}</div>
              <div><span className="seam-color">●</span> Seams: {metrics.seamVertices}</div>
              <div>Developable: {(metrics.developableRatio * 100).toFixed(1)}%</div>
              <div>Regions: {metrics.numRegions}</div>
              <div>Vertices: {metrics.totalVertices}</div>
              <div>Avg Curvature: {metrics.averageEnergy.toExponential(2)}</div>
              <div>Speed: {iterationsPerSecond.toFixed(1)} it/s</div>
              <div>F-evals: {metrics.functionEvals}</div>
              {useCompiled && metrics.kernelCount > 0 && (
                <>
                  <div>Kernels: {metrics.kernelCount}</div>
                  <div>Reuse: {metrics.kernelReuse.toFixed(1)}x</div>
                </>
              )}
              <div style={{ gridColumn: '1 / -1', fontSize: '11px', opacity: 0.7, marginTop: '4px' }}>
                Threshold: 10% of sphere curvature ({((0.1 * 4 * Math.PI / metrics.totalVertices) * 180 / Math.PI).toFixed(2)}°)
              </div>
            </div>
          </div>

          <div className="control-section">
            <h3>Settings</h3>

            <label className="checkbox-label">
              <input
                type="checkbox"
                checked={useCompiled}
                onChange={(e) => setUseCompiled(e.target.checked)}
                disabled={isOptimizing}
              />
              Use Compiled Gradients (faster)
            </label>

            <label className="checkbox-label">
              <input
                type="checkbox"
                checked={useMultiRes}
                onChange={(e) => setUseMultiRes(e.target.checked)}
                disabled={isOptimizing}
              />
              Multi-Resolution (🎯 fixes fragmentation!)
            </label>

            {!useMultiRes ? (
              <label>
                Subdivisions: {subdivisions}
                <input
                  type="range"
                  min="0"
                  max="6"
                  value={subdivisions}
                  onChange={(e) => setSubdivisions(parseInt(e.target.value))}
                  disabled={isOptimizing}
                />
                <span className="hint">
                  {subdivisions === 0 && '12 verts (icosahedron)'}
                  {subdivisions === 1 && '42 verts'}
                  {subdivisions === 2 && '162 verts'}
                  {subdivisions === 3 && '642 verts'}
                  {subdivisions === 4 && '2562 verts'}
                  {subdivisions === 5 && '10242 verts'}
                  {subdivisions === 6 && '40962 verts'}
                </span>
              </label>
            ) : (
              <>
                <label>
                  Start Level: {multiResStartLevel}
                  <input
                    type="range"
                    min="0"
                    max="2"
                    value={multiResStartLevel}
                    onChange={(e) => setMultiResStartLevel(parseInt(e.target.value))}
                    disabled={isOptimizing}
                  />
                  <span className="hint">
                    {multiResStartLevel === 0 && '12 verts (coarsest)'}
                    {multiResStartLevel === 1 && '42 verts (coarse)'}
                    {multiResStartLevel === 2 && '162 verts (medium)'}
                  </span>
                </label>

                <label>
                  Target Level: {multiResTargetLevel}
                  <input
                    type="range"
                    min={multiResStartLevel}
                    max="4"
                    value={multiResTargetLevel}
                    onChange={(e) => setMultiResTargetLevel(parseInt(e.target.value))}
                    disabled={isOptimizing}
                  />
                  <span className="hint">
                    {multiResTargetLevel === 0 && '12 verts'}
                    {multiResTargetLevel === 1 && '42 verts'}
                    {multiResTargetLevel === 2 && '162 verts'}
                    {multiResTargetLevel === 3 && '642 verts'}
                    {multiResTargetLevel === 4 && '2562 verts'}
                  </span>
                </label>

                <div style={{ fontSize: '11px', opacity: 0.7, padding: '8px', background: 'rgba(76, 175, 80, 0.1)', borderRadius: '4px', marginTop: '4px' }}>
                  💡 Multi-res optimizes at each level from {multiResStartLevel} → {multiResTargetLevel}, preventing fragmentation by establishing topology on coarse mesh first.
                </div>
                <div style={{ fontSize: '10px', opacity: 0.6, marginTop: '4px', fontStyle: 'italic' }}>
                  Paper recommends: E^P (combinatorial) for coarse → E^λ (covariance) for fine
                </div>
              </>
            )}

            <label>
              {useMultiRes ? 'Iterations Per Level' : 'Max Iterations'}: {maxIterations}
              <input
                type="range"
                min="20"
                max="200"
                step="10"
                value={maxIterations}
                onChange={(e) => setMaxIterations(parseInt(e.target.value))}
                disabled={isOptimizing}
              />
              {useMultiRes && (
                <span className="hint" style={{ fontSize: '11px', opacity: 0.7 }}>
                  Each level gets {maxIterations} iterations
                </span>
              )}
            </label>
          </div>

          <div className="control-section">
            <h3>Actions</h3>
            <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
              <button
                onClick={handleOptimize}
                disabled={isOptimizing}
                style={{
                  padding: '6px 12px',
                  fontSize: '13px',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '6px',
                  background: isOptimizing ? '#666' : '#4CAF50',
                  color: 'white',
                  border: 'none',
                  borderRadius: '4px',
                  cursor: isOptimizing ? 'not-allowed' : 'pointer'
                }}
              >
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <polygon points="5 3 19 12 5 21 5 3"></polygon>
                </svg>
                {isOptimizing ? `Running (${progress.iteration})` : 'Run'}
              </button>
              {isOptimizing && (
                <button
                  onClick={handleStop}
                  style={{
                    padding: '6px 12px',
                    fontSize: '13px',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '6px',
                    background: '#f44336',
                    color: 'white',
                    border: 'none',
                    borderRadius: '4px',
                    cursor: 'pointer'
                  }}
                >
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
                    <rect x="6" y="6" width="12" height="12"></rect>
                  </svg>
                  Stop
                </button>
              )}
              <button
                onClick={handleReset}
                disabled={isOptimizing}
                style={{
                  padding: '6px 12px',
                  fontSize: '13px',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '6px',
                  background: isOptimizing ? '#ccc' : '#2196F3',
                  color: 'white',
                  border: 'none',
                  borderRadius: '4px',
                  cursor: isOptimizing ? 'not-allowed' : 'pointer'
                }}
              >
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"></path>
                  <path d="M3 3v5h5"></path>
                </svg>
                Reset
              </button>
            </div>
          </div>

          {history.length > 0 && (
            <div className="control-section">
              <h3>Animation</h3>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <button
                  onClick={() => setIsPlaying(!isPlaying)}
                  style={{
                    padding: '4px 8px',
                    fontSize: '12px',
                    background: '#2196F3',
                    color: 'white',
                    border: 'none',
                    borderRadius: '4px',
                    cursor: 'pointer',
                    display: 'flex',
                    alignItems: 'center'
                  }}
                >
                  {isPlaying ? (
                    <svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor">
                      <rect x="6" y="4" width="4" height="16"></rect>
                      <rect x="14" y="4" width="4" height="16"></rect>
                    </svg>
                  ) : (
                    <svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor">
                      <polygon points="5 3 19 12 5 21 5 3"></polygon>
                    </svg>
                  )}
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
