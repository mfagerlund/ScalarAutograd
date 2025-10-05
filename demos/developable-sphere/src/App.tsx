import { useEffect, useRef, useState } from 'react';
import { IcoSphere } from './mesh/IcoSphere';
import { DevelopableOptimizer } from './optimization/DevelopableOptimizer';
import { CurvatureClassifier } from './energy/CurvatureClassifier';
import { MeshRenderer } from './visualization/MeshRenderer';
import { TriangleMesh } from './mesh/TriangleMesh';
import { EnergyRegistry } from './energy/EnergyRegistry';
import { useLocalStorage } from './hooks/useLocalStorage';
import './App.css';

// Import all energies to ensure they register
import './energy/DevelopableEnergy';
import './energy/PaperCovarianceEnergyELambda';
import './energy/RidgeBasedEnergy';
import './energy/AlignmentBimodalEnergy';
import './energy/PaperPartitionEnergyEP';
import './energy/EigenProxyEnergy';
import './energy/StochasticCovarianceEnergy';
import './energy/FastCovarianceEnergy';
import './energy/GreatCircleEnergyEx';
import './energy/DifferentiablePlaneAlignment';

const VERSION = 'v0';

export default function App() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [renderer, setRenderer] = useState<MeshRenderer | null>(null);

  const [isOptimizing, setIsOptimizing] = useState(false);
  const [isCompiling, setIsCompiling] = useState(false);
  const [progress, setProgress] = useState({ iteration: 0, energy: 0 });
  const [compileProgress, setCompileProgress] = useState({ current: 0, total: 0, percent: 0 });
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

  // Batch runner state
  const [showBatchModal, setShowBatchModal] = useState(false);
  const [selectedModels, setSelectedModels] = useState<Set<string>>(new Set());
  const [isBatchRunning, setIsBatchRunning] = useState(false);
  const [batchProgress, setBatchProgress] = useState({ current: 0, total: 0, currentModel: '' });
  const batchStopRef = useRef(false);

  // Persisted settings (stored in localStorage)
  const [subdivisions, setSubdivisions] = useLocalStorage('subdivisions', 3);
  const [maxIterations, setMaxIterations] = useLocalStorage('maxIterations', 50);
  const [energyType, setEnergyType] = useLocalStorage<string>('energyType', EnergyRegistry.getNames()[0] || 'bimodal');
  const [useCompiled, setUseCompiled] = useLocalStorage('useCompiled', false);
  const [optimizer, setOptimizer] = useLocalStorage<'lbfgs' | 'leastsquares'>('optimizer', 'lbfgs');

  // Non-persisted state
  const [iterationsPerSecond, setIterationsPerSecond] = useState<number>(0);

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

  const [developableHistory, setDevelopableHistory] = useState<Array<{iteration: number, percent: number}>>([]);

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

    // Check if energy supports compilation
    const energyFunction = EnergyRegistry.get(energyType);
    const canCompile = energyFunction?.supportsCompilation ?? true;
    const shouldCompile = useCompiled && canCompile;

    // Set state and give UI a chance to update
    setIsOptimizing(true);
    setIsCompiling(shouldCompile);
    setHistory([]);
    setProgress({ iteration: 0, energy: 0 });
    setCompileProgress({ current: 0, total: 0, percent: 0 });
    setConvergenceInfo(null);
    setIterationsPerSecond(0);
    meshHistoryRef.current = [];
    setDevelopableHistory([]);

    // Yield to browser to update UI (0ms = yield to event loop)
    await new Promise(resolve => setTimeout(resolve, 0));

    const optimizationStartTime = Date.now();

    try {
      // Create fresh sphere
      const sphere = IcoSphere.generate(subdivisions, 1.0);

      // Capture initial developability
      const initialClassification = CurvatureClassifier.classifyVertices(sphere);
      const developableBefore = initialClassification.hingeVertices.length / sphere.vertices.length;
      const regionsBefore = CurvatureClassifier.countDevelopableRegions(sphere);

      // Immediately render the fresh sphere
      renderer.updateMesh(sphere, initialClassification);
      renderer.render();
      updateMetrics(sphere);
      setCurrentMesh(sphere);

      const opt = new DevelopableOptimizer(sphere);
      optimizerRef.current = opt;

      // Run async optimization (non-blocking)
      const result = await opt.optimizeAsync({
        maxIterations,
        gradientTolerance: 1e-8,
        verbose: true,
        captureInterval: Math.max(1, Math.floor(maxIterations / 20)),
        chunkSize: 5, // Fixed chunk size
        energyType,
        useCompiled: shouldCompile,
        optimizer,
        onCompileProgress: (current, total, percent) => {
          setCompileProgress({ current, total, percent });
        },
        onCompileComplete: () => {
          setIsCompiling(false);
        },
        onProgress: (iteration, energy, history) => {
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

            const devPercent = (classification.hingeVertices.length / latestMesh.vertices.length) * 100;
            setDevelopableHistory(prev => [...prev, { iteration, percent: devPercent }]);
          }
        },
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
      console.log(`Energy Function: ${energyType}`);
      console.log(`Mode: ${shouldCompile ? 'Compiled' : 'Non-compiled'}${!canCompile && useCompiled ? ' (energy does not support compilation)' : ''}`);
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
      console.log(`  Iterations: ${result.iterations} / ${maxIterations}`);
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

  const handleBatchRun = async () => {
    if (!renderer || isBatchRunning || selectedModels.size === 0) return;

    const models = Array.from(selectedModels);
    const results: Array<{
      modelName: string;
      energyType: string;
      optimizer: string;
      subdivisions: number;
      maxIterations: number;
      useCompiled: boolean;
      compilationTime?: number;
      iterations: number;
      timeElapsed: number;
      developableBefore: number;
      developableAfter: number;
      regionsBefore: number;
      regionsAfter: number;
      convergenceReason: string;
      functionEvals?: number;
      imageName: string;
      imageData: string;
    }> = [];

    setIsBatchRunning(true);
    setShowBatchModal(false);
    setBatchProgress({ current: 0, total: models.length, currentModel: '' });
    batchStopRef.current = false;

    try {
      for (let i = 0; i < models.length; i++) {
        if (batchStopRef.current) {
          console.log('Batch run stopped by user');
          break;
        }

        const modelName = models[i];
        setBatchProgress({ current: i, total: models.length, currentModel: modelName });

        const energyFunction = EnergyRegistry.get(modelName);
        const canCompile = energyFunction?.supportsCompilation ?? true;
        const shouldCompile = canCompile; // Use what works for each model

        // Create fresh sphere
        const sphere = IcoSphere.generate(subdivisions, 1.0);

        // Capture initial developability
        const initialClassification = CurvatureClassifier.classifyVertices(sphere);
        const developableBefore = initialClassification.hingeVertices.length / sphere.vertices.length;
        const regionsBefore = CurvatureClassifier.countDevelopableRegions(sphere);

        // Update display
        renderer.updateMesh(sphere, initialClassification);
        renderer.render();
        await new Promise(resolve => setTimeout(resolve, 0));

        const opt = new DevelopableOptimizer(sphere);
        const startTime = Date.now();

        // Run optimization
        const result = await opt.optimizeAsync({
          maxIterations,
          gradientTolerance: 1e-8,
          verbose: false,
          captureInterval: Math.max(1, Math.floor(maxIterations / 20)),
          chunkSize: 5,
          energyType: modelName,
          useCompiled: shouldCompile,
          optimizer,
          onProgress: () => {}, // Silent
        });

        const timeElapsed = (Date.now() - startTime) / 1000;

        // Get final mesh
        const finalMesh = result.history[result.history.length - 1];
        const finalClassification = CurvatureClassifier.classifyVertices(finalMesh);
        const developableAfter = finalClassification.hingeVertices.length / finalMesh.vertices.length;
        const regionsAfter = CurvatureClassifier.countDevelopableRegions(finalMesh);

        // Render final mesh and IMMEDIATELY capture screenshot (must be synchronous)
        renderer.updateMesh(finalMesh, finalClassification);
        renderer.render();

        // Must call toDataURL() immediately after render(), in same tick
        const canvas = canvasRef.current;
        let imageData = '';
        if (canvas) {
          imageData = canvas.toDataURL('image/png');
          console.log(`Captured image for ${modelName}, size: ${imageData.length} bytes`);
        }

        // Now yield to UI
        await new Promise(resolve => setTimeout(resolve, 0));

        if (canvas) {
          const imageName = `${modelName.replace(/[^a-zA-Z0-9]/g, '_')}_${Date.now()}.png`;

          results.push({
            modelName,
            energyType: modelName,
            optimizer,
            subdivisions,
            maxIterations,
            useCompiled: shouldCompile,
            compilationTime: result.compilationTime,
            iterations: result.iterations,
            timeElapsed,
            developableBefore,
            developableAfter,
            regionsBefore,
            regionsAfter,
            convergenceReason: result.convergenceReason,
            functionEvals: result.functionEvaluations,
            imageName,
            imageData,
          });

          // Download image
          const link = document.createElement('a');
          link.download = imageName;
          link.href = imageData;
          link.click();
        }
      }

      // Export JSON results
      const jsonData = {
        timestamp: new Date().toISOString(),
        settings: {
          subdivisions,
          maxIterations,
          optimizer,
        },
        results,
      };

      const blob = new Blob([JSON.stringify(jsonData, null, 2)], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.download = `batch_results_${Date.now()}.json`;
      link.href = url;
      link.click();
      URL.revokeObjectURL(url);

      // Generate HTML viewer with embedded images
      const htmlContent = `<!DOCTYPE html>
<html>
<head>
  <title>Batch Results - ${new Date().toISOString()}</title>
  <style>
    body { font-family: Arial, sans-serif; padding: 20px; background: #f5f5f5; }
    h1 { color: #333; }
    .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(400px, 1fr)); gap: 20px; }
    .card { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
    .card h2 { margin-top: 0; color: #2196F3; }
    .card img { width: 100%; border-radius: 4px; margin: 10px 0; box-shadow: 0 2px 8px rgba(0,0,0,0.15); }
    .metric { display: flex; justify-content: space-between; padding: 4px 0; border-bottom: 1px solid #eee; }
    .metric:last-child { border-bottom: none; }
    .label { font-weight: bold; color: #666; }
    .value { color: #333; }
    .positive { color: #4CAF50; }
    .negative { color: #f44336; }
  </style>
</head>
<body>
  <h1>Batch Optimization Results</h1>
  <p>Generated: ${new Date().toISOString()}</p>
  <p>Settings: ${subdivisions} subdivisions, ${maxIterations} max iterations, ${optimizer} optimizer</p>
  <div class="grid">
    ${results.map(r => `
      <div class="card">
        <h2>${r.modelName}</h2>
        ${r.imageData ? `<img src="${r.imageData}" alt="${r.modelName} result" />` : ''}
        <div class="metric"><span class="label">Optimizer:</span><span class="value">${r.optimizer}</span></div>
        <div class="metric"><span class="label">Compiled:</span><span class="value">${r.useCompiled ? 'Yes' : 'No'}</span></div>
        ${r.compilationTime ? `<div class="metric"><span class="label">Compilation Time:</span><span class="value">${r.compilationTime.toFixed(2)}s</span></div>` : ''}
        <div class="metric"><span class="label">Iterations:</span><span class="value">${r.iterations} / ${r.maxIterations}</span></div>
        <div class="metric"><span class="label">Time Elapsed:</span><span class="value">${r.timeElapsed.toFixed(2)}s</span></div>
        <div class="metric"><span class="label">Speed:</span><span class="value">${(r.iterations / r.timeElapsed).toFixed(1)} it/s</span></div>
        <div class="metric"><span class="label">Function Evals:</span><span class="value">${r.functionEvals || 'N/A'}</span></div>
        <div class="metric"><span class="label">Developability Before:</span><span class="value">${(r.developableBefore * 100).toFixed(2)}% (${r.regionsBefore} regions)</span></div>
        <div class="metric"><span class="label">Developability After:</span><span class="value">${(r.developableAfter * 100).toFixed(2)}% (${r.regionsAfter} regions)</span></div>
        <div class="metric"><span class="label">Change:</span><span class="value ${r.developableAfter > r.developableBefore ? 'positive' : 'negative'}">${((r.developableAfter - r.developableBefore) * 100).toFixed(2)}%</span></div>
        <div class="metric"><span class="label">Convergence:</span><span class="value">${r.convergenceReason}</span></div>
      </div>
    `).join('')}
  </div>
  <script>
    const results = ${JSON.stringify(jsonData, null, 2)};
    console.log('Batch results:', results);
  </script>
</body>
</html>`;

      const htmlBlob = new Blob([htmlContent], { type: 'text/html' });
      const htmlUrl = URL.createObjectURL(htmlBlob);
      const htmlLink = document.createElement('a');
      htmlLink.download = `batch_results_${Date.now()}.html`;
      htmlLink.href = htmlUrl;
      htmlLink.click();
      URL.revokeObjectURL(htmlUrl);

      console.log('Batch run complete!', results);
      if (!batchStopRef.current) {
        alert(`Batch run complete! ${results.length} models processed.\nResults saved as HTML and JSON files.`);
      } else {
        alert(`Batch run stopped! ${results.length} of ${models.length} models processed.\nPartial results saved.`);
      }

    } finally {
      setIsBatchRunning(false);
      setBatchProgress({ current: 0, total: 0, currentModel: '' });
      batchStopRef.current = false;
    }
  };

  const handleStopBatch = () => {
    batchStopRef.current = true;
  };

  const handleToggleModel = (modelName: string) => {
    const newSelected = new Set(selectedModels);
    if (newSelected.has(modelName)) {
      newSelected.delete(modelName);
    } else {
      newSelected.add(modelName);
    }
    setSelectedModels(newSelected);
  };

  const handleSelectAll = () => {
    setSelectedModels(new Set(EnergyRegistry.getNames()));
  };

  const handleSelectNone = () => {
    setSelectedModels(new Set());
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
            background: isOptimizing
              ? 'linear-gradient(45deg, rgba(27, 94, 32, 0.95) 25%, rgba(56, 142, 60, 0.95) 25%, rgba(56, 142, 60, 0.95) 50%, rgba(27, 94, 32, 0.95) 50%, rgba(27, 94, 32, 0.95) 75%, rgba(56, 142, 60, 0.95) 75%)'
              : 'rgba(0, 0, 0, 0.7)',
            backgroundSize: isOptimizing ? '40px 40px' : 'auto',
            animation: isOptimizing ? 'slide 1s linear infinite' : 'none',
            color: 'white',
            padding: '12px',
            borderRadius: '4px',
            fontSize: '12px',
            pointerEvents: 'none',
            minWidth: '200px',
            border: isOptimizing ? '2px solid rgba(76, 175, 80, 1)' : 'none',
            textShadow: isOptimizing ? '0 1px 3px rgba(0, 0, 0, 0.8)' : 'none'
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
                    <div>Compiling {compileProgress.current} / {compileProgress.total} functions</div>
                    <div style={{ fontSize: '11px', opacity: 0.7, marginTop: '2px' }}>
                      {compileProgress.percent}% complete
                    </div>
                    <div style={{
                      marginTop: '6px',
                      background: 'rgba(255,255,255,0.2)',
                      borderRadius: '2px',
                      height: '4px',
                      overflow: 'hidden'
                    }}>
                      <div style={{
                        width: `${compileProgress.percent}%`,
                        height: '100%',
                        background: '#FFA726',
                        transition: 'width 0.3s'
                      }} />
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

          {/* Developable% History Chart - Top Right */}
          {developableHistory.length > 1 && (
            <div style={{
              position: 'absolute',
              top: '10px',
              right: '10px',
              background: 'rgba(0, 0, 0, 0.7)',
              color: 'white',
              padding: '12px',
              borderRadius: '4px',
              pointerEvents: 'none',
              width: '480px',
              height: '160px'
            }}>
              <div style={{ fontSize: '12px', fontWeight: 'bold', marginBottom: '8px' }}>
                Developable %
              </div>
              <svg width="456" height="110" style={{ background: 'rgba(255,255,255,0.05)', borderRadius: '2px' }}>
                {(() => {
                  const data = developableHistory;
                  const width = 456;
                  const height = 110;
                  const padding = 5;
                  const chartWidth = width - padding * 2;
                  const chartHeight = height - padding * 2;

                  const minPercent = Math.min(...data.map(d => d.percent));
                  const maxPercent = Math.max(...data.map(d => d.percent));
                  const rangePercent = maxPercent - minPercent || 1;

                  const maxIteration = Math.max(...data.map(d => d.iteration));

                  const points = data.map((d) => {
                    const x = padding + (d.iteration / maxIteration) * chartWidth;
                    const y = padding + chartHeight - ((d.percent - minPercent) / rangePercent) * chartHeight;
                    return `${x},${y}`;
                  }).join(' ');

                  return (
                    <>
                      <polyline
                        points={points}
                        fill="none"
                        stroke="#4CAF50"
                        strokeWidth="2"
                      />
                      <line x1={padding} y1={height - padding} x2={width - padding} y2={height - padding} stroke="rgba(255,255,255,0.2)" strokeWidth="1" />
                      <line x1={padding} y1={padding} x2={padding} y2={height - padding} stroke="rgba(255,255,255,0.2)" strokeWidth="1" />

                      <text x={padding + 2} y={padding + 10} fill="white" fontSize="10" opacity="0.7">
                        {maxPercent.toFixed(1)}%
                      </text>
                      <text x={padding + 2} y={height - padding - 2} fill="white" fontSize="10" opacity="0.7">
                        {minPercent.toFixed(1)}%
                      </text>
                    </>
                  );
                })()}
              </svg>
            </div>
          )}

          {/* CSS for animations */}
          <style>{`
            @keyframes spin {
              0% { transform: rotate(0deg); }
              100% { transform: rotate(360deg); }
            }
            @keyframes slide {
              0% { background-position: 0 0; }
              100% { background-position: 40px 40px; }
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
                style={{
                  fontFamily: 'monospace',
                  fontSize: '12px',
                  backgroundColor: isOptimizing ? '#fff' : undefined,
                  color: isOptimizing ? '#000' : undefined,
                  fontWeight: isOptimizing ? 'bold' : undefined
                }}
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
                style={{
                  fontFamily: 'monospace',
                  fontSize: '12px',
                  backgroundColor: isOptimizing ? '#fff' : undefined,
                  color: isOptimizing ? '#000' : undefined,
                  fontWeight: isOptimizing ? 'bold' : undefined
                }}
                title={EnergyRegistry.get(energyType)?.description || ''}
              >
                {EnergyRegistry.getAll().map((energy) => {
                  return (
                    <option key={energy.name} value={energy.name} title={energy.description}>
                      {energy.name}{energy.className ? ` (${energy.className})` : ''}
                    </option>
                  );
                })}
              </select>
            </label>
            {EnergyRegistry.get(energyType)?.description && (
              <div style={{
                fontSize: '11px',
                color: '#888',
                marginTop: '-4px',
                marginBottom: '4px',
                fontFamily: 'monospace'
              }}>
                {EnergyRegistry.get(energyType)?.description}
              </div>
            )}

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
                checked={useCompiled && (EnergyRegistry.get(energyType)?.supportsCompilation ?? true)}
                onChange={(e) => setUseCompiled(e.target.checked)}
                disabled={isOptimizing || !(EnergyRegistry.get(energyType)?.supportsCompilation ?? true)}
              />
              Use Compiled Gradients (faster)
              {!(EnergyRegistry.get(energyType)?.supportsCompilation ?? true) && (
                <span style={{ fontSize: '11px', color: '#f44336', marginLeft: '8px' }}>
                  (not supported for this energy)
                </span>
              )}
            </label>

            <label>
              Subdivisions: {subdivisions}
              <div style={{ position: 'relative' }}>
                <div style={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  position: 'absolute',
                  width: '100%',
                  top: '9px',
                  pointerEvents: 'none'
                }}>
                  {[0, 1, 2, 3, 4, 5, 6].map(v => (
                    <div key={v} style={{
                      width: '6px',
                      height: '6px',
                      borderRadius: '50%',
                      backgroundColor: '#808080',
                      marginLeft: v === 0 ? '0' : '-3px',
                      marginRight: v === 6 ? '0' : '-3px'
                    }} />
                  ))}
                </div>
                <input
                  type="range"
                  min="0"
                  max="6"
                  value={subdivisions}
                  onChange={(e) => setSubdivisions(parseInt(e.target.value))}
                  disabled={isOptimizing}
                  list="subdivisions-ticks"
                />
                <datalist id="subdivisions-ticks">
                  <option value="0" label="0"></option>
                  <option value="1" label="1"></option>
                  <option value="2" label="2"></option>
                  <option value="3" label="3"></option>
                  <option value="4" label="4"></option>
                  <option value="5" label="5"></option>
                  <option value="6" label="6"></option>
                </datalist>
                <div className="slider-ticks" style={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  marginTop: '4px',
                  pointerEvents: 'none'
                }}>
                  {[0, 1, 2, 3, 4, 5, 6].map(v => (
                    <span key={v} style={{
                      fontSize: '10px',
                      color: '#808080',
                      width: '8px',
                      textAlign: 'center'
                    }}>{v}</span>
                  ))}
                </div>
              </div>
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

            <label>
              Max Iterations: {maxIterations}
              <div style={{ position: 'relative' }}>
                <div style={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  position: 'absolute',
                  width: '100%',
                  top: '9px',
                  pointerEvents: 'none'
                }}>
                  {[20, 50, 100, 150, 200, 300].map(v => (
                    <div key={v} style={{
                      width: '6px',
                      height: '6px',
                      borderRadius: '50%',
                      backgroundColor: '#808080',
                      marginLeft: v === 20 ? '0' : '-3px',
                      marginRight: v === 300 ? '0' : '-3px'
                    }} />
                  ))}
                </div>
                <input
                  type="range"
                  min="20"
                  max="300"
                  step="10"
                  value={maxIterations}
                  onChange={(e) => setMaxIterations(parseInt(e.target.value))}
                  disabled={isOptimizing}
                  list="iterations-ticks"
                />
                <datalist id="iterations-ticks">
                  <option value="20" label="20"></option>
                  <option value="50" label="50"></option>
                  <option value="100" label="100"></option>
                  <option value="150" label="150"></option>
                  <option value="200" label="200"></option>
                  <option value="300" label="300"></option>
                </datalist>
                <div className="slider-ticks" style={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  marginTop: '4px',
                  pointerEvents: 'none'
                }}>
                  {[20, 50, 100, 150, 200, 300].map(v => (
                    <span key={v} style={{
                      fontSize: '10px',
                      color: '#808080',
                      width: '20px',
                      textAlign: 'center'
                    }}>{v}</span>
                  ))}
                </div>
              </div>
            </label>
          </div>

          <div className="control-section">
            <h3>Actions</h3>
            <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
              <button
                onClick={handleOptimize}
                disabled={isOptimizing || isBatchRunning}
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
                  cursor: isOptimizing || isBatchRunning ? 'not-allowed' : 'pointer'
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
                disabled={isOptimizing || isBatchRunning}
                style={{
                  padding: '6px 12px',
                  fontSize: '13px',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '6px',
                  background: isOptimizing || isBatchRunning ? '#ccc' : '#2196F3',
                  color: 'white',
                  border: 'none',
                  borderRadius: '4px',
                  cursor: isOptimizing || isBatchRunning ? 'not-allowed' : 'pointer'
                }}
              >
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"></path>
                  <path d="M3 3v5h5"></path>
                </svg>
                Reset
              </button>
              <button
                onClick={() => setShowBatchModal(true)}
                disabled={isOptimizing || isBatchRunning}
                style={{
                  padding: '6px 12px',
                  fontSize: '13px',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '6px',
                  background: isBatchRunning ? '#666' : '#FF9800',
                  color: 'white',
                  border: 'none',
                  borderRadius: '4px',
                  cursor: isOptimizing || isBatchRunning ? 'not-allowed' : 'pointer'
                }}
              >
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <rect x="3" y="3" width="7" height="7"></rect>
                  <rect x="14" y="3" width="7" height="7"></rect>
                  <rect x="14" y="14" width="7" height="7"></rect>
                  <rect x="3" y="14" width="7" height="7"></rect>
                </svg>
                {isBatchRunning ? `Batch (${batchProgress.current}/${batchProgress.total})` : 'Batch Run'}
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

      {/* Batch Progress Overlay */}
      {isBatchRunning && (
        <div style={{
          position: 'fixed',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          background: 'rgba(0, 0, 0, 0.8)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          zIndex: 1000,
        }}>
          <div style={{
            background: 'white',
            padding: '30px',
            borderRadius: '8px',
            minWidth: '400px',
            textAlign: 'center',
          }}>
            <div className="spinner" style={{
              width: '40px',
              height: '40px',
              border: '4px solid rgba(0,0,0,0.1)',
              borderTop: '4px solid #FF9800',
              borderRadius: '50%',
              animation: 'spin 0.8s linear infinite',
              margin: '0 auto 20px',
            }} />
            <h2 style={{ margin: '0 0 10px', color: '#333' }}>Running Batch Optimization</h2>
            <p style={{ margin: '0 0 20px', color: '#666' }}>
              Processing {batchProgress.current + 1} of {batchProgress.total}
            </p>
            <p style={{ margin: '0 0 20px', color: '#FF9800', fontWeight: 'bold' }}>
              {batchProgress.currentModel}
            </p>
            <div style={{
              background: '#f0f0f0',
              borderRadius: '4px',
              height: '8px',
              overflow: 'hidden',
              marginBottom: '10px',
            }}>
              <div style={{
                width: `${((batchProgress.current) / batchProgress.total) * 100}%`,
                height: '100%',
                background: '#FF9800',
                transition: 'width 0.3s',
              }} />
            </div>
            <p style={{ margin: 0, fontSize: '12px', color: '#999' }}>
              {Math.round(((batchProgress.current) / batchProgress.total) * 100)}% complete
            </p>
            <button
              onClick={handleStopBatch}
              style={{
                marginTop: '20px',
                padding: '10px 20px',
                background: '#f44336',
                color: 'white',
                border: 'none',
                borderRadius: '4px',
                cursor: 'pointer',
                fontSize: '14px',
                fontWeight: 'bold',
              }}
            >
              Stop Batch
            </button>
          </div>
        </div>
      )}

      {/* Batch Model Selection Modal */}
      {showBatchModal && (
        <div style={{
          position: 'fixed',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          background: 'rgba(0, 0, 0, 0.5)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          zIndex: 1000,
        }}>
          <div style={{
            background: 'white',
            padding: '30px',
            borderRadius: '8px',
            width: '90%',
            maxWidth: '1200px',
            maxHeight: '90vh',
            overflow: 'auto',
          }}>
            <h2 style={{ margin: '0 0 20px', color: '#333' }}>Select Models for Batch Run</h2>

            <div style={{ marginBottom: '20px', display: 'flex', gap: '10px' }}>
              <button
                onClick={handleSelectAll}
                style={{
                  padding: '8px 16px',
                  background: '#2196F3',
                  color: 'white',
                  border: 'none',
                  borderRadius: '4px',
                  cursor: 'pointer',
                  fontSize: '13px',
                }}
              >
                Select All
              </button>
              <button
                onClick={handleSelectNone}
                style={{
                  padding: '8px 16px',
                  background: '#757575',
                  color: 'white',
                  border: 'none',
                  borderRadius: '4px',
                  cursor: 'pointer',
                  fontSize: '13px',
                }}
              >
                Clear All
              </button>
            </div>

            <div style={{
              border: '1px solid #ddd',
              borderRadius: '4px',
              overflow: 'auto',
            }}>
              <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                <thead>
                  <tr style={{ background: '#f5f5f5', borderBottom: '2px solid #ddd' }}>
                    <th style={{ padding: '8px', textAlign: 'left', width: '40px' }}>
                      <input
                        type="checkbox"
                        checked={selectedModels.size === EnergyRegistry.getAll().length}
                        onChange={(e) => e.target.checked ? handleSelectAll() : handleSelectNone()}
                        style={{ cursor: 'pointer' }}
                      />
                    </th>
                    <th style={{ padding: '8px', textAlign: 'left', fontWeight: 'bold', color: '#333' }}>Model</th>
                    <th style={{ padding: '8px', textAlign: 'left', fontWeight: 'bold', color: '#333' }}>Description</th>
                    <th style={{ padding: '8px', textAlign: 'center', fontWeight: 'bold', color: '#333', width: '120px' }}>Compilation</th>
                  </tr>
                </thead>
                <tbody>
                  {EnergyRegistry.getAll().map((energy) => (
                    <tr
                      key={energy.name}
                      onClick={() => handleToggleModel(energy.name)}
                      style={{
                        cursor: 'pointer',
                        borderBottom: '1px solid #eee',
                        background: selectedModels.has(energy.name) ? '#e3f2fd' : 'white',
                        transition: 'background 0.1s',
                      }}
                      onMouseEnter={(e) => {
                        if (!selectedModels.has(energy.name)) {
                          e.currentTarget.style.background = '#f5f5f5';
                        }
                      }}
                      onMouseLeave={(e) => {
                        e.currentTarget.style.background = selectedModels.has(energy.name) ? '#e3f2fd' : 'white';
                      }}
                    >
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}>
                        <input
                          type="checkbox"
                          checked={selectedModels.has(energy.name)}
                          onChange={() => handleToggleModel(energy.name)}
                          style={{ cursor: 'pointer' }}
                          onClick={(e) => e.stopPropagation()}
                        />
                      </td>
                      <td style={{ padding: '6px 8px', fontWeight: 'bold', color: '#333', fontFamily: 'monospace', fontSize: '13px' }}>
                        {energy.name}
                      </td>
                      <td style={{ padding: '6px 8px', fontSize: '12px', color: '#666' }}>
                        {energy.description}
                      </td>
                      <td style={{ padding: '6px 8px', textAlign: 'center', fontSize: '11px' }}>
                        <span style={{
                          padding: '4px 8px',
                          borderRadius: '4px',
                          background: energy.supportsCompilation ? '#e8f5e9' : '#fff3e0',
                          color: energy.supportsCompilation ? '#2e7d32' : '#e65100',
                          fontWeight: 'bold',
                        }}>
                          {energy.supportsCompilation ? '✓ Yes' : '⚠ No'}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            <div style={{
              marginTop: '20px',
              padding: '15px',
              background: '#f5f5f5',
              borderRadius: '4px',
            }}>
              <h4 style={{ margin: '0 0 10px', fontSize: '14px', color: '#333' }}>Batch Settings</h4>
              <div style={{ fontSize: '13px', color: '#666' }}>
                <div>Subdivisions: <strong>{subdivisions}</strong></div>
                <div>Max Iterations: <strong>{maxIterations}</strong></div>
                <div>Optimizer: <strong>{optimizer}</strong></div>
                <div>Selected Models: <strong>{selectedModels.size}</strong></div>
              </div>
            </div>

            <div style={{ marginTop: '20px', display: 'flex', gap: '10px', justifyContent: 'flex-end' }}>
              <button
                onClick={() => setShowBatchModal(false)}
                style={{
                  padding: '10px 20px',
                  background: '#757575',
                  color: 'white',
                  border: 'none',
                  borderRadius: '4px',
                  cursor: 'pointer',
                  fontSize: '14px',
                }}
              >
                Cancel
              </button>
              <button
                onClick={handleBatchRun}
                disabled={selectedModels.size === 0}
                style={{
                  padding: '10px 20px',
                  background: selectedModels.size === 0 ? '#ccc' : '#FF9800',
                  color: 'white',
                  border: 'none',
                  borderRadius: '4px',
                  cursor: selectedModels.size === 0 ? 'not-allowed' : 'pointer',
                  fontSize: '14px',
                  fontWeight: 'bold',
                }}
              >
                Run {selectedModels.size} Model{selectedModels.size !== 1 ? 's' : ''}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
