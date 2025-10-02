import { Value } from './Value';
import { compileGradientFunction } from './jit-compile-value';
import * as fs from 'fs';
import * as path from 'path';

interface BenchmarkResult {
  graphSize: number;
  iterations: number;
  traditionalTime: number;
  compiledTime: number;
  speedup: number;
}

function runBenchmark(): BenchmarkResult[] {
  const graphSizes = [3, 5, 10, 20, 50, 100];
  const iterationCounts = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000];

  const results: BenchmarkResult[] = [];

  console.log('Running comprehensive JIT benchmark...\n');
  console.log('Graph Size | Iterations | Traditional | Compiled | Speedup');
  console.log('-----------|------------|-------------|----------|--------');

  for (const size of graphSizes) {
    for (const iters of iterationCounts) {
      const traditionalTime = performance.now();
      for (let i = 0; i < iters; i++) {
        const vars = Array.from({ length: size }, (_, i) => new Value(i + 1, `x${i}`, true));
        let result = vars[0];
        for (let j = 1; j < size; j++) {
          if (j % 4 === 0) result = result.add(vars[j]);
          else if (j % 4 === 1) result = result.sub(vars[j]);
          else if (j % 4 === 2) result = result.mul(vars[j]);
          else result = result.div(vars[j]);
        }
        result.backward();
      }
      const traditionalDuration = performance.now() - traditionalTime;

      const compiledTime = performance.now();
      const vars = Array.from({ length: size }, (_, i) => {
        const v = new Value(i + 1, `x${i}`, true);
        v.paramName = `x${i}`;
        return v;
      });
      let result = vars[0];
      for (let j = 1; j < size; j++) {
        if (j % 4 === 0) result = result.add(vars[j]);
        else if (j % 4 === 1) result = result.sub(vars[j]);
        else if (j % 4 === 2) result = result.mul(vars[j]);
        else result = result.div(vars[j]);
      }

      const compiledFn = compileGradientFunction(result, vars);

      const inputValues = Array.from({ length: size }, (_, i) => i + 1);
      for (let i = 0; i < iters; i++) {
        compiledFn(...inputValues);
      }
      const compiledDuration = performance.now() - compiledTime;

      const speedup = traditionalDuration / compiledDuration;

      results.push({
        graphSize: size,
        iterations: iters,
        traditionalTime: traditionalDuration,
        compiledTime: compiledDuration,
        speedup
      });

      console.log(
        `${size.toString().padStart(10)} | ` +
        `${iters.toString().padStart(10)} | ` +
        `${traditionalDuration.toFixed(2).padStart(11)}ms | ` +
        `${compiledDuration.toFixed(2).padStart(8)}ms | ` +
        `${speedup.toFixed(2)}x`
      );
    }
  }

  return results;
}

function generateFiles(results: BenchmarkResult[]): void {
  const data = JSON.stringify(results, null, 2);
  const jsonPath = path.join(__dirname, '../docs/jit/data.json');

  fs.writeFileSync(jsonPath, data);
  console.log(`\n✓ Updated benchmark data: ${jsonPath}`);
}

function generateHTMLOld(results: BenchmarkResult[]): string {
  const data = JSON.stringify(results);

  return `<!DOCTYPE html>
<html>
<head>
  <title>JIT Compilation Performance Analysis</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <style>
    body {
      font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
      max-width: 1400px;
      margin: 0 auto;
      padding: 20px;
      background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
      min-height: 100vh;
    }
    .container {
      background: white;
      border-radius: 12px;
      padding: 30px;
      box-shadow: 0 10px 40px rgba(0,0,0,0.2);
    }
    h1 {
      color: #333;
      text-align: center;
      margin-bottom: 10px;
    }
    .subtitle {
      text-align: center;
      color: #666;
      margin-bottom: 30px;
      font-size: 18px;
    }
    .charts {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 30px;
      margin-bottom: 30px;
    }
    .chart-container {
      background: #f8f9fa;
      border-radius: 8px;
      padding: 20px;
      box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    .chart-container.full {
      grid-column: 1 / -1;
    }
    h2 {
      color: #444;
      margin-top: 0;
      font-size: 20px;
      margin-bottom: 15px;
    }
    .stats {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
      gap: 20px;
      margin-top: 30px;
    }
    .stat-card {
      background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
      color: white;
      padding: 20px;
      border-radius: 8px;
      box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    .stat-value {
      font-size: 32px;
      font-weight: bold;
      margin: 10px 0;
    }
    .stat-label {
      font-size: 14px;
      opacity: 0.9;
    }
    canvas {
      max-height: 400px;
    }
  </style>
</head>
<body>
  <div class="container">
    <h1>ScalarAutograd JIT Compilation Performance Analysis</h1>
    <p class="subtitle">Comparing Traditional Autodiff vs JIT-Compiled Gradient Functions</p>

    <div class="charts">
      <div class="chart-container full">
        <h2>Speedup Heatmap: JIT vs Traditional (lighter = faster)</h2>
        <canvas id="heatmap"></canvas>
      </div>

      <div class="chart-container">
        <h2>Speedup by Iteration Count</h2>
        <canvas id="iterationsChart"></canvas>
      </div>

      <div class="chart-container">
        <h2>Speedup by Graph Size</h2>
        <canvas id="graphSizeChart"></canvas>
      </div>
    </div>

    <div class="stats">
      <div class="stat-card">
        <div class="stat-label">Break-Even Point (Simple Graphs)</div>
        <div class="stat-value" id="breakEven">~5 iterations</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">Maximum Speedup Observed</div>
        <div class="stat-value" id="maxSpeedup">--</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">Average Speedup (>10 iterations)</div>
        <div class="stat-value" id="avgSpeedup">--</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">Best Use Case</div>
        <div class="stat-value" id="bestCase">--</div>
      </div>
    </div>
  </div>

  <script>
    const results = ${data};

    const graphSizes = [...new Set(results.map(r => r.graphSize))];
    const iterations = [...new Set(results.map(r => r.iterations))];

    const maxSpeedup = Math.max(...results.map(r => r.speedup));
    const avgSpeedupOver10 = results
      .filter(r => r.iterations > 10)
      .reduce((sum, r) => sum + r.speedup, 0) / results.filter(r => r.iterations > 10).length;

    const bestResult = results.reduce((best, r) => r.speedup > best.speedup ? r : best);

    document.getElementById('maxSpeedup').textContent = maxSpeedup.toFixed(1) + 'x';
    document.getElementById('avgSpeedup').textContent = avgSpeedupOver10.toFixed(1) + 'x';
    document.getElementById('bestCase').textContent =
      \`\${bestResult.graphSize} vars, \${bestResult.iterations} iters\`;

    const smallGraphResults = results.filter(r => r.graphSize <= 10);
    let breakEven = 1;
    for (const r of smallGraphResults.sort((a, b) => a.iterations - b.iterations)) {
      if (r.speedup > 1.0) {
        breakEven = r.iterations;
        break;
      }
    }
    document.getElementById('breakEven').textContent = \`~\${breakEven} iterations\`;

    const heatmapData = {
      labels: iterations,
      datasets: graphSizes.map((size, idx) => {
        const color = \`hsl(\${(idx * 360) / graphSizes.length}, 70%, 50%)\`;
        return {
          label: \`\${size} variables\`,
          data: iterations.map(iter => {
            const result = results.find(r => r.graphSize === size && r.iterations === iter);
            return { x: iter, y: result ? result.speedup : null };
          }),
          backgroundColor: color,
          borderColor: color,
          borderWidth: 1
        };
      })
    };

    new Chart(document.getElementById('heatmap'), {
      type: 'line',
      data: heatmapData,
      options: {
        responsive: true,
        maintainAspectRatio: true,
        scales: {
          x: {
            type: 'logarithmic',
            title: { display: true, text: 'Iterations (log scale)' }
          },
          y: {
            type: 'logarithmic',
            title: { display: true, text: 'Speedup (log scale)' },
            ticks: {
              callback: function(value) {
                return value.toFixed(1) + 'x';
              }
            }
          }
        },
        plugins: {
          tooltip: {
            callbacks: {
              label: function(context) {
                return context.dataset.label + ': ' + context.parsed.y.toFixed(2) + 'x speedup';
              }
            }
          },
          annotation: {
            annotations: {
              line1: {
                type: 'line',
                yMin: 1,
                yMax: 1,
                borderColor: 'red',
                borderWidth: 2,
                borderDash: [5, 5],
                label: {
                  content: 'Break-even (1x)',
                  enabled: true,
                  position: 'end'
                }
              }
            }
          }
        }
      }
    });

    const iterData = {
      labels: iterations,
      datasets: graphSizes.map((size, idx) => {
        const color = \`hsl(\${(idx * 360) / graphSizes.length}, 70%, 50%)\`;
        return {
          label: \`\${size} vars\`,
          data: iterations.map(iter => {
            const result = results.find(r => r.graphSize === size && r.iterations === iter);
            return result ? result.speedup : null;
          }),
          borderColor: color,
          backgroundColor: color,
          tension: 0.3
        };
      })
    };

    new Chart(document.getElementById('iterationsChart'), {
      type: 'line',
      data: iterData,
      options: {
        responsive: true,
        maintainAspectRatio: true,
        scales: {
          x: { type: 'logarithmic', title: { display: true, text: 'Iterations' } },
          y: {
            title: { display: true, text: 'Speedup' },
            ticks: { callback: v => v.toFixed(1) + 'x' }
          }
        }
      }
    });

    const sizeData = {
      labels: graphSizes,
      datasets: iterations.filter(i => [1, 10, 100, 1000].includes(i)).map((iter, idx) => {
        const colors = ['#e74c3c', '#f39c12', '#2ecc71', '#3498db'];
        return {
          label: \`\${iter} iterations\`,
          data: graphSizes.map(size => {
            const result = results.find(r => r.graphSize === size && r.iterations === iter);
            return result ? result.speedup : null;
          }),
          borderColor: colors[idx],
          backgroundColor: colors[idx],
          tension: 0.3
        };
      })
    };

    new Chart(document.getElementById('graphSizeChart'), {
      type: 'line',
      data: sizeData,
      options: {
        responsive: true,
        maintainAspectRatio: true,
        scales: {
          x: { title: { display: true, text: 'Graph Size (variables)' } },
          y: {
            title: { display: true, text: 'Speedup' },
            ticks: { callback: v => v.toFixed(1) + 'x' }
          }
        }
      }
    });
  </script>
</body>
</html>`;
}

if (require.main === module) {
  console.log('Starting benchmark...\n');
  const results = runBenchmark();

  generateFiles(results);

  const htmlPath = path.join(__dirname, '../docs/jit/index.html');
  console.log(`✓ View results: ${htmlPath}`);
  console.log(`✓ Or visit: http://localhost:8899/docs/jit/`);
}

export { runBenchmark, generateFiles };
