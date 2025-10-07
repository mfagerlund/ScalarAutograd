import { useState, useEffect } from 'react';

export interface BatchResult {
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
  imageUrl: string;
}

export interface BatchResultsData {
  timestamp: string;
  settings: {
    subdivisions: number;
    maxIterations: number;
    optimizer: string;
  };
  results: BatchResult[];
}

type SortField = 'modelName' | 'iterations' | 'timeElapsed' | 'developableAfter' | 'developableChange';
type SortDirection = 'asc' | 'desc';

export function BatchResults() {
  const [data, setData] = useState<BatchResultsData | null>(null);
  const [availableRuns, setAvailableRuns] = useState<string[]>([]);
  const [selectedRun, setSelectedRun] = useState<string>('');
  const [sortField, setSortField] = useState<SortField>('modelName');
  const [sortDirection, setSortDirection] = useState<SortDirection>('asc');
  const [filterText, setFilterText] = useState('');

  useEffect(() => {
    loadAvailableRuns();
  }, []);

  useEffect(() => {
    if (selectedRun) {
      loadResults(selectedRun);
    }
  }, [selectedRun]);

  const loadAvailableRuns = async () => {
    try {
      const response = await fetch('/api/batch-runs');
      if (response.ok) {
        const runs = await response.json();
        setAvailableRuns(runs);
        if (runs.length > 0) {
          setSelectedRun(runs[0]);
        }
      }
    } catch (error) {
      console.error('Failed to load available runs:', error);
    }
  };

  const loadResults = async (runId: string) => {
    try {
      const response = await fetch(`/api/batch-results/${runId}`);
      if (response.ok) {
        const data = await response.json();
        setData(data);
      }
    } catch (error) {
      console.error('Failed to load batch results:', error);
    }
  };

  if (!data) {
    return (
      <div style={{ padding: '20px', textAlign: 'center', color: '#666' }}>
        No batch results available. Run a batch optimization to see results here.
      </div>
    );
  }

  const handleSort = (field: SortField) => {
    if (sortField === field) {
      setSortDirection(sortDirection === 'asc' ? 'desc' : 'asc');
    } else {
      setSortField(field);
      setSortDirection('asc');
    }
  };

  const getSortedAndFilteredResults = () => {
    let results = [...data.results];

    if (filterText) {
      const lower = filterText.toLowerCase();
      results = results.filter(r =>
        r.modelName.toLowerCase().includes(lower) ||
        r.convergenceReason.toLowerCase().includes(lower)
      );
    }

    results.sort((a, b) => {
      let aVal: number | string;
      let bVal: number | string;

      switch (sortField) {
        case 'modelName':
          aVal = a.modelName;
          bVal = b.modelName;
          break;
        case 'iterations':
          aVal = a.iterations;
          bVal = b.iterations;
          break;
        case 'timeElapsed':
          aVal = a.timeElapsed;
          bVal = b.timeElapsed;
          break;
        case 'developableAfter':
          aVal = a.developableAfter;
          bVal = b.developableAfter;
          break;
        case 'developableChange':
          aVal = a.developableAfter - a.developableBefore;
          bVal = b.developableAfter - b.developableBefore;
          break;
        default:
          return 0;
      }

      if (typeof aVal === 'string') {
        return sortDirection === 'asc'
          ? aVal.localeCompare(bVal as string)
          : (bVal as string).localeCompare(aVal);
      }

      return sortDirection === 'asc'
        ? (aVal as number) - (bVal as number)
        : (bVal as number) - (aVal as number);
    });

    return results;
  };

  const results = getSortedAndFilteredResults();

  const SortIcon = ({ field }: { field: SortField }) => {
    if (sortField !== field) return <span style={{ opacity: 0.3 }}>↕</span>;
    return <span>{sortDirection === 'asc' ? '↑' : '↓'}</span>;
  };

  return (
    <div style={{ padding: '20px', background: '#1e1e1e', minHeight: '100vh' }}>
      <div style={{ marginBottom: '20px' }}>
        <h1 style={{ margin: '0 0 10px', color: '#e0e0e0' }}>Batch Results</h1>

        {availableRuns.length > 0 && (
          <div style={{ marginBottom: '20px' }}>
            <label style={{ color: '#999', fontSize: '14px', marginRight: '10px' }}>
              Select Run:
            </label>
            <select
              value={selectedRun}
              onChange={(e) => setSelectedRun(e.target.value)}
              style={{
                padding: '8px 12px',
                fontSize: '14px',
                background: '#2a2a2a',
                color: '#e0e0e0',
                border: '1px solid #444',
                borderRadius: '4px',
              }}
            >
              {availableRuns.map(run => (
                <option key={run} value={run}>
                  {new Date(run.replace(/-/g, ':')).toLocaleString()}
                </option>
              ))}
            </select>
          </div>
        )}

        <p style={{ margin: '0 0 20px', color: '#999', fontSize: '14px' }}>
          Settings: {data.settings.subdivisions} subdivisions, {data.settings.maxIterations} max iterations, {data.settings.optimizer} optimizer
        </p>

        <input
          type="text"
          placeholder="Filter by model name or convergence reason..."
          value={filterText}
          onChange={(e) => setFilterText(e.target.value)}
          style={{
            width: '100%',
            maxWidth: '500px',
            padding: '10px',
            fontSize: '14px',
            border: '1px solid #ddd',
            borderRadius: '4px',
            boxSizing: 'border-box',
          }}
        />
      </div>

      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fill, minmax(400px, 1fr))',
        gap: '20px'
      }}>
        {results.map(r => (
          <div
            key={r.modelName}
            style={{
              background: 'white',
              padding: '20px',
              borderRadius: '8px',
              boxShadow: '0 2px 4px rgba(0,0,0,0.1)'
            }}
          >
            <h2 style={{ marginTop: 0, color: '#2196F3', fontSize: '18px' }}>{r.modelName}</h2>
            <img
              src={r.imageUrl}
              alt={r.modelName}
              style={{
                width: '100%',
                borderRadius: '4px',
                marginBottom: '10px',
                boxShadow: '0 2px 8px rgba(0,0,0,0.15)'
              }}
            />

            <table style={{ width: '100%', fontSize: '13px', borderCollapse: 'collapse' }}>
              <tbody>
                <tr style={{ borderBottom: '1px solid #eee' }}>
                  <td style={{ padding: '4px 0', fontWeight: 'bold', color: '#666' }}>Optimizer:</td>
                  <td style={{ padding: '4px 0', color: '#333' }}>{r.optimizer}</td>
                </tr>
                <tr style={{ borderBottom: '1px solid #eee' }}>
                  <td style={{ padding: '4px 0', fontWeight: 'bold', color: '#666' }}>Compiled:</td>
                  <td style={{ padding: '4px 0', color: '#333' }}>{r.useCompiled ? 'Yes' : 'No'}</td>
                </tr>
                {r.compilationTime && (
                  <tr style={{ borderBottom: '1px solid #eee' }}>
                    <td style={{ padding: '4px 0', fontWeight: 'bold', color: '#666' }}>Compilation Time:</td>
                    <td style={{ padding: '4px 0', color: '#333' }}>{r.compilationTime.toFixed(2)}s</td>
                  </tr>
                )}
                <tr style={{ borderBottom: '1px solid #eee' }}>
                  <td style={{ padding: '4px 0', fontWeight: 'bold', color: '#666' }}>Iterations:</td>
                  <td style={{ padding: '4px 0', color: '#333' }}>{r.iterations} / {r.maxIterations}</td>
                </tr>
                <tr style={{ borderBottom: '1px solid #eee' }}>
                  <td style={{ padding: '4px 0', fontWeight: 'bold', color: '#666' }}>Time:</td>
                  <td style={{ padding: '4px 0', color: '#333' }}>{r.timeElapsed.toFixed(2)}s ({(r.iterations / r.timeElapsed).toFixed(1)} it/s)</td>
                </tr>
                <tr style={{ borderBottom: '1px solid #eee' }}>
                  <td style={{ padding: '4px 0', fontWeight: 'bold', color: '#666' }}>Function Evals:</td>
                  <td style={{ padding: '4px 0', color: '#333' }}>{r.functionEvals || 'N/A'}</td>
                </tr>
                <tr style={{ borderBottom: '1px solid #eee' }}>
                  <td style={{ padding: '4px 0', fontWeight: 'bold', color: '#666' }}>Before:</td>
                  <td style={{ padding: '4px 0', color: '#333' }}>{(r.developableBefore * 100).toFixed(2)}% ({r.regionsBefore} regions)</td>
                </tr>
                <tr style={{ borderBottom: '1px solid #eee' }}>
                  <td style={{ padding: '4px 0', fontWeight: 'bold', color: '#666' }}>After:</td>
                  <td style={{ padding: '4px 0', color: '#333' }}>{(r.developableAfter * 100).toFixed(2)}% ({r.regionsAfter} regions)</td>
                </tr>
                <tr style={{ borderBottom: '1px solid #eee' }}>
                  <td style={{ padding: '4px 0', fontWeight: 'bold', color: '#666' }}>Change:</td>
                  <td style={{
                    padding: '4px 0',
                    color: r.developableAfter > r.developableBefore ? '#4CAF50' : '#f44336',
                    fontWeight: 'bold'
                  }}>
                    {((r.developableAfter - r.developableBefore) * 100).toFixed(2)}%
                  </td>
                </tr>
                <tr>
                  <td style={{ padding: '4px 0', fontWeight: 'bold', color: '#666' }}>Convergence:</td>
                  <td style={{ padding: '4px 0', color: '#333', fontSize: '11px' }}>{r.convergenceReason}</td>
                </tr>
              </tbody>
            </table>
          </div>
        ))}
      </div>

      {results.length === 0 && filterText && (
        <div style={{ textAlign: 'center', padding: '40px', color: '#999' }}>
          No results match your filter
        </div>
      )}

      <div style={{ marginTop: '30px', padding: '20px', background: 'white', borderRadius: '8px' }}>
        <h3 style={{ marginTop: 0 }}>Quick Sort</h3>
        <div style={{ display: 'flex', gap: '10px', flexWrap: 'wrap' }}>
          <button onClick={() => handleSort('modelName')} style={buttonStyle}>
            Name <SortIcon field="modelName" />
          </button>
          <button onClick={() => handleSort('iterations')} style={buttonStyle}>
            Iterations <SortIcon field="iterations" />
          </button>
          <button onClick={() => handleSort('timeElapsed')} style={buttonStyle}>
            Time <SortIcon field="timeElapsed" />
          </button>
          <button onClick={() => handleSort('developableAfter')} style={buttonStyle}>
            Final % <SortIcon field="developableAfter" />
          </button>
          <button onClick={() => handleSort('developableChange')} style={buttonStyle}>
            Change <SortIcon field="developableChange" />
          </button>
        </div>
      </div>
    </div>
  );
}

const buttonStyle: React.CSSProperties = {
  padding: '8px 16px',
  fontSize: '13px',
  background: '#2196F3',
  color: 'white',
  border: 'none',
  borderRadius: '4px',
  cursor: 'pointer',
  display: 'flex',
  alignItems: 'center',
  gap: '6px',
};
