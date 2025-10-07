import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import fs from 'fs';
import path from 'path';

const BATCH_RESULTS_DIR = 'c:\\slask\\dev-sphere-runs';

export default defineConfig({
  plugins: [
    react(),
    {
      name: 'debug-logger',
      configureServer(server) {
        server.middlewares.use((req, res, next) => {
          if (req.url?.startsWith('/DEBUG/')) {
            const msg = decodeURIComponent(req.url.substring(7));
            console.log(`[CLIENT] ${msg}`);
            res.writeHead(200);
            res.end('OK');
          } else {
            next();
          }
        });
      },
    },
    {
      name: 'batch-results-api',
      configureServer(server) {
        server.middlewares.use(async (req, res, next) => {
          if (req.url === '/api/batch-runs') {
            try {
              if (!fs.existsSync(BATCH_RESULTS_DIR)) {
                res.writeHead(200, { 'Content-Type': 'application/json' });
                res.end(JSON.stringify([]));
                return;
              }

              const dirs = fs.readdirSync(BATCH_RESULTS_DIR)
                .filter(d => fs.statSync(path.join(BATCH_RESULTS_DIR, d)).isDirectory())
                .sort()
                .reverse();

              res.writeHead(200, { 'Content-Type': 'application/json' });
              res.end(JSON.stringify(dirs));
            } catch (error) {
              res.writeHead(500, { 'Content-Type': 'application/json' });
              res.end(JSON.stringify({ error: String(error) }));
            }
          } else if (req.url?.startsWith('/api/batch-results/')) {
            const runId = req.url.substring('/api/batch-results/'.length);
            try {
              const jsonPath = path.join(BATCH_RESULTS_DIR, runId, 'results.json');

              if (!fs.existsSync(jsonPath)) {
                res.writeHead(404, { 'Content-Type': 'application/json' });
                res.end(JSON.stringify({ error: 'Results file not found' }));
                return;
              }

              const data = JSON.parse(fs.readFileSync(jsonPath, 'utf-8'));

              data.results = data.results.map((r: any) => ({
                ...r,
                imageUrl: `/api/batch-image/${runId}/${r.imageName}`
              }));

              res.writeHead(200, { 'Content-Type': 'application/json' });
              res.end(JSON.stringify(data));
            } catch (error) {
              res.writeHead(500, { 'Content-Type': 'application/json' });
              res.end(JSON.stringify({ error: String(error) }));
            }
          } else if (req.url === '/api/latest-batch-results') {
            try {
              if (!fs.existsSync(BATCH_RESULTS_DIR)) {
                res.writeHead(404, { 'Content-Type': 'application/json' });
                res.end(JSON.stringify({ error: 'No results directory' }));
                return;
              }

              const dirs = fs.readdirSync(BATCH_RESULTS_DIR)
                .filter(d => fs.statSync(path.join(BATCH_RESULTS_DIR, d)).isDirectory())
                .sort()
                .reverse();

              if (dirs.length === 0) {
                res.writeHead(404, { 'Content-Type': 'application/json' });
                res.end(JSON.stringify({ error: 'No results found' }));
                return;
              }

              const latestDir = dirs[0];
              const jsonPath = path.join(BATCH_RESULTS_DIR, latestDir, 'results.json');

              if (!fs.existsSync(jsonPath)) {
                res.writeHead(404, { 'Content-Type': 'application/json' });
                res.end(JSON.stringify({ error: 'Results file not found' }));
                return;
              }

              const data = JSON.parse(fs.readFileSync(jsonPath, 'utf-8'));

              data.results = data.results.map((r: any) => ({
                ...r,
                imageUrl: `/api/batch-image/${latestDir}/${r.imageName}`
              }));

              res.writeHead(200, { 'Content-Type': 'application/json' });
              res.end(JSON.stringify(data));
            } catch (error) {
              res.writeHead(500, { 'Content-Type': 'application/json' });
              res.end(JSON.stringify({ error: String(error) }));
            }
          } else if (req.url?.startsWith('/api/batch-image/')) {
            const urlPath = req.url.substring('/api/batch-image/'.length);
            const imagePath = path.join(BATCH_RESULTS_DIR, urlPath);

            if (fs.existsSync(imagePath)) {
              const data = fs.readFileSync(imagePath);
              res.writeHead(200, { 'Content-Type': 'image/png' });
              res.end(data);
            } else {
              res.writeHead(404);
              res.end('Image not found');
            }
          } else if (req.url === '/api/save-batch-results' && req.method === 'POST') {
            let body = '';
            req.on('data', chunk => { body += chunk; });
            req.on('end', () => {
              try {
                const data = JSON.parse(body);
                const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
                const runDir = path.join(BATCH_RESULTS_DIR, timestamp);

                console.log('[BATCH] Saving results to:', runDir);

                if (!fs.existsSync(BATCH_RESULTS_DIR)) {
                  console.log('[BATCH] Creating base directory:', BATCH_RESULTS_DIR);
                  fs.mkdirSync(BATCH_RESULTS_DIR, { recursive: true });
                }
                fs.mkdirSync(runDir, { recursive: true });
                console.log('[BATCH] Created run directory');

                const resultsWithFiles = data.results.map((r: any) => {
                  const imageBuffer = Buffer.from(r.imageData.split(',')[1], 'base64');
                  const imagePath = path.join(runDir, r.imageName);
                  fs.writeFileSync(imagePath, imageBuffer);

                  const { imageData, ...rest } = r;
                  return rest;
                });

                const saveData = {
                  ...data,
                  results: resultsWithFiles
                };

                fs.writeFileSync(
                  path.join(runDir, 'results.json'),
                  JSON.stringify(saveData, null, 2)
                );

                res.writeHead(200, { 'Content-Type': 'application/json' });
                res.end(JSON.stringify({ success: true, path: runDir }));
              } catch (error) {
                res.writeHead(500, { 'Content-Type': 'application/json' });
                res.end(JSON.stringify({ error: String(error) }));
              }
            });
          } else {
            next();
          }
        });
      },
    },
  ],
});
