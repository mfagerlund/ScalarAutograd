import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

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
  ],
});
