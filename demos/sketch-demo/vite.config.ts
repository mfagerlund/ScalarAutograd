import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { resolve } from 'path'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      'scalar-autograd': resolve(__dirname, '../../dist/index.js'),
    },
  },
  optimizeDeps: {
    exclude: ['scalar-autograd'],
  },
})
