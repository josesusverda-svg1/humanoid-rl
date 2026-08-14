import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// The built bundle is served by FastAPI from dashboard/frontend/dist, so `base` stays '/'.
// During `npm run dev` the proxy forwards /api to the backend on port 8000, which keeps the
// frontend code free of environment-dependent URLs.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: 'dist',
    sourcemap: true,
  },
})
