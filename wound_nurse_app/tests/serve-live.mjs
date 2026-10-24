import { createServer } from 'vite'
import { existsSync, readFileSync, writeFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import path from 'node:path'
const app = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..')
const phase = path.resolve(process.argv[2])
const config = JSON.parse(readFileSync(path.join(phase, 'runtime_paths.json'), 'utf8'))
const server = await createServer({ root: app, configFile: path.join(app, 'vite.config.ts'), clearScreen: false,
  server: { host: '127.0.0.1', port: 5174, strictPort: true } })
await server.listen()
writeFileSync(path.join(phase, 'frontend_ready.json'), JSON.stringify({ pid: process.pid, host: '127.0.0.1', port: 5174, backend: process.env.WOUNDCARE_API_TARGET }))
const timer = setInterval(async () => {
  if (existsSync(path.join(config.private_root, 'stop_frontend.request'))) {
    clearInterval(timer)
    await server.close()
    process.exit(0)
  }
}, 200)
