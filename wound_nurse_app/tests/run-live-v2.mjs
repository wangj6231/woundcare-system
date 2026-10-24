import { build } from 'esbuild'
import { fileURLToPath } from 'node:url'
import path from 'node:path'
import { spawnSync } from 'node:child_process'
const app = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..')
const phase = path.resolve(app, '../deployment/APP-LIVE-V2-01')
const bundle = path.join(phase, 'frontend-test-bundle.cjs')
await build({ entryPoints: [path.join(app, 'tests/live-v2.test.tsx')], outfile: bundle, bundle: true, platform: 'node', format: 'cjs', jsx: 'automatic', logLevel: 'warning' })
const result = spawnSync(process.execPath, ['--test', bundle], { stdio: 'inherit' })
process.exitCode = result.status ?? 1
