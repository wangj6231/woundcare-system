import { build } from 'esbuild'
import { fileURLToPath } from 'node:url'
import path from 'node:path'
import { spawnSync } from 'node:child_process'
import { mkdirSync } from 'node:fs'
const app=path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..')
const phase=path.resolve(app,'../deployment/APP-UX-SAFETY-01')
mkdirSync(phase,{recursive:true})
const bundle=path.join(phase,'ux-safety-tests.cjs')
await build({entryPoints:[path.join(app,'tests/ux-safety.test.tsx')],outfile:bundle,bundle:true,platform:'node',format:'cjs',jsx:'automatic',logLevel:'warning'})
const result=spawnSync(process.execPath,['--test',bundle],{stdio:'inherit'})
process.exitCode=result.status??1
