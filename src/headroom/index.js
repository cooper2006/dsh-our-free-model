/**
 * Headroom lifecycle: the plugin's managed copy of the headroom proxy.
 *
 * Headroom is an external Python service (pip package `headroom-ai[proxy]`,
 * CLI `headroom proxy`). This module owns that process the way `src/egress.js`
 * owns mihomo: a module-level handle that survives a hot reload, a loopback
 * port with a per-start key, and a start that can explain why it failed so the
 * settings page can show an actionable line instead of a dead switch.
 *
 * Why a managed subprocess rather than an in-process call: the compression
 * pipeline (SmartCrusher, CodeCompressor, the Kompress-v2-base ONNX model,
 * Magika sniffing) is Python-only; the only supported in-process surface is the
 * proxy's HTTP API. The plugin's job stays the same seam as egress: settings in,
 * a loopback URL out, `headroomFetch` unchanged between "running" and "absent".
 *
 * Failure model: every failure mode below is a *string* in `state`, never an
 * exception — a request must never fail because the thing that was supposed to
 * make it cheaper broke. The pipeline reads `state` and skips itself when the
 * proxy is not ready.
 *
 * @module src/headroom/index.js
 */

import fs from 'node:fs'
import net from 'node:net'
import path from 'node:path'
import os from 'node:os'
import { spawn, spawnSync } from 'node:child_process'
import { randomBytes } from 'node:crypto'

/** The loopback address the proxy listens on. Never a routable one. */
const HOST = '127.0.0.1'
/** A fresh proxy gets this long to answer a GET /v1/retrieve/stats on its port.
 *  First boot downloads nothing (models are cached), but ONNX import of the
 *  kompress model is slow, so 45s is generous without hanging a request. */
const READY_TIMEOUT_MS = 45_000
/** One `/v1/compress` call gets this long before the pipeline gives up and
 *  falls back to RTK alone. A wedged proxy must not stall the turn. */
const COMPRESS_TIMEOUT_MS = 30_000
/** The port the managed proxy always binds; a different generator owns it when
 *  the settings page shows `already-running`, in which case the plugin joins
 *  that process instead of fighting over the port. */
const DEFAULT_PORT = 8787

/** The live handle, owned by whichever generation started it. Module-level on
 *  purpose: `adapter.js` imports the client with no handle on the index fiber,
 *  and a hot reload swaps this module while the old disposer tears the old one
 *  down — the same trick egress.js uses for the relay. */
let active = null

/**
 * What the settings page and the pipeline see, always a plain object:
 * `{ state, detail, port, pid, running, reason }`.
 *
 * `state` is one of: `off` (switch closed), `starting`, `ready`,
 * `already-running`, `unavailable` (binary/python missing), `failed`
 * (spawn or port-open failed, see `reason`), `dead` (the process exited after
 * a ready start, see `reason`). `detail` is the human line; `running` is
 * `state === 'ready' || state === 'already-running'` and is the flag the
 * pipeline keys on.
 */
export function headroomStatus() {
  if (active === null) return { state: 'off', detail: '', running: false }
  return active.status()
}

/**
 * Start the managed proxy for the current settings; idempotence is the caller's
 * (`syncHeadroom`) — it closes the old handle first. Mirrors
 * `startEgressRelay`'s contract: this function only builds the new one or fails
 * with a message the settings page can show.
 *
 * @param {{settings: () => object, dataDir: string, log: Function, pythonPath?: string, onDead?: Function, port?: number}} options
 * @returns {Promise<{close: Function, status: Function, port: number}>}
 */
export async function startHeadroom({ settings, dataDir, log = () => {}, pythonPath, onDead, port: requestedPort }) {
  const cfg = () => settings()?.tokenSaver?.headroom ?? {}
  if (cfg().enabled === false) throw new Error('headroom is disabled in the settings')
  const dir = path.join(dataDir, 'headroom')
  fs.mkdirSync(dir, { recursive: true })
  const logFile = path.join(dir, 'headroom.log')

  // Find the binary: an explicit pythonPath setting first, then `headroom` on
  // PATH, then `python3 -m headroom.proxy` as a last resort. No downloads —
  // a missing binary is an error the settings page explains.
  const binary = findHeadroomBinary(pythonPath ?? cfg().pythonPath)

  // `requestedPort` exists for the tests: the suite stands a fake proxy on the
  // port a managed start would bind, and a developer machine that already runs
  // a real headroom on 8787 would otherwise turn every managed-start check into
  // a join. Production never passes it — DEFAULT_PORT is the product contract.
  const port = Number.isInteger(requestedPort) && requestedPort > 0 ? requestedPort : DEFAULT_PORT
  // Join a live process on the port instead of failing: a headroom the user
  // started by hand (or a previous generation that outlived the disposer)
  // answers the health check and is usable as-is.
  if (await portAnswers(port)) {
    const handle = joinHandle(port, log, logFile)
    active = handle
    log(`headroom: joined the process already listening on ${HOST}:${port}`)
    return handle
  }

  // `--no-optimize` makes the proxy a pure compression endpoint (the
  // `/v1/compress` API) with the LLM-relay half of the product off — exactly
  // what the pipeline wants: it calls the compress API itself and ships the
  // result to its own provider, so the proxy must not also forward anywhere.
  let child = spawn(binary.command, [...binary.args, '--port', String(port), '--no-optimize', '--no-cache'], {
    cwd: dir,
    windowsHide: true,
    stdio: ['ignore', 'ignore', logFileFd(logFile)],
    env: { ...process.env, HEADROOM_BEACON: 'off', DO_NOT_TRACK: '1', HEADROOM_OFFLINE: '1' },
  })
  // The beacon and the HF hub check are the two things that would phone home
  // from a local tool; both are opted out for a plugin running on the user's
  // machine with somebody else's traffic.
  let spawnError = null
  let exited = null
  child.on('error', error => { spawnError = error })
  child.on('exit', (code, signal) => { if (exited === null && spawnError === null) exited = `headroom exited (${signal ?? code})` })
  try {
    await waitForPort(HOST, port, READY_TIMEOUT_MS, () => {
      if (spawnError !== null) throw new Error(`headroom could not start (${spawnError.message}) — see ${logFile}`)
      if (exited !== null) throw new Error(`${exited} — see ${logFile}`)
    })
  } catch (error) {
    await killChild(child)
    throw error
  }
  // A ready proxy is a background service: it must not vote on keeping this
  // process alive. Kept ref'd through the start so the exit/spawn listeners
  // above can answer the port wait honestly, and unref'd once it answers —
  // otherwise a host that booted one could never let its loop drain, and a
  // crashed host would refuse to die while its compressor breathes. `close`
  // and `stopHeadroom` still reap it; unref only stops it from haunting.
  child.unref()

  const handle = {
    child,
    port,
    key: randomBytes(12).toString('hex'),
    log,
    logFile,
    dead: '',
    status() {
      if (this.dead !== '') return { state: 'dead', detail: this.dead, running: false, port, pid: this.child?.pid ?? null }
      if (this.child?.exitCode !== null || this.child?.signalCode !== null) return { state: 'dead', detail: 'the managed headroom process exited', running: false, port, pid: this.child?.pid ?? null }
      return { state: 'ready', detail: `listening on ${HOST}:${port}`, running: true, port, pid: this.child?.pid ?? null }
    },
    close: async () => {
      // `handle`, not `this`: this is an arrow property, so `this` here is the
      // module scope, not the handle — the one path where a managed proxy
      // must die on stop. `status()` above is a method, so its `this` is real.
      if (active === handle) active = null
      await killChild(handle.child)
    },
  }
  child.on('exit', () => {
    handle.dead = 'the managed headroom process exited'
    handle.deadSince = Date.now()
    log('headroom: the managed process exited; a restart has been scheduled')
    onDead?.()
  })
  active = handle
  handle.readyAt = Date.now()
  log(`headroom: managed proxy on ${HOST}:${port} (pid ${child.pid}, log ${logFile})`)
  return handle
}

/**
 * Stop the handle this generation owns. A handle that joined a user-started
 * process closes nothing — the user owns that process, not the plugin.
 */
export async function stopHeadroom() {
  const handle = active
  active = null
  if (handle === null || handle.close === undefined) return
  await handle.close()
}

/**
 * What the pipeline gets from one `/v1/compress` round-trip, or `null` when the
 * proxy is not usable — the caller then falls back to RTK alone. Fail-open by
 * construction: every error path returns a null with a reason, never throws.
 *
 * @param {Array<object>} openaiMessages - messages in the proxy's lingua franca
 * @param {{model?: string, ccr?: boolean}} options
 * @returns {Promise<{ok: boolean, reason?: string, result?: object}>}
 */
export async function headroomCompress(openaiMessages, options = {}) {
  const status = headroomStatus()
  if (!status.running) return { ok: false, reason: status.state === 'off' ? 'disabled' : `not running (${status.state})` }
  const url = `http://${HOST}:${status.port}/v1/compress`
  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), COMPRESS_TIMEOUT_MS)
  try {
    const body = { messages: openaiMessages, model: options.model ?? 'gpt-4o' }
    // The proxy's compress contract is `config.mode`: unset means a
    // marker-free body you can hand straight to a provider, "ccr" means the
    // cached-originals + `Retrieve more: hash=…` marker mode our retrieve
    // tool answers. Anything else is a 400. Off therefore sends no mode at
    // all — sending `{ccr:false}` would be ignored and a `{mode:"lossy"}`
    // guess would fail the turn.
    if (options.ccr === true) body.config = { mode: 'ccr' }
    const response = await fetch(url, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify(body),
      signal: controller.signal,
    })
    if (!response.ok) return { ok: false, reason: `the proxy answered ${response.status}` }
    const result = await response.json()
    return { ok: true, result }
  } catch (error) {
    return { ok: false, reason: error?.name === 'AbortError' ? `timed out after ${COMPRESS_TIMEOUT_MS}ms` : (error?.message ?? String(error)) }
  } finally {
    clearTimeout(timer)
  }
}

/**
 * The CCR half: turn the model's `headroom_retrieve` tool call into a tool
 * result. This is what makes the compression reversible — a compressed block
 * carries a hash marker, the model asks for the original, and the proxy's CCR
 * store answers. Without this, the compression is one-way and the model works
 * blind on anything it asked to keep.
 *
 * @param {string} hash - the marker the model read out of a compressed block
 * @returns {Promise<{ok: boolean, content?: string, reason?: string}>}
 */
export async function headroomRetrieve(hash, options = {}) {
  const status = headroomStatus()
  if (!status.running) return { ok: false, reason: status.state === 'off' ? 'disabled' : 'the headroom proxy is not running, so nothing was cached to retrieve' }
  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), 10_000)
  try {
    const response = await fetch(`http://${HOST}:${status.port}/v1/retrieve/${encodeURIComponent(hash)}`, { signal: controller.signal })
    if (!response.ok) return { ok: false, reason: `the CCR store answered ${response.status} for this hash` }
    const result = await response.json()
    return { ok: true, content: result.original_content ?? '' }
  } catch (error) {
    return { ok: false, reason: error?.name === 'AbortError' ? 'the CCR lookup timed out' : (error?.message ?? String(error)) }
  } finally {
    clearTimeout(timer)
  }
}

/**
 * The tool definition the pipeline injects when CCR is on: it is declared to
 * the provider with every compressed request so the model can call it, and the
 * harness's tool loop resolves it through this function. The description is
 * written for the model, not the user — it names the marker shape.
 */
export function headroomRetrieveTool() {
  return {
    type: 'function',
    function: {
      name: 'headroom_retrieve',
      description: 'Retrieve the original content of a block that was compressed with CCR. Compressed blocks end with a marker like `Retrieve more: hash=3f8a12b0c9d4` (12-24 hex digits) — call this tool with that hash whenever you need detail the compacted form dropped. The original comes back verbatim.',
      parameters: {
        type: 'object',
        properties: { hash: { type: 'string', description: 'The hash after `hash=` in the compressed block\'s marker, e.g. 3f8a12b0c9d4.' } },
        required: ['hash'],
      },
    },
  }
}

/**
 * One line for the stats panel: what the CCR store itself reports. The proxy
 * keeps the counts, the panel shows them; a zero is a fact, not a hole.
 */
export async function headroomStats() {
  const status = headroomStatus()
  if (!status.running) return { ok: false, reason: 'not running' }
  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), 5_000)
  try {
    const response = await fetch(`http://${HOST}:${status.port}/v1/retrieve/stats`, { signal: controller.signal })
    if (!response.ok) return { ok: false, reason: `the stats endpoint answered ${response.status}` }
    return { ok: true, result: await response.json() }
  } catch (error) {
    return { ok: false, reason: error?.message ?? String(error) }
  } finally {
    clearTimeout(timer)
  }
}

// ── internals ─────────────────────────────────────────────────────────────────

function joinHandle(port, log, logFile) {
  // A joined process is not owned: `close` releases our view of it but never
  // kills somebody else's server.
  const handle = {
    child: null,
    port,
    key: '',
    log,
    logFile,
    joined: true,
    status() {
      // `headroomStatus()` is sync, so the joined handle reports what it
      // proved at start time: the port answered. A death surfaces on the next
      // health read (`probeHeadroom`) rather than being re-probed here.
      return { state: 'already-running', detail: `a headroom proxy already answers on ${HOST}:${port} — joined, not spawned`, running: true, port, pid: null }
    },
    close: async () => {
      // Release our view of the process without touching it — it belongs to
      // whoever started it. Leaving `active` pointing at a closed handle would
      // keep the pipeline compressing against a proxy this generation has
      // already let go of.
      if (active === handle) active = null
    },
  }
  return handle
}

/** TCP check: does anything answer on the port right now? */
async function portAnswers(port) {
  return new Promise(resolve => {
    const socket = net.connect({ host: HOST, port })
    const done = ok => { socket.removeAllListeners(); socket.destroy(); resolve(ok) }
    socket.once('connect', () => done(true))
    socket.once('error', () => done(false))
    socket.setTimeout(500, () => done(false))
  })
}

/**
 * Locate the headroom entry: the explicit setting first, then `headroom` on
 * PATH, then `python3 -m headroom.proxy` — the same "no downloads" rule as the
 * mihomo lookup. A python candidate only counts when the `headroom.proxy`
 * module actually imports: spawning one that dies on `ModuleNotFoundError`
 * teaches nothing this five-cent check cannot, and a managed start that waits
 * on a doomed process pays for it at every boot.
 */
export function findHeadroomBinary(explicit) {
  const given = String(explicit ?? '').trim()
  if (given !== '') {
    if (statOrNull(given) === null) throw new Error(`the headroom binary "${given}" does not exist`)
    return { command: given, args: ['proxy'] }
  }
  for (const dir of searchRoots()) {
    const candidate = path.join(dir, process.platform === 'win32' ? 'headroom.exe' : 'headroom')
    if (isExecutable(candidate)) return { command: candidate, args: ['proxy'] }
  }
  for (const dir of searchRoots()) {
    const candidate = path.join(dir, 'python3')
    if (isExecutable(candidate) && headroomModuleImports(candidate)) return { command: candidate, args: ['-m', 'headroom.proxy'] }
  }
  throw new Error('no headroom binary found - run `pip install headroom-ai[proxy]` (or set the binary path in the headroom settings)')
}

/** One probe, five-second cap: does this python actually import the proxy? */
function headroomModuleImports(python) {
  try {
    const probe = spawnSync(python, ['-c', 'import headroom.proxy'], { timeout: 5_000, stdio: 'ignore', windowsHide: true })
    return probe.status === 0
  } catch {
    return false
  }
}

function searchRoots() {
  const roots = []
  for (const entry of (process.env.PATH ?? '').split(path.delimiter)) {
    if (entry.trim() !== '') roots.push(entry)
  }
  const home = os.homedir()
  if (typeof home === 'string' && home !== '') roots.push(path.join(home, '.local', 'bin'), path.join(home, '.venv', 'bin'))
  roots.push('/usr/local/bin', '/usr/bin', '/opt/homebrew/bin')
  return roots
}

function isExecutable(candidate) {
  if (statOrNull(candidate) === null) return false
  try { if (process.platform !== 'win32') fs.accessSync(candidate, fs.constants.X_OK); return true } catch { return false }
}

function statOrNull(target) {
  try { return fs.statSync(target) } catch { return null }
}

/** The child's stderr, one line at a time, into the log file. */
function logFileFd(logFile) {
  try {
    const fd = fs.openSync(logFile, fs.constants.O_WRONLY | fs.constants.O_CREAT | fs.constants.O_APPEND)
    return fd
  } catch {
    return 'ignore'
  }
}

/** Poll a TCP port until it answers or the budget runs out. */
async function waitForPort(host, port, timeoutMs, check) {
  const deadline = Date.now() + timeoutMs
  for (;;) {
    check?.()
    const open = await new Promise(resolve => {
      const socket = net.connect({ host, port })
      const settle = ok => { socket.removeAllListeners(); socket.destroy(); resolve(ok) }
      socket.once('connect', () => settle(true))
      socket.once('error', () => settle(false))
    })
    if (open) return
    if (Date.now() > deadline) throw new Error(`the port ${host}:${port} never opened within ${timeoutMs}ms`)
    await new Promise(resolve => { const t = setTimeout(resolve, 500); t.unref?.() })
  }
}

/** Ask nicely, then insist — the same two-stage kill egress.js uses. */
async function killChild(proc) {
  if (proc === null || proc.exitCode !== null || proc.signalCode !== null) return
  proc.removeAllListeners('exit')
  try { proc.kill() } catch { /* already gone */ }
  await new Promise(resolve => {
    const timer = setTimeout(resolve, 3_000)
    proc.once('exit', () => { clearTimeout(timer); resolve() })
  })
  if (proc.exitCode === null && proc.signalCode === null) {
    try { proc.kill('SIGKILL') } catch { /* already gone */ }
  }
}
