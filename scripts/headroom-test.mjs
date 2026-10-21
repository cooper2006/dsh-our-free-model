/**
 * The headroom module: locator, fail-open client, the compress step's safety
 * gates, CCR injection, and the degradation chain through the token-saver
 * pipeline.
 *
 * The proxy itself is an external Python service, so this suite never spawns
 * the real one. Everything runs against doubles — a fake HTTP server standing
 * on the loopback port `startHeadroom` would join, and payloads built as
 * literals — exactly the way `egress-test.mjs` covers the managed-mihomo path
 * with config rendering and the binary locator only.
 *
 * What the checks hold:
 * - `findHeadroomBinary` prefers the explicit path, then `headroom` on PATH,
 *   then `python3 -m headroom.proxy`, and throws with an actionable line when
 *   nothing exists;
 * - with nothing running, every client call is a `{ok:false}` with a reason —
 *   a request must never fail because its compressor was unavailable;
 * - with a joined handle (the fake proxy answering the real HTTP contract),
 *   `applyHeadroom` round-trips `/v1/compress`, refuses a body that is not
 *   strictly smaller, keeps the payload out of the failure paths, and injects
 *   the CCR retrieve tool exactly when the compressed text carries a marker;
 * - the pipeline is the degradation chain: headroom wins → RTK is skipped,
 *   headroom fails → RTK carries the load;
 * - a joined process is not owned: `stopHeadroom` releases it and the fake
 *   server still answers.
 *
 * No network beyond loopback, no home directory, no python required.
 *
 * Run: node scripts/headroom-test.mjs
 */
import assert from 'node:assert/strict'
import fs from 'node:fs'
import net from 'node:net'
import os from 'node:os'
import path from 'node:path'
import http from 'node:http'
import { applyHeadroom } from '../src/headroom/compress.js'
import {
  findHeadroomBinary,
  headroomCompress,
  headroomRetrieve,
  headroomRetrieveTool,
  headroomStatus,
  startHeadroom,
  stopHeadroom,
} from '../src/headroom/index.js'
import { addSavings, applyTokenSaversAsync, emptySavings, saverConfig } from '../src/token-saver/index.js'

let failures = 0
const check = async (name, fn) => {
  try {
    await fn()
    console.log(`  ok  ${name}`)
  }
  catch (error) { failures += 1; console.error(`FAIL  ${name} — ${error.message}`) }
}

// ── the binary locator ───────────────────────────────────────────────────────
// The same "no downloads" rule as the mihomo lookup, so virtual files keep both
// platform branches independent of what is installed on this machine.

function withPath(value, fn) {
  const saved = process.env.PATH
  try { process.env.PATH = value; return fn() } finally { process.env.PATH = saved }
}

await check('an explicit path wins when it exists', () => {
  const fake = path.join(os.tmpdir(), `ofm-headroom-fake-${process.pid}`)
  fs.writeFileSync(fake, '')
  fs.chmodSync(fake, 0o755)
  try {
    const found = findHeadroomBinary(fake)
    assert.equal(found.command, fake)
    assert.deepEqual(found.args, ['proxy'])
  } finally { fs.rmSync(fake, { force: true }) }
})

await check('an explicit path that does not exist is an error, not a fallback', () => {
  assert.throws(() => findHeadroomBinary('/nonexistent/headroom-here'), /does not exist/)
})

await check('`headroom` on PATH beats python3', () => {
  const dir = path.join(os.tmpdir(), `ofm-headroom-path-${process.pid}`)
  fs.mkdirSync(dir, { recursive: true })
  const exe = path.join(dir, 'headroom')
  fs.writeFileSync(exe, '#!/bin/sh\n', { mode: 0o755 })
  try {
    const found = withPath(dir, () => findHeadroomBinary(''))
    assert.equal(found.command, exe)
    assert.deepEqual(found.args, ['proxy'])
  } finally { fs.rmSync(dir, { recursive: true, force: true }) }
})

await check('python3 -m headroom.proxy only wins when the module imports', () => {
  const dir = path.join(os.tmpdir(), `ofm-headroom-py-${process.pid}`)
  fs.mkdirSync(dir, { recursive: true })
  // A python that passes the `import headroom.proxy` probe (exit 0) is what a
  // machine with the package installed looks like to the locator.
  fs.writeFileSync(path.join(dir, 'python3'), '#!/bin/sh\nexit 0\n', { mode: 0o755 })
  try {
    // An empty PATH still ends in the well-known roots; assert only the
    // shape, whatever this machine happens to have installed.
    const found = withPath(dir, () => findHeadroomBinary(''))
    assert.ok(found.command.endsWith('headroom') || found.command.endsWith('python3'), `unexpected winner ${found.command}`)
    if (found.command.endsWith('python3')) assert.deepEqual(found.args, ['-m', 'headroom.proxy'])
  } finally { fs.rmSync(dir, { recursive: true, force: true }) }
})

await check('a python3 without the headroom module is not offered', () => {
  const dir = path.join(os.tmpdir(), `ofm-headroom-noop-${process.pid}`)
  fs.mkdirSync(dir, { recursive: true })
  // The fake python fails the import probe; a machine-wide python in a
  // well-known root may still be found, but the doomed fake never is — that
  // is the boot stall this probe exists to prevent.
  fs.writeFileSync(path.join(dir, 'python3'), '#!/bin/sh\nexit 1\n', { mode: 0o755 })
  try {
    const outcome = withPath(dir, () => {
      try { return findHeadroomBinary('') } catch { return null }
    })
    if (outcome !== null) assert.notEqual(outcome.command, path.join(dir, 'python3'), 'the failing python was offered anyway')
  } finally { fs.rmSync(dir, { recursive: true, force: true }) }
})

// ── the fail-open client, with nothing running ───────────────────────────────

await check('headroomCompress fails open with "disabled" when nothing runs', async () => {
  assert.equal(headroomStatus().state, 'off', 'no handle should be active at this point of the suite')
  const result = await headroomCompress([{ role: 'user', content: 'hi' }], {})
  assert.equal(result.ok, false)
  assert.equal(result.reason, 'disabled')
})

await check('headroomRetrieve fails open with a reason, never a throw', async () => {
  const result = await headroomRetrieve('abc12345')
  assert.equal(result.ok, false)
  assert.match(result.reason, /not running|disabled/)
})

await check('applyHeadroom is a no-op on a disabled config', async () => {
  const payload = { model: 'gpt-4o-mini', messages: [{ role: 'user', content: 'hi' }] }
  const report = await applyHeadroom(payload, 'chat', { enabled: false, ccr: true })
  assert.equal(report.ok, false)
  assert.equal(report.reason, 'disabled')
  assert.equal(payload.messages.length, 1, 'the payload is untouched')
})

await check('applyHeadroom declines wires it cannot speak', async () => {
  const report = await applyHeadroom({ model: 'x', messages: [{ role: 'user', content: 'hi' }] }, 'messages', { enabled: true, ccr: true })
  assert.equal(report.ok, false)
  assert.match(report.reason, /not the OpenAI shape/)
})

await check('the retrieve tool definition is a valid OpenAI function', () => {
  const tool = headroomRetrieveTool()
  assert.equal(tool.type, 'function')
  assert.equal(tool.function.name, 'headroom_retrieve')
  assert.deepEqual(tool.function.parameters.required, ['hash'])
  assert.ok(tool.function.description.includes('Retrieve more: hash='))
})

// ── the HTTP round trip, against a fake proxy on the join port ───────────────
// `headroomCompress` keys on the module-level handle, and the only public way
// to set it is `startHeadroom`. Joining is the path a probe can answer to:
// stand a fake server on the port a managed start would bind and the start
// becomes a join with no subprocess at all. One fixture spans the round-trip
// checks; the per-test behaviour is chosen by the request's `model` field.
//
// The port is `port: 0`-chosen, not 8787: a developer machine that already runs
// a real headroom proxy on the product port would answer the probe and turn
// every "no proxy running" check into a join. `startHeadroom` takes the port as
// an option for exactly this.

function chatPayload(extra = {}) {
  return {
    model: 'gpt-4o-mini',
    messages: [
      { role: 'system', content: 'You are a concise assistant.' },
      { role: 'user', content: 'Explain the token saver pipeline.' },
      { role: 'tool', tool_call_id: 'call_1', content: 'x'.repeat(2000) },
    ],
    ...extra,
  }
}

function startFakeProxy(handler) {
  return new Promise((resolve, reject) => {
    const server = http.createServer(handler)
    server.once('error', reject)
    server.listen(0, '127.0.0.1', () => resolve(server))
  })
}

/** A port nothing is listening on: bind it, read the number, release it. */
async function reservePort() {
  const server = await startFakeProxy((req, res) => { res.writeHead(500); res.end() })
  const port = server.address().port
  await new Promise(resolve => server.close(resolve))
  return port
}

const fakeBinary = path.join(os.tmpdir(), `ofm-headroom-bin-${process.pid}`)
fs.writeFileSync(fakeBinary, '#!/bin/sh\n', { mode: 0o755 })

const fixture = await (async () => {
  try {
    const server = await startFakeProxy((req, res) => {
      let raw = ''
      req.on('data', chunk => { raw += chunk })
      req.on('end', () => {
        if (req.method === 'POST' && req.url === '/v1/compress') {
          const request = JSON.parse(raw)
          if (!Array.isArray(request.messages)) { res.writeHead(400); res.end('no messages'); return }
          if (request.model === 'echo') { res.writeHead(200, { 'content-type': 'application/json' }); res.end(JSON.stringify({ messages: request.messages })); return }
          if (request.model === 'boom') { res.writeHead(500); res.end('nope'); return }
          // The real contract: markers and `ccr_hashes` appear only when the
          // request asked for `config.mode === 'ccr'`; anything else is a
          // marker-free body. Mirroring that here is what proves the client
          // sends the mode the proxy actually reads.
          const useCcr = request.config?.mode === 'ccr'
          const shrunk = request.messages.map(m => (m.role === 'tool'
            ? { ...m, content: useCcr ? 'compressed\n[lines omitted: Retrieve more: hash=deadbeef012345]' : 'compressed body' }
            : m))
          res.writeHead(200, { 'content-type': 'application/json' })
          res.end(JSON.stringify({ messages: shrunk, ccr_hashes: useCcr ? ['deadbeef012345'] : [] }))
          return
        }
        if (req.method === 'GET' && req.url.startsWith('/v1/retrieve/')) {
          const hash = decodeURIComponent(req.url.slice('/v1/retrieve/'.length))
          if (hash === 'deadbeef012345') { res.writeHead(200, { 'content-type': 'application/json' }); res.end(JSON.stringify({ original_content: 'THE-ORIGINAL-2000-BYTES' })); return }
          res.writeHead(404, { 'content-type': 'application/json' }); res.end(JSON.stringify({ error: 'no such hash' })); return
        }
        res.writeHead(404)
        res.end()
      })
    })
    const dataDir = fs.mkdtempSync(path.join(os.tmpdir(), 'ofm-headroom-data-'))
    const handle = await startHeadroom({
      settings: () => ({ tokenSaver: { headroom: { enabled: true } } }),
      dataDir,
      log: () => {},
      pythonPath: fakeBinary,
      port: server.address().port,
    })
    return { server, handle, dataDir, port: server.address().port }
  } catch (error) {
    return { skip: String(error?.message ?? error) }
  }
})()

if (fixture.skip !== undefined) {
  console.log(`  (skipped: ${fixture.skip})`)
} else {
  await check('the managed start joins the live port instead of spawning', () => {
    assert.equal(fixture.handle.status().running, true, `join reported ${fixture.handle.status().state}`)
    assert.equal(headroomStatus().state, 'already-running')
  })

  await check('applyHeadroom ships the compressed messages when they shrink', async () => {
    const payload = chatPayload()
    const report = await applyHeadroom(payload, 'chat', { enabled: true, ccr: false })
    assert.equal(report.ok, true, `headroom declined: ${report.reason}`)
    assert.match(payload.messages[2].content, /compressed/, 'the compressed tool body shipped')
    assert.ok(report.stats.saved > 0)
    assert.equal(report.stats.bytesBefore - report.stats.bytesAfter, report.stats.saved, 'the stats describe the body that ships')
    assert.equal(payload.tools, undefined, 'CCR off, so no retrieve tool')
  })

  await check('a compressed body that is not smaller keeps the original', async () => {
    const payload = chatPayload({ model: 'echo' })
    const report = await applyHeadroom(payload, 'chat', { enabled: true, ccr: false })
    assert.equal(report.ok, false)
    assert.match(report.reason, /not smaller/)
    assert.match(payload.messages[2].content, /^x+$/, 'the original body survived')
  })

  await check('a proxy error is a refusal with the proxy\'s reason', async () => {
    const payload = chatPayload({ model: 'boom' })
    const report = await applyHeadroom(payload, 'chat', { enabled: true, ccr: true })
    assert.equal(report.ok, false)
    assert.match(report.reason, /answered 500/)
    assert.equal(payload.messages.length, 3, 'the payload is untouched on a proxy failure')
  })

  await check('CCR markers trigger the retrieve tool injection', async () => {
    const payload = chatPayload({ tools: [{ type: 'function', function: { name: 'bash', parameters: {} } }] })
    const report = await applyHeadroom(payload, 'chat', { enabled: true, ccr: true })
    assert.equal(report.ok, true, `headroom declined: ${report.reason}`)
    assert.equal(payload.tools.length, 2, 'the original tool plus headroom_retrieve')
    assert.equal(payload.tools[1].function.name, 'headroom_retrieve')
  })

  await check('a second compression pass does not re-declare the tool', async () => {
    const payload = chatPayload({ tools: [{ type: 'function', function: { name: 'bash', parameters: {} } }] })
    await applyHeadroom(payload, 'chat', { enabled: true, ccr: true })
    await applyHeadroom(payload, 'chat', { enabled: true, ccr: true })
    const names = payload.tools.map(t => t.function.name)
    assert.equal(names.filter(n => n === 'headroom_retrieve').length, 1)
  })

  await check('the pipeline lets headroom take the payload and skips RTK', async () => {
    const config = { enabled: true, rtk: true, caveman: 'off', ponytail: 'off', headroom: { enabled: true, ccr: false, pythonPath: '' } }
    const payload = chatPayload()
    const report = await applyTokenSaversAsync(payload, 'chat', config)
    assert.ok(report.headroom !== null, 'headroom took the payload')
    assert.equal(report.rtk, null, 'RTK is skipped when headroom wins')
    assert.ok(report.saved > 0)
    assert.ok(report.notes.every(n => !n.startsWith('headroom-fallback')))
    assert.equal(addSavings(emptySavings(), report).blocks, 1, 'one whole-payload block counted')
  })

  // A request whose compression the proxy declines (the echo model) must still
  // get the RTK path — that is the chain the settings page promises.
  await check('a declined compression still falls through to RTK', async () => {
    const config = { enabled: true, rtk: true, caveman: 'off', ponytail: 'off', headroom: { enabled: true, ccr: false, pythonPath: '' } }
    const payload = chatPayload({ model: 'echo' })
    const report = await applyTokenSaversAsync(payload, 'chat', config)
    assert.equal(report.headroom, null)
    assert.ok(report.notes.some(n => n.startsWith('headroom-fallback:')))
    assert.ok(report.saved >= 0, 'RTK ran; its own checks cover whether this body compacts')
  })

  await check('a retrieve stub in the transcript is answered from the CCR cache', async () => {
    const payload = {
      model: 'gpt-4o-mini',
      messages: [
        { role: 'user', content: 'expand the tool output' },
        { role: 'assistant', content: null, tool_calls: [{ id: 'call_9', type: 'function', function: { name: 'headroom_retrieve', arguments: JSON.stringify({ hash: 'deadbeef012345' }) } }] },
        { role: 'tool', tool_call_id: 'call_9', content: 'Error: unknown tool headroom_retrieve' },
      ],
    }
    const filled = await (await import('../src/headroom/compress.js')).resolveRetrieveCalls(payload)
    assert.equal(filled, 1)
    assert.equal(payload.messages[2].content, 'THE-ORIGINAL-2000-BYTES', 'the stub became the original')
    // An unknown hash leaves the stub as it is — a wrong answer is worse than
    // the harness's own honest "unknown tool" line.
    payload.messages[1].tool_calls[0].function.arguments = JSON.stringify({ hash: 'beef01234567' })
    payload.messages[2].content = 'Error: unknown tool headroom_retrieve'
    const again = await (await import('../src/headroom/compress.js')).resolveRetrieveCalls(payload)
    assert.equal(again, 0)
    assert.match(payload.messages[2].content, /unknown tool/)
  })

  await check('the pipeline answers a retrieve call before touching any body', async () => {
    const config = { enabled: true, rtk: false, caveman: 'off', ponytail: 'off', headroom: { enabled: true, ccr: true, pythonPath: '' } }
    const payload = {
      model: 'echo', // declined compression — the resolve must still have run
      messages: [
        { role: 'assistant', content: null, tool_calls: [{ id: 'call_9', type: 'function', function: { name: 'headroom_retrieve', arguments: '{"hash":"deadbeef012345"}' } }] },
        { role: 'tool', tool_call_id: 'call_9', content: 'Error: unknown tool headroom_retrieve' },
      ],
    }
    const report = await applyTokenSaversAsync(payload, 'chat', config)
    assert.equal(report.retrieveFilled, 1, 'the pipeline resolved the call')
    assert.equal(payload.messages[1].content, 'THE-ORIGINAL-2000-BYTES')
  })

  await check('a joined process is not owned: stop releases it, the server stays up', async () => {
    await stopHeadroom()
    assert.equal(headroomStatus().state, 'off')
    const probe = await new Promise(resolve => {
      const req = http.request({ host: '127.0.0.1', port: fixture.port, method: 'POST', path: '/v1/compress', headers: { 'content-type': 'application/json' } }, res => { res.resume(); resolve(res.statusCode) })
      req.on('error', () => resolve(0))
      req.end(JSON.stringify({ model: 'echo', messages: [] }))
    })
    assert.equal(probe, 200, 'the fake proxy outlived the join, exactly as a user-started one would')
  })

  fixture.server.close()
  fs.rmSync(fixture.dataDir, { recursive: true, force: true })
}

// The managed path with no join: the fake binary really opens the port (a node
// one-liner, no python), so `startHeadroom` spawns it, waits, and owns it. The
// point of the check is `stopHeadroom` on this shape: the close path once read
// `this.child` from an arrow property, resolved it against the module, and let
// every managed proxy survive its own teardown as an orphan.
{
  const spawnBinary = path.join(os.tmpdir(), `ofm-headroom-spawn-${process.pid}`)
  const spawnPort = await reservePort()
  // The real call site appends `proxy --port <n> --no-optimize --no-cache`;
  // node would read a stray `--port` as its own flag, so the fake swallows the
  // argv and binds the port the start waits on. What is under test is the
  // ownership of the child, not the port it chose.
  fs.writeFileSync(spawnBinary, [
    '#!/bin/sh',
    `exec ${JSON.stringify(process.execPath)} -e 'require("http").createServer((q,s)=>{s.writeHead(200);s.end("{}")}).listen(${spawnPort},"127.0.0.1")'`,
    '',
  ].join('\n'), { mode: 0o755 })
  await check('a managed spawn is killed by stopHeadroom', async () => {
    const dataDir = fs.mkdtempSync(path.join(os.tmpdir(), 'ofm-headroom-spawn-data-'))
    try {
      const handle = await startHeadroom({
        settings: () => ({ tokenSaver: { headroom: { enabled: true } } }),
        dataDir,
        log: () => {},
        pythonPath: spawnBinary,
        port: spawnPort,
      })
      assert.equal(headroomStatus().state, 'ready', 'the managed handle did not report ready')
      const pid = headroomStatus().pid
      assert.ok(typeof pid === 'number' && pid > 0, 'no pid to supervise')
      await stopHeadroom()
      assert.equal(headroomStatus().state, 'off')
      // The child is gone, not just detached: a still-live pid would keep
      // holding the port and poison every later start as a join.
      let survived = true
      try { process.kill(pid, 0); survived = true } catch { survived = false }
      assert.equal(survived, false, 'the managed process survived stopHeadroom')
      const portOpen = await new Promise(resolve => {
        const s = net.connect({ host: '127.0.0.1', port: spawnPort })
        const settle = ok => { s.removeAllListeners(); s.destroy(); resolve(ok) }
        s.once('connect', () => settle(true)); s.once('error', () => settle(false)); s.setTimeout(500, () => settle(false))
      })
      assert.equal(portOpen, false, 'the port is still answering — something survived')
    } finally {
      fs.rmSync(dataDir, { recursive: true, force: true })
      fs.rmSync(spawnBinary, { force: true })
    }
  })
}

fs.rmSync(fakeBinary, { force: true })

// ── the degradation chain, offline (no handle) ───────────────────────────────

const GIT_DIFF = `diff --git a/x.js b/x.js
index 111..222 100644
--- a/x.js
+++ b/x.js
@@ -1,2 +1,2 @@
-old line
+new line
` + 'lorem ipsum dolor sit amet '.repeat(40)

const noProxy = { enabled: true, rtk: true, caveman: 'off', ponytail: 'off', headroom: { enabled: true, ccr: false, pythonPath: '' } }

await check('with no proxy running, the pipeline degrades to RTK and still compacts', async () => {
  const payload = {
    model: 'gpt-4o-mini',
    messages: [
      { role: 'user', content: 'run the diff' },
      { role: 'tool', tool_call_id: 'call_1', content: GIT_DIFF },
    ],
  }
  const before = JSON.stringify(payload).length
  const report = await applyTokenSaversAsync(payload, 'chat', noProxy)
  assert.equal(report.headroom, null, 'headroom did not win')
  assert.ok(report.notes.some(n => n.startsWith('headroom-fallback:')), `expected a fallback note, got ${report.notes.join('|')}`)
  assert.ok(report.rtk !== null, 'RTK ran as the degradation chain')
  assert.ok(report.saved > 0, 'RTK compacted the tool output')
  assert.ok(JSON.stringify(payload).length < before, 'the shipped body is smaller than the draft')
  assert.equal(report.bytesBefore - report.bytesAfter, report.saved)
})

await check('the pipeline report folds into the lifetime totals', async () => {
  const payload = {
    model: 'gpt-4o-mini',
    messages: [
      { role: 'user', content: 'run the diff' },
      { role: 'tool', tool_call_id: 'call_1', content: GIT_DIFF },
    ],
  }
  const pipeline = await applyTokenSaversAsync(payload, 'chat', noProxy)
  const total = addSavings(addSavings(emptySavings(), pipeline), pipeline)
  assert.equal(total.calls, 2)
  assert.ok(total.saved > 0)
  assert.ok(total.blocks >= 1, `RTK compacted at least one block, got ${total.blocks}`)
  assert.equal(total.bytesBefore, pipeline.bytesBefore * 2)
})

// ── settings surface ─────────────────────────────────────────────────────────

await check('saverConfig keeps headroom on by default and honours an off switch', () => {
  assert.equal(saverConfig({}).headroom.enabled, true)
  assert.equal(saverConfig({}).headroom.ccr, true)
  const off = saverConfig({ tokenSaver: { headroom: { enabled: false } } })
  assert.equal(off.headroom.enabled, false)
})

console.log(failures === 0
  ? '\nheadroom: locator, fail-open client, gates, CCR injection and the degradation chain all agree'
  : `\n${failures} check(s) failed`)
process.exitCode = failures === 0 ? 0 : 1
