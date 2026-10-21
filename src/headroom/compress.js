/**
 * The headroom compression step for the token-saver pipeline.
 *
 * Sits between prompt shaping and RTK: headroom is the stronger, model-aware
 * compressor (SmartCrusher for JSON, Kompress-v2-base for prose) and runs
 * first; RTK is the zero-dependency pattern-matcher that catches whatever
 * headroom missed or was unavailable to reach. Both report in the same byte
 * shape so the stats panel is one number, not two.
 *
 * Wire coverage: headroom speaks OpenAI natively. The `chat` wire is OpenAI
 * already, so it goes straight to the proxy. The `messages` (Anthropic) and
 * `responses` (OpenAI Responses) wires would need a full format round-trip
 * that headroom's HTTP API does not provide; on those wires this step is a
 * no-op and RTK carries the load. This is deliberate — a lossy conversion
 * that guesses at Anthropic block shapes is worse than no headroom at all.
 *
 * CCR: when the proxy compresses a block it caches the original and stamps
 * the compressed text with a hash marker. The tool the pipeline injects
 * (`headroom_retrieve`, see `./index.js`) is how the model asks for the
 * original back. The harness's tool loop resolves that call through
 * `headroomRetrieve` in `./index.js`.
 *
 * @module src/headroom/compress.js
 */

import { headroomCompress, headroomRetrieve, headroomRetrieveTool } from './index.js'

/**
 * Answer the model's `headroom_retrieve` calls on the way back in.
 *
 * The tool is declared to the provider, so a model that wants a dropped
 * original calls it — but the harness that executes tool calls has never heard
 * of it, and answers the call with an "unknown tool" stub. This is the seam
 * that makes the stub true: on the next request, the tool result carrying a
 * retrieve call id is replaced with the original the proxy cached for that
 * hash. The model asked once, the harness shrugged, and the plugin answered —
 * one round trip later, exactly what CCR promised.
 *
 * Runs before compression so a retrieved original is not immediately
 * re-shrunk in the same build; if the proxy re-compresses it later, it gets a
 * fresh marker and remains retrievable again.
 *
 * @param {object} payload - the request body about to be built
 * @returns {Promise<number>} how many tool results were replaced
 */
export async function resolveRetrieveCalls(payload) {
  if (!Array.isArray(payload.messages)) return 0
  // Map the call ids: only assistant messages declare a tool call's name, and
  // the hash lives in its JSON arguments. A malformed argument string is
  // ignored — the stub result stays and the model works from context.
  const wanted = new Map()
  for (const message of payload.messages) {
    if (message?.role !== 'assistant' || !Array.isArray(message.tool_calls)) continue
    for (const call of message.tool_calls) {
      if (call?.function?.name !== 'headroom_retrieve' || call.id === undefined) continue
      let hash = ''
      try { hash = String(JSON.parse(call.function.arguments ?? '')?.hash ?? '') } catch { hash = '' }
      if (hash !== '') wanted.set(String(call.id), hash)
    }
  }
  if (wanted.size === 0) return 0
  let filled = 0
  for (const message of payload.messages) {
    if (message?.role !== 'tool') continue
    const hash = wanted.get(String(message.tool_call_id ?? ''))
    if (hash === undefined) continue
    const result = await headroomRetrieve(hash)
    if (result.ok === true && result.content !== '') {
      message.content = result.content
      filled += 1
    }
  }
  return filled
}

/**
 * Run the headroom step over one payload.
 *
 * Mutates `payload.messages` in place when the wire is `chat` and the proxy
 * answers; a no-op on every other wire or when the proxy is not running.
 * Fail-open throughout: a proxy error, a timeout, or a non-`chat` wire all
 * return `{ ok: false }` and the caller falls through to RTK.
 *
 * @param {object} payload - the request body, already prompt-shaped
 * @param {'chat'|'messages'|'responses'} wire
 * @param {{ccr: boolean, pythonPath?: string}} config - the `tokenSaver.headroom` section
 * @returns {Promise<{ok: boolean, reason?: string, stats?: {bytesBefore: number, bytesAfter: number, saved: number, blocks: number}}}>}
 */
export async function applyHeadroom(payload, wire, config) {
  const report = { ok: false }
  if (payload === null || typeof payload !== 'object') return report
  if (config?.enabled === false) { report.reason = 'disabled'; return report }
  if (wire !== 'chat') { report.reason = `wire "${wire}" is not the OpenAI shape headroom speaks; RTK carries it`; return report }
  if (!Array.isArray(payload.messages) || payload.messages.length === 0) { report.reason = 'no messages'; return report }

  const bytesBefore = JSON.stringify(payload).length
  const result = await headroomCompress(payload.messages, { model: payload.model, ccr: config.ccr !== false })
  if (!result.ok) { report.reason = result.reason; return report }

  const compressed = result.result?.messages
  if (!Array.isArray(compressed) || compressed.length === 0) {
    report.reason = 'the proxy returned an empty message list'
    return report
  }

  // Safety: never ship a payload that is larger than the one we started with.
  // The comparison is whole-payload on both sides, and it includes the CCR
  // retrieve tool the injection below adds — measuring the compressed body
  // before its own tool declaration would bookkeeping a saving the wire never
  // sees. The draft is a shallow clone: the original survives until the
  // comparison passes.
  const draft = { ...payload, messages: compressed }
  if (Array.isArray(payload.tools)) draft.tools = [...payload.tools]
  if (config.ccr !== false) injectRetrieveTool(draft, ccrHashesOf(result.result))
  const bytesAfter = JSON.stringify(draft).length
  if (bytesAfter >= bytesBefore) {
    report.reason = `the compressed payload (${bytesAfter}B) is not smaller than the original (${bytesBefore}B); keeping the original`
    return report
  }

  payload.messages = draft.messages
  if (draft.tools !== undefined) payload.tools = draft.tools
  const saved = bytesBefore - bytesAfter
  report.ok = true
  report.stats = { bytesBefore, bytesAfter, saved, blocks: 1, ccrHashes: ccrHashesOf(result.result).length }
  return report
}

/**
 * Add the `headroom_retrieve` tool to the payload's tool list when the
 * compressed text actually carries CCR markers. Without the tool the markers
 * are unresolvable and the compression is one-way; a tool the model calls
 * that always 404s is worse than no tool, so it is injected only when there
 * is something to retrieve.
 */
function injectRetrieveTool(payload, hashes) {
  if (Array.isArray(hashes) === false || hashes.length === 0) return
  const tool = headroomRetrieveTool()
  if (!Array.isArray(payload.tools)) payload.tools = []
  if (!payload.tools.some(t => t?.function?.name === 'headroom_retrieve' || t?.name === 'headroom_retrieve')) {
    payload.tools.push(tool)
  }
}

/**
 * The CCR hashes of one compress result, the way the proxy itself reports
 * them: a `ccr_hashes` array on the response, and the same markers in the
 * rendered text (`Retrieve more: hash=<hex>`, `Retrieve original: hash=<hex>`,
 * `<<ccr:<hex>>>`) for anything the array missed — a row-drop path can embed
 * a marker without registering it. The union, deduped, is the retrieval set:
 * a hash in neither place has nothing cached to fetch, so the retrieve tool
 * would be declared for a 404 and the model would learn that compression is
 * a liar.
 */
function ccrHashesOf(result) {
  const hashes = new Set()
  for (const hash of Array.isArray(result?.ccr_hashes) ? result.ccr_hashes : []) {
    if (typeof hash === 'string' && /^[a-f0-9]{12,24}$/i.test(hash)) hashes.add(hash.toLowerCase())
  }
  for (const message of Array.isArray(result?.messages) ? result.messages : []) {
    const text = extractText(message)
    if (typeof text !== 'string') continue
    for (const match of text.matchAll(/(?:Retrieve (?:more|original): hash=|<<ccr:)([a-f0-9]{12,24})(?![a-f0-9])/gi)) {
      hashes.add(match[1].toLowerCase())
    }
  }
  return [...hashes]
}

/** The text of one OpenAI message, string or block-array form. */
function extractText(message) {
  if (message === null || typeof message !== 'object') return null
  if (typeof message.content === 'string') return message.content
  if (Array.isArray(message.content)) {
    return message.content.map(part => (part?.type === 'text' ? part.text : '')).filter(Boolean).join('\n')
  }
  return null
}
