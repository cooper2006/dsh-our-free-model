/**
 * The token-saving pipeline.
 *
 * Three independent savers run over one outbound payload, in this order:
 *
 * 1. **Prompt shaping** (`./prompts.js` + `./inject.js`) — a terse-response or
 *    YAGNI instruction is appended to the system prompt. This changes what the
 *    model *writes*, so it is the only saver that can reduce output tokens.
 * 2. **Model-level compression** (`../headroom/`) — the whole message list
 *    goes to a local headroom proxy (SmartCrusher for JSON, Kompress-v2-base
 *    for prose) and comes back smaller, with CCR markers the model can pull
 *    originals through. It runs only on the OpenAI wire and only while its
 *    proxy answers; every decline is a reason string, never a failed request.
 * 3. **Tool-output compaction** (`./rtk/`) — headroom's degradation chain, and
 *    the saver that needs nothing: a `git diff`, a `grep -rn`, an `ls -la` or
 *    a package-manager log is re-encoded into the same information at a
 *    fraction of the bytes before it ever reaches the wire. It runs when
 *    headroom did not take the payload. This is the one saver that is invisible
 *    to the model's reasoning: the compacted text is what the model reads.
 * 4. Nothing else. Response-side trimming (dropping reasoning frames) is
 *    deliberately absent — see `src/stream.js` for why the reasoning channel is
 *    measured rather than muzzled.
 *
 * Every saver is fail-open and independently switchable, and the pipeline
 * reports what it did in bytes so the savings panel shows a measurement instead
 * of a claim. A saver that throws is a saver that is skipped: a request must
 * never fail because the thing that was supposed to make it cheaper broke.
 *
 * @module src/token-saver/index.js
 */

import { compressRequest } from './rtk/index.js'
import { activePrompts } from './prompts.js'
import { injectSystem } from './inject.js'
import { applyHeadroom, resolveRetrieveCalls } from '../headroom/compress.js'

/** Prompt families this pipeline knows, for the settings schema and the UI. */
export const PROMPT_FAMILIES = ['caveman', 'ponytail']

/**
 * Read the `tokenSaver` settings section into the shape the pipeline wants.
 *
 * Every field is optional and every default is "off" except RTK, which is the
 * only saver whose output the user can verify at a glance (the compacted tool
 * result is right there in the transcript) and the only one that cannot change
 * the model's behaviour. A prompt-level saver changes the assistant's voice, so
 * it is opt-in.
 */
export function saverConfig(settings) {
  const section = settings?.tokenSaver ?? {}
  const prompts = section.prompts ?? {}
  const headroom = section.headroom ?? {}
  return {
    enabled: section.enabled !== false,
    rtk: section.rtk !== false,
    caveman: PROMPT_FAMILIES.includes('caveman') ? String(prompts.caveman ?? 'off') : 'off',
    ponytail: String(prompts.ponytail ?? 'off'),
    headroom: {
      enabled: headroom.enabled !== false,
      ccr: headroom.ccr !== false,
      pythonPath: typeof headroom.pythonPath === 'string' ? headroom.pythonPath : '',
    },
    /** Per-request header/query opt-out is handled by the caller; this is the
     *  stored preference only. */
  }
}

/**
 * Run every saver over one payload, in place.
 *
 * Mutating in place is what the adapter expects: `buildPayload` hands back a
 * fresh object per attempt, `applyFingerprint` already mutates it, and the
 * continuation path re-builds rather than reusing. Returning a copy would make
 * the fingerprint's rename map and the payload disagree about which object is
 * on the wire.
 *
 * @param {object} payload - the request body, already fingerprinted
 * @param {'chat'|'messages'|'responses'} wire
 * @param {object} config - from `saverConfig()`
 * @returns {{prompt: object|null, rtk: object|null, saved: number, bytesBefore: number, bytesAfter: number, notes: string[]}}
 */
export function applyTokenSavers(payload, wire, config) {
  const report = { prompt: null, rtk: null, saved: 0, bytesBefore: 0, bytesAfter: 0, notes: [] }
  if (config?.enabled === false || payload === null || typeof payload !== 'object') return report

  // Prompt shaping first: the injected text is a system instruction, never a
  // tool result, so the order does not change RTK's input — but running it first
  // means the reported byte counts describe the payload that actually shipped.
  if (config.caveman !== 'off' || config.ponytail !== 'off') {
    try {
      const prompts = activePrompts({ caveman: config.caveman, ponytail: config.ponytail })
      if (prompts.length > 0) {
        const text = prompts.map(entry => entry.text).join('\n\n')
        const result = injectSystem(payload, wire, text)
        report.prompt = { injected: result.injected === true, reason: result.reason ?? '', families: prompts.map(entry => `${entry.id}:${entry.level}`) }
        if (result.injected !== true && result.reason !== undefined) report.notes.push(`prompt:${result.reason}`)
      }
    } catch (error) {
      report.notes.push(`prompt-failed:${error?.message ?? error}`)
    }
  }

  if (config.rtk !== false) {
    try {
      const result = compressRequest(payload, wire, true)
      report.rtk = result.stats
      report.saved = Number(result.stats?.saved) || 0
      report.bytesBefore = Number(result.stats?.bytesBefore) || 0
      report.bytesAfter = Number(result.stats?.bytesAfter) || 0
    } catch (error) {
      report.notes.push(`rtk-failed:${error?.message ?? error}`)
    }
  }
  return report
}

/**
 * The async pipeline: prompt shaping → headroom → RTK as the degradation chain.
 *
 * Headroom is the primary compressor: when its proxy is reachable and the wire
 * is its OpenAI shape it does the whole-payload compression (SmartCrusher for
 * JSON, Kompress-v2-base for prose) in one model-aware pass, and RTK is
 * skipped — running RTK on an already-compressed payload would just burn
 * bytes chasing patterns headroom already resolved. When headroom is off,
 * disabled, on a wire it cannot speak, or its proxy is unreachable, RTK is
 * the fallback that keeps tool outputs compacted with zero external service.
 *
 * The report carries the winning stage's byte measurement under `headroom`
 * or `rtk` and the same totals under `bytesBefore`/`bytesAfter`/`saved` so
 * `addSavings` and the stats panel stay one number either way.
 *
 * @param {object} payload - the request body, already fingerprinted
 * @param {'chat'|'messages'|'responses'} wire
 * @param {object} config - from `saverConfig()`
 * @returns {Promise<{prompt: object|null, rtk: object|null, headroom: object|null, saved: number, bytesBefore: number, bytesAfter: number, notes: string[]}>}
 */
export async function applyTokenSaversAsync(payload, wire, config) {
  const report = { prompt: null, rtk: null, headroom: null, retrieveFilled: 0, saved: 0, bytesBefore: 0, bytesAfter: 0, notes: [] }
  if (config?.enabled === false || payload === null || typeof payload !== 'object') return report

  // First the retrieve round trip: a tool result that is really a harness stub
  // for a `headroom_retrieve` call becomes the cached original it asked for.
  // Ahead of every compressor so a retrieved original is not re-shrunk in the
  // same build. Only the chat wire declares the retrieve tool today, so this is
  // the only place a stub of that name can arrive.
  if (config.headroom?.enabled !== false && wire === 'chat') {
    try {
      report.retrieveFilled = await resolveRetrieveCalls(payload)
    } catch (error) {
      report.notes.push(`retrieve-failed:${error?.message ?? error}`)
    }
  }

  // Prompt shaping first: the injected text is a system instruction, never a
  // tool result, so it runs ahead of whichever body compressor follows and the
  // reported byte counts describe the payload that actually ships.
  if (config.caveman !== 'off' || config.ponytail !== 'off') {
    try {
      const prompts = activePrompts({ caveman: config.caveman, ponytail: config.ponytail })
      if (prompts.length > 0) {
        const text = prompts.map(entry => entry.text).join('\n\n')
        const result = injectSystem(payload, wire, text)
        report.prompt = { injected: result.injected === true, reason: result.reason ?? '', families: prompts.map(entry => `${entry.id}:${entry.level}`) }
        if (result.injected !== true && result.reason !== undefined) report.notes.push(`prompt:${result.reason}`)
      }
    } catch (error) {
      report.notes.push(`prompt-failed:${error?.message ?? error}`)
    }
  }

  let headroomWon = false
  if (config.headroom?.enabled !== false) {
    try {
      const result = await applyHeadroom(payload, wire, config.headroom)
      if (result.ok === true) {
        headroomWon = true
        report.headroom = result.stats
        report.saved = Number(result.stats?.saved) || 0
        report.bytesBefore = Number(result.stats?.bytesBefore) || 0
        report.bytesAfter = Number(result.stats?.bytesAfter) || 0
      } else {
        report.notes.push(`headroom-fallback:${result.reason}`)
      }
    } catch (error) {
      report.notes.push(`headroom-failed:${error?.message ?? error}`)
    }
  }

  // The degradation chain: RTK only when headroom did not take the payload.
  if (!headroomWon && config.rtk !== false) {
    try {
      const result = compressRequest(payload, wire, true)
      report.rtk = result.stats
      report.saved = Number(result.stats?.saved) || 0
      report.bytesBefore = Number(result.stats?.bytesBefore) || 0
      report.bytesAfter = Number(result.stats?.bytesAfter) || 0
    } catch (error) {
      report.notes.push(`rtk-failed:${error?.message ?? error}`)
    }
  }
  return report
}

/** A zeroed lifetime accumulator, so the stats document has a stable shape. */
export function emptySavings() {
  return { bytesBefore: 0, bytesAfter: 0, saved: 0, blocks: 0, calls: 0, promptCalls: 0 }
}

/**
 * Fold one pipeline report into a lifetime accumulator.
 *
 * `calls` counts every request the pipeline ran on — including the ones where
 * nothing was compressible — because "0 bytes saved over 400 calls" and "0
 * bytes saved over 0 calls" are different facts, and only one of them means the
 * saver is working. Blocks counts whichever stage won the request: RTK's own
 * hits, or headroom's one whole-payload block per compressed message list.
 */
export function addSavings(total, report) {
  const base = total ?? emptySavings()
  const blocks = Array.isArray(report?.rtk?.hits) ? report.rtk.hits.length : (Number(report?.headroom?.blocks) || 0)
  return {
    bytesBefore: base.bytesBefore + (Number(report?.bytesBefore) || 0),
    bytesAfter: base.bytesAfter + (Number(report?.bytesAfter) || 0),
    saved: base.saved + (Number(report?.saved) || 0),
    blocks: base.blocks + blocks,
    calls: base.calls + 1,
    promptCalls: base.promptCalls + (report?.prompt?.injected === true ? 1 : 0),
  }
}

/** Percent saved, rounded, or 0 when nothing was measured. */
export function savingsPercent(total) {
  const before = Number(total?.bytesBefore) || 0
  const saved = Number(total?.saved) || 0
  if (before <= 0) return 0
  return Math.round((saved / before) * 100)
}

/** The one-line summary the logs and the panel show. */
export function describeSavingsTotal(total) {
  const safe = total ?? emptySavings()
  const percent = savingsPercent(safe)
  if (safe.calls === 0) return 'token saver: no requests measured yet'
  return `token saver: ${formatBytes(safe.saved)} saved over ${safe.calls} request(s) (${percent}%), ${safe.blocks} block(s) compacted`
}

export function formatBytes(value) {
  const bytes = Number(value) || 0
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`
}
