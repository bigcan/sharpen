export const meta = {
  name: 'deep_strategy_audit',
  description: 'Tier-2 deep lifecycle audit: finder + skeptic per pillar (default: the 11 RL lifecycle pillars, data -> live; or caller-supplied pillars), skeptic-adjudicated severity, roadmap-ranked report. NOT diff-scoped — re-derives correctness regardless of what changed. Caught the sg1-btc X2 coarse-bar leak that hundreds of routine /audit runs missed.',
  whenToUse: 'Before promoting a strategy to capital (live/paper), before reading a deploy-gating WF/OOS verdict, on a calendar cadence per live strategy, or on demand. args: an object, or the same object JSON-encoded as a string: {"workstream": "sg1-btc", "scope": "all"} (workstream required; scope "all" or "P2,P3"). The default pillars P1-P11 audit the RL pipeline; for a non-RL workstream (TSMOM/TAILWIND books, ATL x Jev) also pass "pillars": [{"key", "title", "focus", "look"}, ...] to replace them and "context": "strategy summary, key files, ground rules" to inject into every finder, skeptic and synthesis prompt. Missing or malformed args abort before any agent starts.',
  phases: [
    { title: 'Find', detail: 'one finder agent per lifecycle pillar (parallel)' },
    { title: 'Adjudicate', detail: 'independent skeptic per pillar refutes weak claims + regrades severity' },
    { title: 'Synthesize', detail: 'merge, rank by (robustness x consistency / effort), write report' },
  ],
}

// ── Parameters ──────────────────────────────────────────────────────────────
// args is an object, or the same object JSON-encoded as a string (the Skill /
// slash-command path delivers a string). Keys:
//   workstream  REQUIRED, e.g. "sg1-btc", "atl-jev". Finders use it to locate configs/results;
//               it names the report docs/research/<workstream>_deep_lifecycle_audit_<date>.md.
//   scope       optional: "all" (default), a comma-list "P2,P3,P4", or an array ["P2", "P3"].
//   pillars     optional array of {key, title, focus, look} that REPLACES the default RL pillars,
//               for workstreams that never run the RL pipeline (linear TSMOM/TAILWIND books, the
//               ATL x Jev filing-text signal) and would otherwise be audited on code they never use.
//               Worked example (pillars J1-J7 + ground rules + P4-readiness verdict), a session copy:
//               ~/.claude/projects/C--FinRL-FinRL-Pro-DS/e39f52d8-c93d-4941-87e2-5a32df8b6818/
//               workflows/scripts/atl-jev-deep-audit-wf_121619a6-9d1.js
//   context     optional string injected into every finder, skeptic and synthesis prompt: what the
//               strategy is, its key files, ground rules (read-only, forbidden data windows, which
//               branch to read), and any extra verdict the report must deliver.
// Anything missing or malformed ABORTS before an agent starts; there is deliberately no fallback.
// The old `(args && args.workstream) || 'UNSPECIFIED'` read a JSON-string args as "no args" and
// produced docs/research/UNSPECIFIED_deep_lifecycle_audit_2026-06-{01,18,29}.md.
const ARG_KEYS = ['workstream', 'scope', 'pillars', 'context']
const PILLAR_FIELDS = ['key', 'title', 'focus', 'look']
const USAGE = 'Usage: args = {"workstream": "sg1-btc", "scope": "all"}, optionally plus "pillars": [{"key": "J1", "title": "...", "focus": "...", "look": "..."}, ...] and "context": "..." (an object or its JSON string).'

// Returns { problems, workstream, context, custom, pillars, scopeKeys }. Every problem is
// collected, so one failed launch reports all of them.
function parseArgs(raw, defaultPillars) {
  let a = raw
  if (typeof a === 'string') {
    try {
      a = JSON.parse(a)
    } catch (e) {
      return { problems: [`args is a string but not valid JSON (${e.message}); received ${JSON.stringify(raw.slice(0, 200))}`] }
    }
  }
  if (a === undefined || a === null) return { problems: ['no args: "workstream" is required'] }
  if (typeof a !== 'object' || Array.isArray(a)) {
    return { problems: [`args must be an object or a JSON string encoding one; got ${Array.isArray(a) ? 'an array' : typeof a}`] }
  }

  const problems = []
  const unknown = Object.keys(a).filter(k => !ARG_KEYS.includes(k))
  if (unknown.length) problems.push(`unknown key(s) ${unknown.join(', ')}; accepted: ${ARG_KEYS.join(', ')} (a misspelt key would otherwise be ignored silently)`)

  const workstream = typeof a.workstream === 'string' ? a.workstream.trim() : ''
  if (!workstream) problems.push('"workstream" is required: a non-empty string such as "sg1-btc"')
  else if (!/^[A-Za-z0-9][A-Za-z0-9._-]*$/.test(workstream)) problems.push(`"workstream" ${JSON.stringify(workstream)} may use only letters, digits, ".", "_" and "-" (it becomes part of the report filename)`)

  let context = null
  if (a.context != null) {
    if (typeof a.context === 'string' && a.context.trim()) context = a.context.trim()
    else problems.push('"context" must be a non-empty string')
  }

  const custom = a.pillars != null
  let pillars = custom ? null : defaultPillars
  if (custom) {
    if (!Array.isArray(a.pillars) || !a.pillars.length) {
      problems.push('"pillars" must be a non-empty array of {key, title, focus, look}')
    } else {
      const before = problems.length
      const seen = new Set()
      a.pillars.forEach((p, i) => {
        if (p === null || typeof p !== 'object' || Array.isArray(p)) {
          problems.push(`pillars[${i}] must be an object {key, title, focus, look}`)
          return
        }
        const extra = Object.keys(p).filter(f => !PILLAR_FIELDS.includes(f))
        if (extra.length) problems.push(`pillars[${i}] has unknown field(s) ${extra.join(', ')}; allowed: ${PILLAR_FIELDS.join(', ')}`)
        const empty = PILLAR_FIELDS.filter(f => typeof p[f] !== 'string' || !p[f].trim())
        if (empty.length) problems.push(`pillars[${i}] needs non-empty string field(s): ${empty.join(', ')}`)
        if (empty.includes('key')) return
        const k = p.key.trim().toUpperCase()
        if (!/^[A-Z0-9_-]+$/.test(k)) problems.push(`pillars[${i}].key ${JSON.stringify(p.key)} may use only letters, digits, "_" and "-" (it prefixes finding IDs)`)
        if (seen.has(k)) problems.push(`pillars[${i}].key ${JSON.stringify(p.key)} duplicates an earlier key`)
        seen.add(k)
      })
      if (problems.length === before) {
        pillars = a.pillars.map(p => ({ key: p.key.trim(), title: p.title.trim(), focus: p.focus.trim(), look: p.look.trim() }))
      }
    }
  }

  // scope: absent, "" or "all" = every pillar; otherwise every named key must exist.
  let scopeKeys = null
  const s = a.scope
  if (!(s == null || (typeof s === 'string' && ['', 'all'].includes(s.trim().toLowerCase())))) {
    const list = typeof s === 'string' ? s.split(',') : Array.isArray(s) ? s : null
    if (!list || list.some(k => typeof k !== 'string')) {
      problems.push('"scope" must be "all", a comma-list such as "P2,P3", or an array of pillar keys')
    } else {
      scopeKeys = list.map(k => k.trim().toUpperCase()).filter(Boolean)
      const known = pillars ? pillars.map(p => p.key.toUpperCase()) : null
      const stray = known ? scopeKeys.filter(k => !known.includes(k)) : []
      if (!scopeKeys.length) problems.push('"scope" names no pillar key')
      else if (stray.length) problems.push(`"scope" key(s) ${stray.join(', ')} match no pillar; valid: ${known.join(', ')} or "all"`)
    }
  }

  return { problems, workstream, context, custom, pillars, scopeKeys }
}

// ── DEFAULT pillars: the 11 RL lifecycle pillars (mirror docs/research/sg1_btc_strategy_audit_2026-05-29.md).
// Replaced wholesale when args.pillars is given.
const DEFAULT_PILLARS = [
  { key: 'P1', title: 'Data preparation & integrity',
    focus: 'OHLCV cleaning (clean_ohlcv.py, DATA-CLEAN), manifest provenance (sha256/source/gap_count), split boundaries (splitter.py, buffer_days/embargo), ffill/gap bridging, dedup/monotonic asserts, regime-coverage gate. Verify DATA-CLEAN is actually enforced (consumer exists), not self-certified.',
    look: 'scripts/clean_ohlcv.py, finrl_pro_ds/data/build_data_manifest.py, finrl_pro_ds/data/splitter.py, multiscale_handler._load_data, schemas/manifest.schema.json' },
  { key: 'P2', title: 'Feature engineering & normalization / leakage',
    focus: 'LEAK-1 EMA-Z per-split isolation AND LEAK-2 temporal causality. CAUS-01: coarse->base map must hit the last CLOSED bar (searchsorted on base_ts - scale_ns); CAUS-02: sim==live at a MID-INTERVAL truncation (not just end-of-data); CAUS-03: ATR/full-series warmup carry across split; any future-indexed array reaching obs. THIS is the pillar that caught the X2 leak — be maximally adversarial.',
    look: 'finrl_pro_ds/data/multiscale_handler.py, finrl_pro_ds/data/feature_engineering.py, finrl_pro_ds/crypto/live/live_obs_builder.py, tests/data/test_multiscale_causality.py, tests/crypto/test_live_obs_parity.py' },
  { key: 'P3', title: 'Environment mechanics & execution realism',
    focus: 'Step ordering (decide t, earn t->t+1), SHORT-ACCT (no notional_debt, debt-free buyback), fill model (taker-at-close + slippage; is slippage > 0?), conditional ATR-cap sim-vs-live divergence, DD-termination mode sim-vs-live, deadband. Demand a real-env golden-trajectory test (mocks hide accounting bugs).',
    look: 'finrl_pro_ds/envs/continuous_swing_env.py, crypto risk manager, risk_shaping_wrapper.py, signal_gated_wrapper.py, broker classes' },
  { key: 'P4', title: 'Reward design',
    focus: 'DSR formula vs MATH-R01 (pre-update A/B, 3/2 exponent, 0.5 factor, var clamp) — demand a numeric tripwire test. Turnover-penalty presence, cost-in-reward effectiveness vs DSR scale-invariance, DD-penalty scale tuning, normalize_accumulated_reward semantics, reward<->PF (HPO objective) alignment.',
    look: 'finrl_pro_ds/envs/dsr.py, finrl_pro_ds/envs/continuous_swing_env.py, signal_gated_wrapper.py, risk_shaping_wrapper.py, ~/.claude/skills/math/FORMULAS.md (MATH-R01/R04)' },
  { key: 'P5', title: 'Training & HPO',
    focus: 'BUG-01 (objective=PF, reward locked). Training-health kill gates (entropy/q-div/action-sat) — DECLARED vs WIRED? Training-budget multiplicity [15,40] check actually in validate_config? Per-trial seeding/reproducibility, val-window single-point risk, reward<->PF Spearman computed?',
    look: 'finrl_pro_ds/training/objective.py, sac_trainer.py, scripts/run_full_pipeline.py, evaluate.py, distributed_hpo_worker.py, scripts/validate_config.py, docs/sharpops.md' },
  { key: 'P6', title: 'Validation — L1 multiseed (Stage 2)',
    focus: 'Gate code workstream-agnostic (or XAUUSD-hardcoded)? Seed convergence — does CV measure robustness or seed-degeneracy (inter-seed action corr)? val_argmax top-3 vs test top-3 decoupling. Full-N eval baseline + challenge_target_hit_rate present?',
    look: 'scripts/auto_queue_wf_after_l1.py, scripts/launch_l1_multiseed.py, results/<workstream>*/seed_report.json, *.gates.yaml' },
  { key: 'P7', title: 'Walk-forward + stress (Stage 3)',
    focus: 'Cost-corrected seeds re-derived or inherited from a cost-free cohort? Fixed-lot stress sub-report present (post-S466)? DD truncation capping per-fold DD + G4 buffer? Aggregation rule graded == rule chosen at Stage 2.5 (chosen_rule from verdict)? Fold-overlap inflating G5 CV? Every-fold-profitable gate?',
    look: 'scripts/*ensemble_eval*.py, <workstream>*.gates.yaml, finrl_pro_ds/data/splitter.py, scripts/validate_config.py, results/<workstream>*_wf/verdict.json' },
  { key: 'P8', title: 'Recent-OOS & compliance (Stage 4)',
    focus: 'OOS verdict tests the DEPLOYED seed/bundle (not a retired one)? Honest cost (taker+slippage, not cheap)? Baseline = WF-median PF? Sanity bounds on absurd returns? Stage actually wired in run_full_pipeline --stage oos? FTMO/Velotrade compliance denominators.',
    look: 'configs/*oos*.yaml, scripts/aggregate_*oos*.py, scripts/ftmo_compliance_report.py, <workstream>*.gates.yaml, results/*oos*/verdict.json' },
  { key: 'P9', title: 'Ensemble formation (Stage 2.5)',
    focus: 'Diversity-aware selector actually selecting (pool > K) or dead code? seed_pfs metric-consistent with the script PFs? val reused with multiplicity correction? ens_pf_weighted weights from VAL (not TEST) PFs? chosen_rule single-source-of-truth?',
    look: 'scripts/*ensemble_eval*.py, results/<workstream>*/verdict.json, finrl_pro_ds ensemble bundle writer, seed_report.json' },
  { key: 'P10', title: 'Live / drift / sim-to-live consistency',
    focus: 'Live cost model == corrected training cost? Staleness cadence (max_age_days) matches the timeframe? §4.5 retrain triggers wired (cost_drift computed)? Drift-threshold precedence single-source (gates.drift)? norm-warmup pkl inside bundle SHA chain / fail-closed? Action-drift baseline regime-conditioned.',
    look: 'finrl_pro_ds/crypto/live/live_engine.py, scripts/check_retrain_triggers.py, agent_loader.py, ensemble_bundle.py, configs/live_<workstream>*.yaml' },
  { key: 'P11', title: 'External SOTA benchmark',
    focus: 'Compare the pipeline to financial-ML best practice: deflated-Sharpe / PBO / CSCV (Bailey & Lopez de Prado), purged/embargoed CPCV, risk-sensitive reward (CVaR/Calmar/distributional), domain randomization / obs-noise / exec-failure stress, regime-conditioning. Name the single highest-value MISSING overfitting control. Query the NotebookLM KB (4aef5475-7fec-4d1f-96a7-efb3cafbb371) before web.',
    look: 'docs/sharpops.md, results verdicts, literature via Researcher/NotebookLM' },
]

const parsed = parseArgs(args, DEFAULT_PILLARS)
if (parsed.problems.length) {
  log(`ABORTED: invalid args, no agent was started.\n- ${parsed.problems.join('\n- ')}\n${USAGE}`)
  return { error: 'invalid_args', problems: parsed.problems, usage: USAGE }
}
const { workstream, context, custom } = parsed
const selected = parsed.scopeKeys ? parsed.pillars.filter(p => parsed.scopeKeys.includes(p.key.toUpperCase())) : parsed.pillars
const scopeLabel = parsed.scopeKeys ? parsed.scopeKeys.join(',') : 'all'
log(`Deep lifecycle audit — workstream="${workstream}", ${custom ? 'caller-supplied' : 'default RL'} pillars=${selected.map(p => p.key).join(',')}${context ? `, context=${context.length} chars` : ''}`)

const ADVERSARIAL = [
  'Be ADVERSARIAL: try to break it, do not certify a checklist. A truthful "invariant PASS" is worthless if the invariant never covered the real bug.',
  'Hunt these silent-failure classes: (1) look-ahead/temporal leak — any feature at bar t seeing data > t; (2) train<->serve (sim<->live) skew; (3) frictionless artifact — a too-good metric from zero/stale cost; (4) declared-but-not-wired — a safeguard/gate/invariant in a doc/config/comment with NO python consumer; (5) test blind spot — passing tests that never exercise the boundary where the bug lives; (6) latent-in-stable-code — the real bug in an UNCHANGED line, not a recent diff.',
  'Cite EVERY claim with file:line. No claim without evidence. Prefer false positives over false negatives, but mark uncertainty honestly.',
].join(' ')

// Caller context goes right under each prompt's header and outranks the generic
// instructions (the synthesis step keeps one carve-out: it must write its report).
const CONTEXT = context && `WORKSTREAM CONTEXT (supplied by the caller; where it conflicts with a generic instruction in this prompt, follow the context):\n${context}`
// The default pillars cite RL-pipeline invariants and protocol stages; caller-supplied
// pillars get a scoped reading rule and stage-neutral roadmap buckets instead.
const READING = custom
  ? 'Apply the CLAUDE.md invariants this workstream\'s code actually exercises (LEAK-2 temporal causality always; DATA-CLEAN wherever OHLCV is consumed). Do not audit RL-pipeline code (envs, reward, HPO, multiseed, ensembles, live engine) unless the pillar focus or the context puts it in scope.'
  : 'Also read CLAUDE.md (invariants incl. LEAK-1/LEAK-2/SHORT-ACCT/BUG-01/DATA-CLEAN/PF-XCHECK) and docs/sharpops.md for the stage contract.'
const BUCKETS = custom
  ? 'NOW (code/test/doc/config, no new run) / NEXT (needs a re-run or new data) / RESEARCH (open question)'
  : 'NOW (config/test/doc, no retrain) / NEXT (needs a stage re-run) / RESEARCH (open question)'

const FINDINGS_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['pillar', 'findings'],
  properties: {
    pillar: { type: 'string' },
    findings: {
      type: 'array',
      items: {
        type: 'object',
        additionalProperties: false,
        required: ['id', 'severity', 'finding', 'evidence', 'improvement', 'effort'],
        properties: {
          id: { type: 'string', description: 'e.g. P2-01' },
          severity: { type: 'string', enum: ['S1', 'S2', 'S3', 'S4'], description: 'S1=will cause live loss/leakage, S2=materially weakens robustness, S3=rigor/hygiene gap, S4=minor or POSITIVE' },
          finding: { type: 'string' },
          evidence: { type: 'string', description: 'file:line citations proving it' },
          improvement: { type: 'string' },
          effort: { type: 'string', enum: ['Now', 'Next', 'Research'] },
        },
      },
    },
  },
}

const ADJUDICATED_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['pillar', 'verdicts'],
  properties: {
    pillar: { type: 'string' },
    verdicts: {
      type: 'array',
      items: {
        type: 'object',
        additionalProperties: false,
        required: ['id', 'verdict', 'adjusted_severity', 'note'],
        properties: {
          id: { type: 'string' },
          verdict: { type: 'string', enum: ['CONFIRMED', 'REFUTED', 'NEEDS-DATA'] },
          adjusted_severity: { type: 'string', enum: ['S1', 'S2', 'S3', 'S4'] },
          note: { type: 'string', description: 'why confirmed/refuted; corrected evidence if the finder mis-cited' },
        },
      },
    },
  },
}

function finderPrompt(p) {
  return [
    `TIER-2 DEEP LIFECYCLE AUDIT — Pillar ${p.key}: ${p.title}. Workstream: "${workstream}".`,
    CONTEXT,
    ADVERSARIAL,
    `Pillar focus: ${p.focus}`,
    `Start by reading (then follow imports/configs/results as needed): ${p.look}`,
    READING,
    `Return ALL findings for this pillar (include explicit POSITIVES as S4 so the skeptic can credit them). Use IDs like ${p.key}-01, ${p.key}-02. Every finding needs file:line evidence.`,
  ].filter(Boolean).join('\n\n')
}

function skepticPrompt(p, findings) {
  return [
    `TIER-2 SKEPTIC — adjudicate the finder's claims for Pillar ${p.key}: ${p.title}. Workstream "${workstream}".`,
    CONTEXT,
    `You are an independent skeptic. For EACH finding: re-open the cited file:line, verify the claim is real, and either CONFIRM, REFUTE (with the counter-evidence), or mark NEEDS-DATA (requires a run/measurement not available). Re-grade severity yourself (S1..S4) — finders over- and under-rate. Correct any mis-cited evidence. Do NOT rubber-stamp; a finding you cannot independently reproduce from the code is NOT confirmed.`,
    `Finder output (JSON): ${JSON.stringify(findings)}`,
  ].filter(Boolean).join('\n\n')
}

// Finder -> skeptic per pillar; pipeline so each pillar's skeptic starts as soon
// as its finder returns (no barrier — pillars run fully independently). A dead
// finder or skeptic (agent() -> null) makes the item null, so the pillar is reported
// as NOT covered instead of reaching synthesis as an empty "adjudicated" pillar.
const adjudicated = await pipeline(
  selected,
  p => agent(finderPrompt(p), { label: `find:${p.key}`, phase: 'Find', schema: FINDINGS_SCHEMA }),
  (findings, p) => findings && agent(skepticPrompt(p, findings), { label: `skeptic:${p.key}`, phase: 'Adjudicate', schema: ADJUDICATED_SCHEMA })
    .then(adj => adj && ({ pillar: p, findings: findings.findings, verdicts: adj.verdicts })),
)

const ok = adjudicated.filter(Boolean)
const missing = selected.map(p => p.key).filter(k => !ok.some(x => x.pillar.key === k))
if (missing.length) log(`WARNING: no adjudicated result for pillar(s) ${missing.join(', ')}; the report must state they were NOT covered`)

// ── Synthesis: merge finder findings with skeptic verdicts, rank, write report ──
const synthPrompt = [
  `TIER-2 DEEP LIFECYCLE AUDIT — SYNTHESIS. Workstream: "${workstream}".`,
  CONTEXT,
  context && 'Exception to any read-only rule in the context: this step writes exactly one file, the audit report (at the path below unless the context names another), and changes nothing else.',
  `You are given, per pillar, the finder's findings and the skeptic's adjudication. Produce the final audit report.`,
  missing.length && `Pillars with NO adjudicated result (finder or skeptic failed): ${missing.join(', ')}. State in the Executive Summary and the verification log that they were NOT covered.`,
  `Rules: keep only CONFIRMED and NEEDS-DATA findings (drop REFUTED, but note the refutation count per pillar). Use the SKEPTIC's adjusted_severity. Credit S4 positives. Rank a prioritized roadmap in buckets ${BUCKETS}, ordered within bucket by (robustness x consistency impact) / effort.`,
  `Lead with an Executive Summary that names the single dominant cross-cutting fault and the top S1/S2 themes. Then a per-pillar findings table (ID, Sev, Finding, Evidence, Improvement, Effort), then the roadmap, then a verification log (confirmed/refuted/needs-data counts per pillar).`,
  `WRITE the report to docs/research/${workstream}_deep_lifecycle_audit_<YYYY-MM-DD>.md (get the date via a shell 'date +%F'). Use the Write tool. Mirror the structure of docs/research/sg1_btc_strategy_audit_2026-05-29.md.`,
  `Then RETURN a short text: the report path + the executive summary + the count of confirmed S1/S2 findings.`,
  `Adjudicated input (JSON): ${JSON.stringify(ok)}`,
].filter(Boolean).join('\n\n')

const report = await agent(synthPrompt, { label: 'synthesize', phase: 'Synthesize' })

return { workstream, scope: scopeLabel, pillar_set: custom ? 'caller-supplied' : 'default-rl', pillars_run: ok.map(x => x.pillar.key), pillars_missing: missing, report }
