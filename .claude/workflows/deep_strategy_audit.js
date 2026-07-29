export const meta = {
  name: 'deep_strategy_audit',
  description: 'Tier-2 deep lifecycle audit: finder + skeptic per pillar across 11 lifecycle pillars (data -> live), skeptic-adjudicated severity, roadmap-ranked report. NOT diff-scoped — re-derives correctness regardless of what changed. Caught the sg1-btc X2 coarse-bar leak that hundreds of routine /audit runs missed.',
  whenToUse: 'Before promoting a strategy to capital (live/paper), before reading a deploy-gating WF/OOS verdict, on a calendar cadence per live strategy, or on demand. Pass args:{workstream, scope}.',
  phases: [
    { title: 'Find', detail: 'one finder agent per lifecycle pillar (parallel)' },
    { title: 'Adjudicate', detail: 'independent skeptic per pillar refutes weak claims + regrades severity' },
    { title: 'Synthesize', detail: 'merge, rank by (robustness x consistency / effort), write report' },
  ],
}

// ── Parameters (args: { workstream, scope }) ────────────────────────────────
// workstream: e.g. "sg1-btc", "gmgp1-xauusd". Used by finders to locate configs/results.
// scope: "all" (default) or a comma-list of pillar keys, e.g. "P2,P3,P4".
const workstream = (args && args.workstream) || 'UNSPECIFIED'
const scopeRaw = (args && args.scope) || 'all'
const scopeKeys = scopeRaw === 'all' ? null : scopeRaw.split(',').map(s => s.trim().toUpperCase())

// ── The 11 lifecycle pillars (mirrors docs/research/sg1_btc_strategy_audit_2026-05-29.md) ──
const PILLARS = [
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
    look: 'finrl_pro_ds/training/objective.py, sac_trainer.py, scripts/run_full_pipeline.py, evaluate.py, distributed_hpo_worker.py, scripts/validate_config.py, docs/protocol_v2.md' },
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
    look: 'docs/protocol_v2.md, results verdicts, literature via Researcher/NotebookLM' },
]

const selected = scopeKeys ? PILLARS.filter(p => scopeKeys.includes(p.key)) : PILLARS
if (!selected.length) {
  log(`No pillars matched scope="${scopeRaw}". Valid keys: ${PILLARS.map(p => p.key).join(',')} or "all".`)
  return { error: 'empty_scope', scope: scopeRaw }
}
log(`Deep lifecycle audit — workstream="${workstream}", pillars=${selected.map(p => p.key).join(',')}`)

const ADVERSARIAL = [
  'Be ADVERSARIAL: try to break it, do not certify a checklist. A truthful "invariant PASS" is worthless if the invariant never covered the real bug.',
  'Hunt these silent-failure classes: (1) look-ahead/temporal leak — any feature at bar t seeing data > t; (2) train<->serve (sim<->live) skew; (3) frictionless artifact — a too-good metric from zero/stale cost; (4) declared-but-not-wired — a safeguard/gate/invariant in a doc/config/comment with NO python consumer; (5) test blind spot — passing tests that never exercise the boundary where the bug lives; (6) latent-in-stable-code — the real bug in an UNCHANGED line, not a recent diff.',
  'Cite EVERY claim with file:line. No claim without evidence. Prefer false positives over false negatives, but mark uncertainty honestly.',
].join(' ')

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
    ADVERSARIAL,
    `Pillar focus: ${p.focus}`,
    `Start by reading (then follow imports/configs/results as needed): ${p.look}`,
    `Also read CLAUDE.md (invariants incl. LEAK-1/LEAK-2/SHORT-ACCT/BUG-01/DATA-CLEAN/PF-XCHECK) and docs/protocol_v2.md for the stage contract.`,
    `Return ALL findings for this pillar (include explicit POSITIVES as S4 so the skeptic can credit them). Use IDs like ${p.key}-01, ${p.key}-02. Every finding needs file:line evidence.`,
  ].join('\n\n')
}

function skepticPrompt(p, findings) {
  return [
    `TIER-2 SKEPTIC — adjudicate the finder's claims for Pillar ${p.key}: ${p.title}. Workstream "${workstream}".`,
    `You are an independent skeptic. For EACH finding: re-open the cited file:line, verify the claim is real, and either CONFIRM, REFUTE (with the counter-evidence), or mark NEEDS-DATA (requires a run/measurement not available). Re-grade severity yourself (S1..S4) — finders over- and under-rate. Correct any mis-cited evidence. Do NOT rubber-stamp; a finding you cannot independently reproduce from the code is NOT confirmed.`,
    `Finder output (JSON): ${JSON.stringify(findings)}`,
  ].join('\n\n')
}

// Finder -> skeptic per pillar; pipeline so each pillar's skeptic starts as soon
// as its finder returns (no barrier — pillars run fully independently).
const adjudicated = await pipeline(
  selected,
  p => agent(finderPrompt(p), { label: `find:${p.key}`, phase: 'Find', schema: FINDINGS_SCHEMA }),
  (findings, p) => agent(skepticPrompt(p, findings), { label: `skeptic:${p.key}`, phase: 'Adjudicate', schema: ADJUDICATED_SCHEMA })
    .then(adj => ({ pillar: p, findings: findings && findings.findings, verdicts: adj && adj.verdicts })),
)

const ok = adjudicated.filter(Boolean)

// ── Synthesis: merge finder findings with skeptic verdicts, rank, write report ──
const synthPrompt = [
  `TIER-2 DEEP LIFECYCLE AUDIT — SYNTHESIS. Workstream: "${workstream}".`,
  `You are given, per pillar, the finder's findings and the skeptic's adjudication. Produce the final audit report.`,
  `Rules: keep only CONFIRMED and NEEDS-DATA findings (drop REFUTED, but note the refutation count per pillar). Use the SKEPTIC's adjusted_severity. Credit S4 positives. Rank a prioritized roadmap in buckets NOW (config/test/doc, no retrain) / NEXT (needs a stage re-run) / RESEARCH (open question), ordered within bucket by (robustness x consistency impact) / effort.`,
  `Lead with an Executive Summary that names the single dominant cross-cutting fault and the top S1/S2 themes. Then a per-pillar findings table (ID, Sev, Finding, Evidence, Improvement, Effort), then the roadmap, then a verification log (confirmed/refuted/needs-data counts per pillar).`,
  `WRITE the report to docs/research/${workstream}_deep_lifecycle_audit_<YYYY-MM-DD>.md (get the date via a shell 'date +%F'). Use the Write tool. Mirror the structure of docs/research/sg1_btc_strategy_audit_2026-05-29.md.`,
  `Then RETURN a short text: the report path + the executive summary + the count of confirmed S1/S2 findings.`,
  `Adjudicated input (JSON): ${JSON.stringify(ok)}`,
].join('\n\n')

const report = await agent(synthPrompt, { label: 'synthesize', phase: 'Synthesize' })

return { workstream, scope: scopeRaw, pillars_run: ok.map(x => x.pillar.key), report }
