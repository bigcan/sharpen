"""Run a GENUINE Crucible LLM-proposer tick using the local ``claude`` CLI (Claude Code
subscription OAuth) as the proposer transport — instead of the metered console API key.

Motivation (S553-cont-118): the console ``ANTHROPIC_API_KEY`` path returns HTTP 400
"credit balance too low", so ``taiwan_tick.ps1 -Llm`` fails closed. The Claude Code
subscription is a separate billing path that IS funded, and ``claude -p`` (headless) uses
it. This driver routes the LLM proposer through ``claude -p`` so continuous, mechanism-
diverse discovery can run at zero console-API cost.

WHY THIS IS STILL GENUINE DISCOVERY, not a hack around the science
-----------------------------------------------------------------
* Anti-oracle moat (CRU-2 / spec Part A2) preserved BY CONSTRUCTION and VERIFIED. The child
  ``claude`` process is invoked with ``--setting-sources ""`` (drops user/project/local
  settings → NO SessionStart auto-memory hook) and cwd=<tempdir> (no project CLAUDE.md, no
  FinRL auto-memory). A probe confirmed it reports has_context:false for Crucible/FinRL/
  verdicts. Its ONLY inputs are the code-generated system prompt (``_PROMPT_TEMPLATE``) and
  the code-generated ProposalContext user message (``_render_context``) — byte-for-byte what
  the real network path sends. It never sees a score/verdict/DSR.
* NO Crucible module is modified. We monkeypatch only the injectable ``transport`` seam that
  ``llm_proposer.py`` already exposes for exactly this (its tests use it to avoid network/key).
* Provenance honest: ``--llm-model`` is stamped ``claude-cli-local-<model>``; the prompt-
  template hash inside ``model_id`` is unchanged because ``_PROMPT_TEMPLATE`` is passed verbatim.
* Determinism: a live LLM call is non-bit-reproducible on either path (spec accepts this; the
  reproduce contract is "replay recorded specs", not "re-call"). The CLI path is no different
  in kind from the console-API path.

Cost/cadence note: each accepted proposal that clears dedup is charged to the LORD++ FDR
account and run through the full T0–T5 deflation funnel, so keep the per-tick batch modest
(default 16 ≈ ~3 min; the CLI produces ~10s/proposal). Big single-shot batches (128) both
time out and spend FDR wealth wastefully — diversity across ticks beats volume per tick.

Usage (run from the repo root):
    python scripts/research/crucible_taiwan_tick_cli.py
    python scripts/research/crucible_taiwan_tick_cli.py --max-proposals 24 --model sonnet
    python scripts/research/crucible_taiwan_tick_cli.py -- --no-lockbox   # extras pass through
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from sharpen.crucible.agentic import llm_proposer as _llm  # noqa: E402

_CLI = shutil.which("claude") or r"~\.local\bin\claude"
_CHILD_CWD = tempfile.gettempdir()  # neutral dir: no project CLAUDE.md / FinRL auto-memory


def _extract_json_obj(text: str) -> dict:
    """Pull the single JSON object out of the CLI's text result (tolerates ```json fences
    and incidental prose around it)."""
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else t[3:]
        if t.rstrip().endswith("```"):
            t = t.rstrip()[:-3]
    start, end = t.find("{"), t.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("no JSON object found in CLI output")
    return json.loads(t[start:end + 1])


def _make_transport(model: str, timeout: int):
    """Build a transport (url, headers, body) -> Messages-API-shaped dict that routes the
    request through the local ``claude`` CLI on subscription OAuth. Ignores url/headers/key."""

    def claude_cli_transport(url: str, headers: dict, body: dict) -> dict:
        system = body.get("system", "")
        user = body["messages"][0]["content"]
        schema = body["tools"][0]["input_schema"]
        instruction = (
            "\n\nRespond with ONLY a single JSON object and nothing else (no prose, no markdown "
            "fences) that matches this schema exactly -- a top-level \"proposals\" key whose value "
            "is an array of objects each having keys: name, hypothesis, expected_sign, "
            "candidate_type, formula, economic_rationale:\n" + json.dumps(schema)
        )
        # subscription OAuth (host-provided) — must NOT inherit the depleted console key.
        child_env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
        cmd = [
            _CLI, "-p", "--output-format", "json", "--model", model,
            "--setting-sources", "",     # no user/project/local settings -> no memory hook
            "--system-prompt", system,   # ONLY the fixed proposer template
            user + instruction,
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                              env=child_env, cwd=_CHILD_CWD, timeout=timeout)
        if proc.returncode != 0:
            raise RuntimeError(f"claude CLI exit {proc.returncode}: {(proc.stderr or '')[:500]}")
        envelope = json.loads(proc.stdout)
        if envelope.get("is_error"):
            raise RuntimeError(f"claude CLI reported error: {envelope.get('result')!r}")
        obj = _extract_json_obj(envelope.get("result", ""))
        usage = envelope.get("usage", {}) or {}
        n = len(obj.get("proposals", []))
        print(f"[transport] claude CLI returned {n} proposals "
              f"(in={usage.get('input_tokens', 0)} out={usage.get('output_tokens', 0)} tok)",
              file=sys.stderr)
        return {
            "content": [{"type": "tool_use", "name": "propose_hypotheses",
                         "input": {"proposals": obj.get("proposals", [])}}],
            "usage": {"input_tokens": int(usage.get("input_tokens", 0)),
                      "output_tokens": int(usage.get("output_tokens", 0))},
        }

    return claude_cli_transport


_BLIND_PROBE = (
    "Do you have any preloaded context, memory, or CLAUDE.md about: a project named Crucible or "
    "FinRL, TSMOM/BAB strategies, Taiwan TWSE/TAIFEX signals, or any trading-strategy backtest "
    'verdicts or Sharpe ratios? Answer strictly as JSON {"has_context": true|false, "items": '
    '[short strings]}. JSON only.'
)


def _looks_true(v) -> bool:
    """Conservative truthiness for the probe's has_context field (tolerates a stringy "true")."""
    if isinstance(v, str):
        return v.strip().lower() in ("true", "1", "yes")
    return bool(v)


def _pick_resolved_model(envelope: dict, alias: str) -> str | None:
    """The concrete model id that actually served the request, from the CLI envelope's
    ``modelUsage`` map (L1 provenance: ``--model sonnet`` resolves to e.g. ``claude-sonnet-4-6``,
    and the CLI also bills an auxiliary ``haiku`` step — so prefer the key matching the requested
    alias family, and among ties the one that generated the most output tokens)."""
    usage = envelope.get("modelUsage") or {}
    if not usage:
        return None
    base = alias.split("-", 2)[1] if alias.startswith("claude-") else alias
    matches = [k for k in usage if base.lower() in k.lower()]
    pool = matches or list(usage)
    return max(pool, key=lambda k: (usage.get(k) or {}).get("outputTokens", 0))


def _cli_version() -> str | None:
    """Best-effort ``claude`` CLI version for the provenance stamp ("2.1.183 (Claude Code)" ->
    "2.1.183"). Never fatal — provenance degrades to model-only if this can't be read."""
    try:
        out = subprocess.run([_CLI, "--version"], capture_output=True, text=True,
                             encoding="utf-8", timeout=30).stdout.strip()
        return out.split()[0] if out else None
    except Exception:  # noqa: BLE001 - version is a nice-to-have, not a gate
        return None


def _assert_blind(model: str, timeout: int = 120) -> str | None:
    """Fail-closed moat guard (CRU-2): before mining, confirm the CLI child — under the SAME
    isolation flags the proposer transport uses (``--setting-sources ""`` + neutral cwd) — cannot
    see any project verdict/memory context. Abort the whole run (SystemExit) if it can, or if the
    probe cannot be evaluated, rather than silently mining with a contaminated proposer. This wires
    the blindness guarantee that was previously only asserted in a comment + a one-time manual probe
    (audit finding M1), so a future CLI change to --setting-sources semantics fails loudly here.

    Bonus (L1): this call already resolves the concrete model, so it returns the resolved model id
    parsed from the same envelope for the provenance stamp (``None`` if it can't be determined)."""
    child_env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
    cmd = [
        _CLI, "-p", "--output-format", "json", "--model", model,
        "--setting-sources", "",
        "--system-prompt", "You output only JSON.",
        _BLIND_PROBE,
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                              env=child_env, cwd=_CHILD_CWD, timeout=timeout)
        if proc.returncode != 0:
            raise RuntimeError(f"CLI exit {proc.returncode}: {(proc.stderr or '')[:300]}")
        envelope = json.loads(proc.stdout)
        if envelope.get("is_error"):
            raise RuntimeError(f"CLI reported error: {envelope.get('result')!r}")
        obj = _extract_json_obj(envelope.get("result", ""))
    except Exception as exc:  # noqa: BLE001 - any probe failure must block, never silently pass
        raise SystemExit(f"[blindness probe] ABORT — cannot verify the moat ({exc}). "
                         "Refusing to mine with an unverified proposer (CRU-2).") from exc

    items = obj.get("items") or []
    if _looks_true(obj.get("has_context")) or items:
        raise SystemExit(f"[blindness probe] ABORT — MOAT VIOLATION (CRU-2): the CLI child can see "
                         f"project context (has_context={obj.get('has_context')!r}, items={items!r}). "
                         "The proposer is NOT blind; refusing to mine.")
    print("[driver] blindness probe PASS: child reports has_context=false, items=[]", file=sys.stderr)
    return _pick_resolved_model(envelope, model)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Run a genuine Crucible LLM-proposer tick via the local claude CLI "
                    "(subscription OAuth), bypassing the metered console API key. Blindness "
                    "(CRU-2) preserved via --setting-sources '' + neutral cwd. See module "
                    "docstring for the full rationale.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-proposals", type=int, default=16,
                    help="per-tick proposal cap (default 16; keep modest -- each fresh one is "
                         "charged to the FDR account and fully deflated)")
    ap.add_argument("--model", default="sonnet",
                    help="claude CLI --model alias/id for the proposer (default sonnet)")
    ap.add_argument("--nights", type=int, default=1, help="unattended ticks to run")
    ap.add_argument("--config", default="configs/taiwan_signal_eval.gates.yaml")
    ap.add_argument("--out", default="results/crucible_orchestrator/taiwan_v2",
                    help="orchestrator out-dir — keep on taiwan_v2 so FDR wealth + dedup stay "
                         "continuous across every tick (library + CLI-LLM)")
    ap.add_argument("--timeout", type=int, default=600,
                    help="per-CLI-call timeout in seconds (default 600)")
    ap.add_argument("extra", nargs="*",
                    help="extra flags forwarded verbatim to crucible_orchestrator.py "
                         "(put after a literal --), e.g. -- --no-lockbox")
    args = ap.parse_args()

    # Install the seam + satisfy the fail-closed guards (the transport ignores the key value).
    _llm._default_transport = _make_transport(args.model, args.timeout)
    os.environ["ANTHROPIC_API_KEY"] = os.environ.get("ANTHROPIC_API_KEY") or "oauth-via-claude-cli"

    print(f"[driver] transport=claude CLI  model={args.model}  child_cwd={_CHILD_CWD}",
          file=sys.stderr)
    # M1: runtime moat guard — abort before mining if the child can see project verdicts/memory.
    # It also returns the concrete resolved model id for the L1 provenance stamp.
    resolved = _assert_blind(args.model)
    version = _cli_version()
    # agent_model_id becomes llm-<stamp>-prompt<hash>: record CLI version + the CONCRETE model that
    # served the proposals (e.g. claude-sonnet-4-6), not just the rolling "sonnet" alias. Degrade to
    # local-<alias> only if the resolve failed.
    stamp = "-".join(["claude-cli", *([version] if version else []),
                      resolved or f"local-{args.model}"])
    print(f"[driver] provenance agent_model_id = llm-{stamp}-prompt<template-hash>", file=sys.stderr)

    from scripts.research import crucible_orchestrator as orch

    sys.argv = [
        "crucible_orchestrator.py",
        "--config", args.config,
        "--mode", "real", "--force", "--nights", str(args.nights),
        "--max-proposals", str(args.max_proposals),
        "--out", args.out,
        "--proposer", "llm",
        "--llm-model", stamp,
        *args.extra,
    ]
    return orch.main()


if __name__ == "__main__":
    raise SystemExit(main())
