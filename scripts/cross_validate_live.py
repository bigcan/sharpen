#!/usr/bin/env python3
"""XVal Layer 2: Multi-channel cross-validation for live trading.

Collects strategy state from all monitoring channels simultaneously and
compares them pairwise to detect silent divergence.

Channels:
  1. Health JSON   — docker exec cat /tmp/health_status.json
  2. Prometheus    — HTTP API query (requires SSH tunnel or Docker-internal access)
  3. WandB         — Python API (run summary for live/paper runs)

Broker position is read from Prometheus (finrl_broker_position gauge emitted
by XVal Layer 1 in-engine reconciliation), so no additional broker API call.

Usage:
    # Human-readable report (requires SSH tunnel for Prometheus)
    python scripts/cross_validate_live.py

    # JSON output (for piping to watchdog or other tools)
    python scripts/cross_validate_live.py --json

    # Custom Prometheus URL (e.g., inside Docker network)
    python scripts/cross_validate_live.py --prometheus-url http://prometheus:9090

    # Specific containers only
    python scripts/cross_validate_live.py --containers gmgp1-btc gmgp2-xauusd
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ChannelData:
    """Data collected from a single monitoring channel."""
    position: float | None = None
    portfolio_value: float | None = None
    drawdown_pct: float | None = None
    bar_count: int | None = None
    broker_connected: bool | None = None
    timestamp: float | None = None  # Unix epoch of last update
    # Broker-reported (from Prometheus XVal gauges)
    broker_position: float | None = None
    position_divergence: float | None = None
    pv_divergence_pct: float | None = None
    error: str | None = None


@dataclass
class Comparison:
    """Result of comparing one field between two channels."""
    pair: str  # e.g. "health_vs_prometheus"
    field_name: str
    value_a: float | None
    value_b: float | None
    delta: float
    verdict: str  # OK, WARN, CRIT


@dataclass
class StrategyReport:
    """Cross-validation report for one strategy."""
    strategy: str
    channels: dict[str, ChannelData] = field(default_factory=dict)
    comparisons: list[Comparison] = field(default_factory=list)
    verdict: str = "OK"


@dataclass
class XValReport:
    """Full cross-validation report."""
    timestamp: str = ""
    strategies: dict[str, StrategyReport] = field(default_factory=dict)
    verdict: str = "OK"


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

# (field, warn_threshold, crit_threshold)
# For same-source channels (health vs prometheus bar-end): tight tolerance
# For network channels (WandB): looser tolerance
THRESHOLDS = {
    "health_vs_prometheus": {
        "position": (0.01, 0.05),
        "portfolio_value_pct": (0.01, 0.03),  # percentage-based
        "bar_count": (1, 3),
    },
    "health_vs_wandb": {
        "position": (0.02, 0.05),
        "portfolio_value_pct": (0.02, 0.05),
        "bar_count": (5, 10),
    },
    "prometheus_vs_wandb": {
        "position": (0.02, 0.05),
        "portfolio_value_pct": (0.02, 0.05),
        "bar_count": (5, 10),
    },
}


# ---------------------------------------------------------------------------
# Channel collectors
# ---------------------------------------------------------------------------

def collect_health_json(
    container: str, docker_context: str = "finrl-desktop",
) -> ChannelData:
    """Read health_status.json from a container via docker exec."""
    cmd = [
        "docker", "--context", docker_context, "exec", container,
        "sh", "-c", "cat /tmp/health_status.json",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        if result.returncode != 0:
            return ChannelData(error=f"docker exec failed: {result.stderr.strip()}")
        data = json.loads(result.stdout.strip())
        return ChannelData(
            position=data.get("position"),
            portfolio_value=data.get("portfolio_value"),
            drawdown_pct=data.get("drawdown_pct"),
            bar_count=data.get("bar_count"),
            broker_connected=data.get("broker_connected"),
            timestamp=data.get("timestamp"),
        )
    except subprocess.TimeoutExpired:
        return ChannelData(error="docker exec timeout (15s)")
    except (json.JSONDecodeError, Exception) as e:
        return ChannelData(error=str(e))


def _prom_query(prometheus_url: str, query: str) -> list[dict]:
    """Execute an instant PromQL query and return result vector."""
    import requests

    url = f"{prometheus_url}/api/v1/query"
    try:
        resp = requests.get(url, params={"query": query}, timeout=5)
        resp.raise_for_status()
        body = resp.json()
        if body.get("status") != "success":
            return []
        return body.get("data", {}).get("result", [])
    except Exception as e:
        logger.debug(f"Prometheus query failed: {e}")
        return []


def collect_prometheus(
    strategy: str, prometheus_url: str = "http://localhost:9090",
) -> ChannelData:
    """Query Prometheus for a strategy's current metrics."""
    metrics = {}
    gauge_names = [
        "finrl_position", "finrl_portfolio_value", "finrl_drawdown_pct",
        "finrl_bar_count", "finrl_broker_connected", "finrl_last_bar_timestamp",
        "finrl_broker_position", "finrl_position_divergence_abs",
        "finrl_pv_divergence_pct",
    ]
    for metric in gauge_names:
        results = _prom_query(
            prometheus_url,
            f'{metric}{{strategy="{strategy}"}}',
        )
        if results:
            metrics[metric] = float(results[0]["value"][1])

    if not metrics:
        # Check if scrape target is up
        up_results = _prom_query(prometheus_url, f'up{{strategy="{strategy}"}}')
        if up_results and float(up_results[0]["value"][1]) == 0:
            return ChannelData(error="Prometheus scrape DOWN (up=0)")
        return ChannelData(error="No Prometheus data found for strategy")

    return ChannelData(
        position=metrics.get("finrl_position"),
        portfolio_value=metrics.get("finrl_portfolio_value"),
        drawdown_pct=metrics.get("finrl_drawdown_pct"),
        bar_count=int(metrics["finrl_bar_count"]) if "finrl_bar_count" in metrics else None,
        broker_connected=metrics.get("finrl_broker_connected", 0) > 0.5,
        timestamp=metrics.get("finrl_last_bar_timestamp"),
        broker_position=metrics.get("finrl_broker_position"),
        position_divergence=metrics.get("finrl_position_divergence_abs"),
        pv_divergence_pct=metrics.get("finrl_pv_divergence_pct"),
    )


def collect_wandb(
    wandb_project: str = "bigcan-chiwin-technology/FinRL-Pro-DS",
) -> dict[str, ChannelData]:
    """Query WandB for all running live/paper runs. Returns {strategy: data}."""
    results: dict[str, ChannelData] = {}
    try:
        import wandb
        api = wandb.Api()
        runs = api.runs(wandb_project, filters={
            "state": "running",
            "tags": {"$in": ["live", "paper"]},
        })
        for run in runs:
            s = run.summary._json_dict
            # Extract strategy name from tags
            strategy = None
            for tag in run.tags:
                if tag in ("live", "paper"):
                    continue
                # Strategy tags: gmgp1-gold, gmgp1-btc, gmgp2-xauusd, etc.
                if tag.startswith(("gmgp1-", "gmgp2-", "sg1-", "funding-", "sync-", "velotrade-")):
                    strategy = tag
                    break
            if strategy is None:
                # Fallback: derive from run name
                strategy = run.name.replace("live-", "").replace("paper-", "")
                strategy = strategy.rsplit("-", 1)[0] if "-" in strategy else strategy

            results[strategy] = ChannelData(
                position=s.get("position"),
                portfolio_value=s.get("portfolio_value"),
                drawdown_pct=s.get("drawdown_pct"),
                bar_count=s.get("bar"),
                timestamp=s.get("_timestamp"),
            )
    except ImportError:
        logger.warning("wandb not installed — skipping WandB channel")
    except Exception as e:
        logger.warning(f"WandB query failed: {e}")

    return results


# ---------------------------------------------------------------------------
# Reconciliation engine
# ---------------------------------------------------------------------------

def _compare_field(
    pair: str,
    field_name: str,
    val_a: float | None,
    val_b: float | None,
    thresholds: tuple[float, float],
    is_pct: bool = False,
    base_value: float | None = None,
) -> Comparison:
    """Compare a single field between two channels."""
    if val_a is None or val_b is None:
        return Comparison(pair, field_name, val_a, val_b, delta=0.0, verdict="SKIP")

    if is_pct and base_value and base_value > 0:
        delta = abs(val_a - val_b) / base_value
    else:
        delta = abs(val_a - val_b)

    warn, crit = thresholds
    if delta > crit:
        verdict = "CRIT"
    elif delta > warn:
        verdict = "WARN"
    else:
        verdict = "OK"

    return Comparison(pair, field_name, val_a, val_b, delta, verdict)


def reconcile_strategy(
    strategy: str,
    health: ChannelData | None,
    prom: ChannelData | None,
    wandb_data: ChannelData | None,
    bar_interval_seconds: int = 900,
) -> StrategyReport:
    """Compare all channels for a single strategy."""
    report = StrategyReport(strategy=strategy)

    if health:
        report.channels["health"] = health
    if prom:
        report.channels["prometheus"] = prom
    if wandb_data:
        report.channels["wandb"] = wandb_data

    # --- Health vs Prometheus ---
    if health and prom and not health.error and not prom.error:
        t = THRESHOLDS["health_vs_prometheus"]
        report.comparisons.append(_compare_field(
            "health_vs_prometheus", "position",
            health.position, prom.position, t["position"],
        ))
        report.comparisons.append(_compare_field(
            "health_vs_prometheus", "portfolio_value",
            health.portfolio_value, prom.portfolio_value,
            t["portfolio_value_pct"], is_pct=True, base_value=health.portfolio_value,
        ))
        report.comparisons.append(_compare_field(
            "health_vs_prometheus", "bar_count",
            health.bar_count, prom.bar_count, t["bar_count"],
        ))

    # --- Health vs WandB ---
    if health and wandb_data and not health.error and not wandb_data.error:
        t = THRESHOLDS["health_vs_wandb"]
        report.comparisons.append(_compare_field(
            "health_vs_wandb", "position",
            health.position, wandb_data.position, t["position"],
        ))
        report.comparisons.append(_compare_field(
            "health_vs_wandb", "portfolio_value",
            health.portfolio_value, wandb_data.portfolio_value,
            t["portfolio_value_pct"], is_pct=True, base_value=health.portfolio_value,
        ))
        report.comparisons.append(_compare_field(
            "health_vs_wandb", "bar_count",
            health.bar_count, wandb_data.bar_count, t["bar_count"],
        ))

    # --- Prometheus vs WandB ---
    if prom and wandb_data and not prom.error and not wandb_data.error:
        t = THRESHOLDS["prometheus_vs_wandb"]
        report.comparisons.append(_compare_field(
            "prometheus_vs_wandb", "position",
            prom.position, wandb_data.position, t["position"],
        ))
        report.comparisons.append(_compare_field(
            "prometheus_vs_wandb", "portfolio_value",
            prom.portfolio_value, wandb_data.portfolio_value,
            t["portfolio_value_pct"], is_pct=True, base_value=prom.portfolio_value,
        ))
        report.comparisons.append(_compare_field(
            "prometheus_vs_wandb", "bar_count",
            prom.bar_count, wandb_data.bar_count, t["bar_count"],
        ))

    # --- Broker divergence (from Prometheus XVal gauges) ---
    if prom and not prom.error and prom.position_divergence is not None:
        report.comparisons.append(Comparison(
            pair="engine_vs_broker",
            field_name="position_divergence",
            value_a=prom.position_divergence,
            value_b=0.0,
            delta=prom.position_divergence,
            verdict=(
                "CRIT" if prom.position_divergence > 0.15
                else "WARN" if prom.position_divergence > 0.05
                else "OK"
            ),
        ))

    # --- Freshness checks ---
    now = time.time()
    stale_warn = bar_interval_seconds * 1.5
    stale_crit = bar_interval_seconds * 2.5

    for name, channel in [("health", health), ("wandb", wandb_data)]:
        if channel and not channel.error and channel.timestamp:
            age = now - channel.timestamp
            verdict = (
                "CRIT" if age > stale_crit
                else "WARN" if age > stale_warn
                else "OK"
            )
            report.comparisons.append(Comparison(
                pair=f"{name}_freshness",
                field_name="age_seconds",
                value_a=age,
                value_b=0.0,
                delta=age,
                verdict=verdict,
            ))

    # --- Prometheus scrape health ---
    if prom and prom.error and "scrape DOWN" in (prom.error or ""):
        report.comparisons.append(Comparison(
            pair="prometheus_scrape",
            field_name="up",
            value_a=0.0,
            value_b=1.0,
            delta=1.0,
            verdict="CRIT",
        ))

    # --- Channel errors ---
    for name, channel in [("health", health), ("prometheus", prom), ("wandb", wandb_data)]:
        if channel and channel.error:
            report.comparisons.append(Comparison(
                pair=f"{name}_error",
                field_name="availability",
                value_a=None,
                value_b=None,
                delta=0.0,
                verdict="WARN",
            ))

    # Overall verdict = worst of all comparisons
    verdicts = [c.verdict for c in report.comparisons if c.verdict != "SKIP"]
    if "CRIT" in verdicts:
        report.verdict = "CRIT"
    elif "WARN" in verdicts:
        report.verdict = "WARN"
    else:
        report.verdict = "OK"

    return report


# ---------------------------------------------------------------------------
# Cross-Validator (importable class)
# ---------------------------------------------------------------------------

class CrossValidator:
    """Multi-channel cross-validation for live trading strategies."""

    # Known strategy containers (from docker-compose.yaml)
    KNOWN_CONTAINERS = [
        "gmgp1-gold", "sg1-gold", "gmgp1-btc", "funding-arb",
        "sync-1h", "gmgp2-xauusd", "velotrade-btc",
    ]

    def __init__(
        self,
        docker_context: str = "finrl-desktop",
        prometheus_url: str = "http://localhost:9090",
        wandb_project: str = "bigcan-chiwin-technology/FinRL-Pro-DS",
        bar_interval_seconds: int = 900,
    ) -> None:
        self.docker_context = docker_context
        self.prometheus_url = prometheus_url
        self.wandb_project = wandb_project
        self.bar_interval_seconds = bar_interval_seconds

    def discover_containers(self) -> list[str]:
        """Find running strategy containers on remote desktop."""
        cmd = [
            "docker", "--context", self.docker_context, "ps",
            "--filter", "label=finrl.strategy",
            "--format", "{{.Names}}",
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
            if result.returncode == 0:
                names = [n.strip() for n in result.stdout.strip().split("\n") if n.strip()]
                return names
        except Exception as e:
            logger.warning(f"Container discovery failed: {e}")
        return []

    def collect_all(
        self, containers: list[str],
    ) -> dict[str, dict[str, ChannelData]]:
        """Collect data from all channels. Returns {strategy: {channel: data}}."""
        result: dict[str, dict[str, ChannelData]] = {}

        # Health JSON — per container
        for container in containers:
            health = collect_health_json(container, self.docker_context)
            strategy = container  # container name = strategy name
            result.setdefault(strategy, {})["health"] = health

        # Prometheus — per strategy
        for container in containers:
            prom = collect_prometheus(container, self.prometheus_url)
            result.setdefault(container, {})["prometheus"] = prom

        # WandB — batch query
        wandb_data = collect_wandb(self.wandb_project)
        for strategy, data in wandb_data.items():
            result.setdefault(strategy, {})["wandb"] = data

        return result

    def reconcile(
        self, data: dict[str, dict[str, ChannelData]],
    ) -> XValReport:
        """Compare all channels and produce verdicts."""
        report = XValReport(
            timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        )

        for strategy, channels in data.items():
            sr = reconcile_strategy(
                strategy,
                channels.get("health"),
                channels.get("prometheus"),
                channels.get("wandb"),
                self.bar_interval_seconds,
            )
            report.strategies[strategy] = sr

        # Overall = worst
        verdicts = [s.verdict for s in report.strategies.values()]
        if "CRIT" in verdicts:
            report.verdict = "CRIT"
        elif "WARN" in verdicts:
            report.verdict = "WARN"
        else:
            report.verdict = "OK"

        return report

    def run(self, containers: list[str] | None = None) -> XValReport:
        """Full pipeline: discover, collect, reconcile."""
        if containers is None:
            containers = self.discover_containers()
        if not containers:
            report = XValReport(
                timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            )
            report.verdict = "WARN"
            return report

        data = self.collect_all(containers)
        return self.reconcile(data)


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def _fmt_val(val, fmt=".4f") -> str:
    if val is None:
        return "—"
    if isinstance(val, bool):
        return "true" if val else "false"
    if isinstance(val, int):
        return str(val)
    try:
        return f"{val:{fmt}}"
    except (ValueError, TypeError):
        return str(val)


def _fmt_pv(val) -> str:
    if val is None:
        return "—"
    return f"${val:,.2f}"


def _fmt_age(ts: float | None) -> str:
    if ts is None:
        return "—"
    age = time.time() - ts
    if age < 60:
        return f"{age:.0f}s ago"
    if age < 3600:
        return f"{age / 60:.0f}m ago"
    return f"{age / 3600:.1f}h ago"


def print_report(report: XValReport) -> None:
    """Print human-readable cross-validation report."""
    print(f"\n## Cross-Validation Report — {report.timestamp}")
    print()

    if not report.strategies:
        print("  No running strategy containers found.")
        print()
        return

    for strategy, sr in sorted(report.strategies.items()):
        health = sr.channels.get("health")
        prom = sr.channels.get("prometheus")
        wandb_ch = sr.channels.get("wandb")

        print(f"### {strategy}  [{sr.verdict}]")

        # Channel data table
        print(f"  {'Field':<16} {'Health JSON':>14} {'Prometheus':>14} {'WandB':>14}")
        print(f"  {'─' * 16} {'─' * 14} {'─' * 14} {'─' * 14}")

        h, p, w = health or ChannelData(), prom or ChannelData(), wandb_ch or ChannelData()

        print(f"  {'position':<16} {_fmt_val(h.position):>14} {_fmt_val(p.position):>14} {_fmt_val(w.position):>14}")
        print(f"  {'portfolio_val':<16} {_fmt_pv(h.portfolio_value):>14} {_fmt_pv(p.portfolio_value):>14} {_fmt_pv(w.portfolio_value):>14}")
        print(f"  {'drawdown_pct':<16} {_fmt_val(h.drawdown_pct):>14} {_fmt_val(p.drawdown_pct):>14} {_fmt_val(w.drawdown_pct):>14}")
        print(f"  {'bar_count':<16} {_fmt_val(h.bar_count, 'd'):>14} {_fmt_val(p.bar_count, 'd'):>14} {_fmt_val(w.bar_count, 'd'):>14}")
        print(f"  {'freshness':<16} {_fmt_age(h.timestamp):>14} {_fmt_age(p.timestamp):>14} {_fmt_age(w.timestamp):>14}")

        if prom and prom.broker_position is not None:
            print(f"  {'broker_pos':<16} {'—':>14} {_fmt_val(prom.broker_position):>14} {'—':>14}")
            print(f"  {'pos_divergence':<16} {'—':>14} {_fmt_val(prom.position_divergence):>14} {'—':>14}")
            print(f"  {'pv_divg_pct':<16} {'—':>14} {_fmt_val(prom.pv_divergence_pct):>14} {'—':>14}")

        # Channel errors
        for name in ("health", "prometheus", "wandb"):
            ch = sr.channels.get(name)
            if ch and ch.error:
                print(f"  [{name} ERROR] {ch.error}")

        # Comparisons with issues
        issues = [c for c in sr.comparisons if c.verdict in ("WARN", "CRIT")]
        if issues:
            print()
            print(f"  {'Comparison':<30} {'Delta':>10} {'Verdict':>8}")
            print(f"  {'─' * 30} {'─' * 10} {'─' * 8}")
            for c in issues:
                print(f"  {c.pair + '/' + c.field_name:<30} {c.delta:>10.4f} {c.verdict:>8}")
        print()

    print(f"### Overall Verdict: {report.verdict}")
    print()


def _channel_to_dict(ch: ChannelData) -> dict:
    return {
        "position": ch.position,
        "portfolio_value": ch.portfolio_value,
        "drawdown_pct": ch.drawdown_pct,
        "bar_count": ch.bar_count,
        "broker_connected": ch.broker_connected,
        "timestamp": ch.timestamp,
        "broker_position": ch.broker_position,
        "position_divergence": ch.position_divergence,
        "pv_divergence_pct": ch.pv_divergence_pct,
        "error": ch.error,
    }


def report_to_json(report: XValReport) -> dict:
    """Convert report to JSON-serializable dict."""
    strategies = {}
    for name, sr in sorted(report.strategies.items()):
        strategies[name] = {
            "channels": {
                ch_name: _channel_to_dict(ch)
                for ch_name, ch in sr.channels.items()
            },
            "comparisons": [
                {
                    "pair": c.pair,
                    "field": c.field_name,
                    "value_a": c.value_a,
                    "value_b": c.value_b,
                    "delta": c.delta,
                    "verdict": c.verdict,
                }
                for c in sr.comparisons
            ],
            "verdict": sr.verdict,
        }
    return {
        "timestamp": report.timestamp,
        "strategies": strategies,
        "verdict": report.verdict,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="XVal: Multi-channel cross-validation for live trading",
    )
    parser.add_argument(
        "--context", default="finrl-desktop",
        help="Docker context (default: finrl-desktop)",
    )
    parser.add_argument(
        "--prometheus-url", default="http://localhost:9090",
        help="Prometheus API URL (default: http://localhost:9090, requires SSH tunnel)",
    )
    parser.add_argument(
        "--wandb-project", default="bigcan-chiwin-technology/FinRL-Pro-DS",
        help="WandB project path",
    )
    parser.add_argument(
        "--bar-interval", type=int, default=900,
        help="Bar interval in seconds for freshness checks (default: 900 = 15min)",
    )
    parser.add_argument(
        "--containers", nargs="+", default=None,
        help="Specific containers to check (default: auto-discover)",
    )
    parser.add_argument(
        "--json", action="store_true", dest="as_json",
        help="Output as JSON",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.WARNING,
        format="%(levelname)s: %(message)s",
    )

    validator = CrossValidator(
        docker_context=args.context,
        prometheus_url=args.prometheus_url,
        wandb_project=args.wandb_project,
        bar_interval_seconds=args.bar_interval,
    )

    report = validator.run(containers=args.containers)

    if args.as_json:
        print(json.dumps(report_to_json(report), indent=2))
    else:
        print_report(report)

    # Exit code: 0=OK, 1=WARN, 2=CRIT
    if report.verdict == "CRIT":
        sys.exit(2)
    elif report.verdict == "WARN":
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
