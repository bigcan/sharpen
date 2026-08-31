"""Leak-safe, per-ticker Chronos-2 LoRA fine-tune driver for the PRISM daily re-eval.

Produces the **arm-C** models (one per asset) for the base-vs-v2-vs-fresh-fine-tune
comparison. Two reasons this is a NEW project-side script rather than an edit to the
SAFFS `scripts/finetune_chronos.py`:

  1. SAFFS is outside this repo's editable boundary (CLAUDE.md: only
     sharpen/scripts/configs/tests/docs).
  2. The SAFFS script is BTC-hardcoded and, more importantly, drives the model via a
     hand-rolled HF `Trainer` whose `model(context=, future_target=)` call predates the
     installed `chronos-forecasting` (the same signature drift that broke
     `predict_quantiles`). We instead use the library's OFFICIAL, version-matched
     entrypoint `Chronos2Pipeline.fit(..., finetune_mode="lora")`.

LEAK DISCIPLINE (the whole point):
  * `--train-end` HARD-truncates the series before any returns are computed, so nothing
    at/after the OOS boundary can enter training or validation. The eval
    (`prism_predictive_eval.py --oos-start ...`) then reads strictly post-cutoff bars,
    which the fine-tune never saw. `prepare_training_logrets` is factored out and unit
    tested (`tests/prism_research/test_finetune_leak_guard.py`).
  * Temporal train/val split, NO shuffling (LEAK-1).

Output: a FULL, merged model dir (config.json + model.safetensors), identical in shape
to `chronos_finetuned_v2`, so the provider's `ChronosExpert.load` loads it via the exact
v2 code path (`BaseChronosPipeline.from_pretrained(local_path)`) with no PEFT-adapter /
base-model resolution at inference time.

Usage (run with the SAFFS venv python, which has chronos/peft/torch):
    C:/FinRL/SAFFS/.venv/Scripts/python.exe scripts/prism_research/finetune_chronos_pathA.py \
        --ticker btc --train-end 2022-12-31

    # all three, default output dirs under results/prism_research/models/:
    for t in btc gold eurusd; do
        <saffs-venv-python> scripts/prism_research/finetune_chronos_pathA.py --ticker $t --train-end 2022-12-31
    done

To use a produced model in a precompute pass, point the provider at it:
    PRISM_CHRONOS_FINETUNED=results/prism_research/models/chronos_ft_pathA_btc \
        <saffs-venv-python> scripts/prism_research/precompute_prism_pathA.py --assets btc ...
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# Reuse the exact fetch / OHLC-repair / asset-map used by the precompute, so the
# fine-tune sees the same cleaned price basis the eval features are built from.
from scripts.prism_research.precompute_prism_pathA import (  # noqa: E402
    ASSET_TICKERS,
    fetch_daily_ohlcv,
    repair_ohlc_consistency,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("prism_research.finetune_pathA")

DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "results" / "prism_research" / "models"


def prepare_training_logrets(
    ohlcv_df: pd.DataFrame,
    ticker: str,
    train_end: str,
    train_frac: float = 0.85,
    repair_ohlc: bool = True,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Tidy OHLCV (one ticker) -> (train_logret, val_logret, meta), leak-safe.

    LEAK GUARD: every row with ``timestamp > train_end`` is dropped BEFORE log-returns
    are computed, so no post-cutoff information can enter training/validation. The
    train/val split is temporal (no shuffle, LEAK-1). Returns float32 1-D arrays in
    log-return space (the same space Chronos forecasts and the eval scores).
    """
    df = ohlcv_df[ohlcv_df["ticker"] == ticker].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    if repair_ohlc:
        df, _ = repair_ohlc_consistency(df)

    cutoff = pd.Timestamp(train_end)
    df = df[df["timestamp"] <= cutoff].reset_index(drop=True)  # <-- the leak guard

    closes = df["close"].to_numpy(dtype=float)
    logret = np.diff(np.log(np.clip(closes, 1e-10, None))).astype(np.float32)

    n = len(logret)
    t1 = int(n * train_frac)
    train, val = logret[:t1], logret[t1:]
    meta = {
        "ticker": ticker,
        "n_bars_le_cutoff": int(len(df)),
        "n_logret": int(n),
        "n_train": int(len(train)),
        "n_val": int(len(val)),
        "first_ts": (df["timestamp"].iloc[0] if len(df) else None),
        "last_ts": (df["timestamp"].iloc[-1] if len(df) else None),
        "cutoff": cutoff,
    }
    return train, val, meta


def _load_ohlcv(args: argparse.Namespace, ticker: str) -> pd.DataFrame:
    """Return a tidy OHLCV frame for one ticker, via parquet or yfinance fetch."""
    if args.ohlcv_parquet:
        long_df = pd.read_parquet(args.ohlcv_parquet)
        long_df["timestamp"] = pd.to_datetime(long_df["timestamp"])
        sub = long_df[long_df["ticker"] == ticker].copy()
        if sub.empty:
            raise ValueError(f"ticker '{ticker}' not in {args.ohlcv_parquet}")
        logger.info("Loaded %s OHLCV from %s (%d rows)", ticker, args.ohlcv_parquet, len(sub))
        return sub
    return fetch_daily_ohlcv(ticker, ASSET_TICKERS[ticker], args.start, args.end)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Leak-safe per-ticker Chronos-2 LoRA fine-tune (arm C)")
    parser.add_argument("--ticker", required=True, choices=list(ASSET_TICKERS),
                        help="Asset to fine-tune on (one model per ticker)")
    parser.add_argument("--train-end", default="2022-12-31",
                        help="Hard cutoff (inclusive). Nothing after this enters training "
                             "— keeps the post-cutoff eval window out-of-sample.")
    parser.add_argument("--ohlcv-parquet", default=None,
                        help="Pre-built tidy OHLCV parquet [timestamp,ticker,open,high,low,"
                             "close,volume]; else fetch via yfinance.")
    parser.add_argument("--start", default="2015-01-01", help="yfinance fetch start")
    parser.add_argument("--end", default="2026-06-17",
                        help="yfinance fetch end (data is still hard-capped at --train-end)")
    parser.add_argument("--output", default=None,
                        help="Output model dir (default results/prism_research/models/"
                             "chronos_ft_pathA_<ticker>)")
    parser.add_argument("--base-model", default="amazon/chronos-2",
                        help="Base Chronos checkpoint to fine-tune from")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--prediction-length", type=int, default=10,
                        help="Forecast horizon to fine-tune for (matches v2 recipe)")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate (LoRA). Library default for LoRA is 1e-5; "
                             "1e-4 matches the existing v2 recipe.")
    parser.add_argument("--num-steps", type=int, default=1000, help="Fine-tune steps")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Number of (windowed) time-series per batch")
    parser.add_argument("--train-frac", type=float, default=0.85,
                        help="Temporal train fraction of the <=cutoff series (rest = val)")
    parser.add_argument("--no-val", action="store_true",
                        help="Skip validation/best-model selection (faster)")
    args = parser.parse_args()

    import torch
    from chronos import BaseChronosPipeline
    from chronos.chronos2 import Chronos2Pipeline

    ticker = args.ticker
    out_dir = Path(args.output) if args.output else DEFAULT_OUTPUT_ROOT / f"chronos_ft_pathA_{ticker}"
    out_dir.parent.mkdir(parents=True, exist_ok=True)

    # ---- 1. data (leak-safe) ----
    ohlcv = _load_ohlcv(args, ticker)
    train, val, meta = prepare_training_logrets(
        ohlcv, ticker, args.train_end, train_frac=args.train_frac, repair_ohlc=True)
    logger.info("LEAK-GUARD: %s training data ends %s (cutoff %s); "
                "n_logret=%d (train=%d val=%d)",
                ticker, meta["last_ts"], meta["cutoff"].date(),
                meta["n_logret"], meta["n_train"], meta["n_val"])
    min_needed = args.prediction_length + 1
    if meta["n_train"] < min_needed:
        logger.error("Not enough training data (%d) for prediction_length=%d. Aborting.",
                     meta["n_train"], args.prediction_length)
        sys.exit(1)

    # ---- 2. load base + LoRA fine-tune via the OFFICIAL chronos-2 API ----
    device = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    if device != args.device:
        logger.warning("CUDA unavailable — falling back to CPU (slower).")
    logger.info("Loading base %s on %s ...", args.base_model, device)
    base = BaseChronosPipeline.from_pretrained(args.base_model, device_map=device)
    if not hasattr(base, "fit"):
        logger.error("Loaded pipeline %s has no .fit() — not a Chronos-2 model.", type(base))
        sys.exit(1)

    use_val = (not args.no_val) and (meta["n_val"] > args.prediction_length)
    scratch = out_dir.parent / f"_trainer_scratch_{ticker}"
    logger.info("Fine-tuning (LoRA): steps=%d lr=%g batch=%d horizon=%d val=%s",
                args.num_steps, args.lr, args.batch_size, args.prediction_length, use_val)
    # Chronos-2 `fit` expects a 3-D tensor (n_series, n_variates, history_length). We
    # fine-tune on ONE univariate series (the full <=cutoff log-return history); the
    # dataset windows it internally into (context, future) training samples.
    train_in = train.reshape(1, 1, -1)
    val_in = val.reshape(1, 1, -1) if use_val else None
    finetuned = base.fit(
        inputs=train_in,
        prediction_length=args.prediction_length,
        validation_inputs=val_in,
        finetune_mode="lora",
        lora_config=None,            # use the library's validated default target modules
        learning_rate=args.lr,
        num_steps=args.num_steps,
        batch_size=args.batch_size,
        output_dir=str(scratch),
    )

    # ---- 3. merge LoRA into a FULL model so it loads via the v2 code path ----
    model = finetuned.model
    if hasattr(model, "merge_and_unload"):
        logger.info("Merging LoRA adapter into base weights (full model, like v2) ...")
        model = model.merge_and_unload()
    Chronos2Pipeline(model=model).save_pretrained(str(out_dir))
    logger.info("Saved merged fine-tuned model -> %s", out_dir)

    # ---- 4. verify it reloads exactly the way the provider will load it ----
    try:
        _ = BaseChronosPipeline.from_pretrained(str(out_dir), device_map="cpu")
        logger.info("Verified: model reloadable by BaseChronosPipeline.from_pretrained.")
    except Exception as e:  # noqa: BLE001 - surface any load incompatibility loudly
        logger.error("Reload check FAILED for %s: %s", out_dir, e)
        sys.exit(1)

    logger.info("Done. To use: PRISM_CHRONOS_FINETUNED=%s "
                "<saffs-venv-python> scripts/prism_research/precompute_prism_pathA.py "
                "--assets %s --output-dir results/prism_research/arm_ft3 --repair-ohlc",
                out_dir, ticker)


if __name__ == "__main__":
    main()
