"""Environment factory for HPO — shared between serial and distributed HPO.

Verbatim extraction from scripts/run_full_pipeline.py (lines 76-243).
"""
import functools
import logging
import os

import gymnasium as gym

from finrl_pro_ds.data.parquet_handler import ParquetDataHandler
from finrl_pro_ds.envs.deep_scalper_env import DeepScalperEnv
from finrl_pro_ds.envs.swing_scalper_env import SwingScalperEnv

logger = logging.getLogger("FinRL.HPO")


def make_env(config, start_date=None, end_date=None, shm_config=None, norm_cutoff_date=None):
    """Factory to create environment with real data.

    Args:
        norm_cutoff_date: FIX LEAK-1 — When set, rolling normalization statistics
            (z-scores, SMAs) are reset at this date boundary so that training
            data does not leak into val/test feature statistics.
    """
    data_config = config.get("data", {})
    file_path = data_config.get("file_path")
    ticker = data_config.get("ticker", "BTCUSDT")

    sd = start_date or data_config.get("train_start_date")
    ed = end_date or data_config.get("train_end_date")

    env_config = config.get("env", {})
    env_config["reward"] = config.get("env", {}).get("reward", {})
    # Forward network config so env can read micro_config.input_size for fev3 compat
    env_config["network"] = config.get("network", {})
    # Forward features config for include_spread, n_levels, asset_class
    env_config["features"] = config.get("features", {})

    # V6 swing MDP: binary direction-switching (Phase K)
    # V7 continuous swing MDP: SAC position control (GMGP1)
    mdp_version = env_config.get("mdp_version", "v5")

    # CMGP1 uses crypto data pipeline (ccxt), not local parquet — skip file_path check
    if mdp_version != "cmgp1":
        if not file_path or not os.path.exists(file_path):
            raise ValueError(f"Invalid data file path: {file_path}")
    if mdp_version in ("v8", "v9"):
        from finrl_pro_ds.envs.market_making_env import MarketMakingEnv
        features_cfg = config.get("features", {})
        handler_type = config.get("data", {}).get("handler_type", "mm")
        if handler_type == "lob_micro":
            from finrl_pro_ds.data.lob_micro_handler import LOBMicroDataHandler
            mm_handler = LOBMicroDataHandler(
                file_path=file_path,
                ticker=ticker,
                feature_config=features_cfg,
                start_date=sd,
                end_date=ed,
                norm_cutoff_date=norm_cutoff_date,
            )
        elif handler_type == "lob":
            from finrl_pro_ds.data.lob_data_handler import LOBDataHandler
            mm_handler = LOBDataHandler(
                file_path=file_path,
                ticker=ticker,
                feature_config=features_cfg,
                start_date=sd,
                end_date=ed,
                norm_cutoff_date=norm_cutoff_date,
            )
        else:
            from finrl_pro_ds.data.mm_data_handler import MMDataHandler
            mm_handler = MMDataHandler(
                file_path=file_path,
                ticker=ticker,
                feature_config=features_cfg,
                start_date=sd,
                end_date=ed,
                norm_cutoff_date=norm_cutoff_date,
            )
        return MarketMakingEnv(config=env_config, data_handler=mm_handler)

    if mdp_version == "cmgp1":
        from finrl_pro_ds.crypto.data.multiscale_crypto_handler import MultiScaleCryptoHandler
        from finrl_pro_ds.crypto.envs.crypto_perp_swing_env import CryptoPerpSwingEnv

        # Fetch crypto data (auto-cached to data/crypto_cache/)
        data_cfg = config.get("data", {})
        universe_cfg = config.get("universe", {})
        assets = universe_cfg.get("assets", [])
        cache_key = f"_cmgp1_cache_{data_cfg.get('start_date', '')}"
        if not hasattr(make_env, cache_key):
            from finrl_pro_ds.crypto.data.crypto_loader import fetch_crypto_data
            crypto_data = fetch_crypto_data(
                assets=assets,
                start=data_cfg.get("start_date", "2022-01-01"),
                end=data_cfg.get("end_date"),
                exchange=universe_cfg.get("data_exchange", "binance"),
                cache_dir=data_cfg.get("cache_dir", "./data/crypto_cache"),
            )
            setattr(make_env, cache_key, crypto_data)
        crypto_data = getattr(make_env, cache_key)

        features_cfg = config.get("features", {})
        handler = MultiScaleCryptoHandler(
            ohlcv_df=crypto_data["ohlcv"],
            funding_df=crypto_data["funding"],
            assets=assets,
            feature_config=features_cfg,
            start_date=sd,
            end_date=ed,
            norm_cutoff_date=norm_cutoff_date,
        )
        # Forward feature_set_version for SigBoost V1.1 gate
        env_config.setdefault("feature_set_version", features_cfg.get("feature_set_version", "v1"))
        env = CryptoPerpSwingEnv(config=env_config, data_handler=handler)
        gate_cfg = config.get("signal_gate")
        if gate_cfg and gate_cfg.get("enabled", False):
            from finrl_pro_ds.envs.signal_gated_wrapper import SignalGatedWrapper
            env = SignalGatedWrapper(env, gate_config=gate_cfg)
        return env

    if mdp_version == "v7":
        from finrl_pro_ds.data.multiscale_handler import MultiScaleOHLCVHandler
        from finrl_pro_ds.envs.continuous_swing_env import ContinuousSwingEnv
        features_cfg = config.get("features", {})
        # FE-04 (2026-07-08 FE audit): features_per_scale is declared in three
        # places (features:, env:, network.scale_encoder.input_size) that only
        # agreed by convention. Fail fast on divergence instead of silently
        # training a network whose obs-space metadata disagrees with the
        # handler's actual feature width.
        _fps_declared = {
            "features.features_per_scale": features_cfg.get("features_per_scale"),
            "env.features_per_scale": env_config.get("features_per_scale"),
            "network.scale_encoder.input_size":
                config.get("network", {}).get("scale_encoder", {}).get("input_size"),
        }
        _fps_set = {k: int(v) for k, v in _fps_declared.items() if v is not None}
        if len(set(_fps_set.values())) > 1:
            raise ValueError(
                f"features_per_scale declarations disagree: {_fps_set} "
                f"(FE-04 — the handler emits features.features_per_scale; "
                f"env obs-space and network input must match it)",
            )
        ms_handler = MultiScaleOHLCVHandler(
            file_path=file_path,
            ticker=ticker,
            feature_config=features_cfg,
            start_date=sd,
            end_date=ed,
            norm_cutoff_date=norm_cutoff_date,
        )
        env = ContinuousSwingEnv(config=env_config, data_handler=ms_handler)
        gate_cfg = config.get("signal_gate")
        if gate_cfg and gate_cfg.get("enabled", False):
            from finrl_pro_ds.envs.signal_gated_wrapper import SignalGatedWrapper
            env = SignalGatedWrapper(env, gate_config=gate_cfg)
        # Risk shaping wrapper (outermost — after signal gate).
        # Prefer new env.risk: block; fall back to legacy env.prop_firm: with
        # DeprecationWarning. See .agent/artifacts/prop_firm_decoupling_architecture.md.
        risk_cfg = env_config.get("risk", {})
        pf_cfg = env_config.get("prop_firm", {})
        if risk_cfg.get("enabled", False):
            from finrl_pro_ds.envs.risk_shaping_wrapper import RiskShapingWrapper
            env = RiskShapingWrapper(
                env,
                max_trailing_drawdown_pct=float(risk_cfg.get("max_trailing_drawdown_pct", 0.10)),
                max_daily_loss_pct=float(risk_cfg.get("max_daily_loss_pct", 0.0)),
                eod_hour_utc=int(risk_cfg.get("eod_hour_utc", 0)),
                drawdown_penalty_start=float(risk_cfg.get("drawdown_penalty_start", 0.05)),
                drawdown_penalty_scale=float(risk_cfg.get("drawdown_penalty_scale", 5.0)),
                daily_loss_penalty_start=float(risk_cfg.get("daily_loss_penalty_start", 0.0)),
                daily_loss_penalty_scale=float(risk_cfg.get("daily_loss_penalty_scale", 0.0)),
                augment_obs=str(risk_cfg.get("augment_obs", "off")),
                static_peak=bool(risk_cfg.get("static_peak", True)),
            )
        elif pf_cfg.get("enabled", False):
            import warnings
            warnings.warn(
                "env.prop_firm: is deprecated — migrate to env.risk: + "
                "top-level challenge: block. The PropFirmWrapperV7 adapter "
                "will be removed after Step 6 of the prop-firm decoupling.",
                DeprecationWarning,
                stacklevel=2,
            )
            from finrl_pro_ds.envs.prop_firm_wrapper import PropFirmWrapperV7
            env = PropFirmWrapperV7(
                env,
                profit_target_pct=float(pf_cfg.get("profit_target_pct", 0.10)),
                max_trailing_drawdown_pct=float(pf_cfg.get("max_trailing_drawdown_pct", 0.10)),
                max_daily_loss_pct=float(pf_cfg.get("max_daily_loss_pct", 0.0)),
                eod_hour_utc=int(pf_cfg.get("eod_hour_utc", 0)),
                drawdown_penalty_start=float(pf_cfg.get("drawdown_penalty_start", 0.05)),
                drawdown_penalty_scale=float(pf_cfg.get("drawdown_penalty_scale", 5.0)),
                daily_loss_penalty_start=float(pf_cfg.get("daily_loss_penalty_start", 0.0)),
                daily_loss_penalty_scale=float(pf_cfg.get("daily_loss_penalty_scale", 0.0)),
                success_bonus=float(pf_cfg.get("success_bonus", 10.0)),
                augment_obs=bool(pf_cfg.get("augment_obs", False)),
                static_peak=bool(pf_cfg.get("static_peak", True)),
            )
        return env

    handler = ParquetDataHandler(
        file_path=file_path,
        ticker=ticker,
        feature_config=config.get("features", {}),
        start_date=sd,
        end_date=ed,
        shared_memory_config=shm_config,
        norm_cutoff_date=norm_cutoff_date,  # FIX LEAK-1
    )

    if mdp_version == "v6":
        return SwingScalperEnv(config=env_config, data_handler=handler)
    return DeepScalperEnv(config=env_config, data_handler=handler)


def create_vector_env(config, num_envs, start_date=None, end_date=None,
                      shm_config=None, gym_shm=True, use_sync=False,
                      norm_cutoff_date=None):
    """Create vectorized environment for training.

    Note: Using AsyncVectorEnv for parallel data loading. Context 'spawn' is used
    for CUDA/PyTorch safety. Set use_sync=True to use SyncVectorEnv (no subprocesses),
    which avoids IPC/FD limits on constrained containers.
    """
    # FIX BUG-02: Forward norm_cutoff_date to individual envs for normalization isolation
    env_factory = functools.partial(
        make_env, config=config, start_date=start_date, end_date=end_date,
        shm_config=shm_config, norm_cutoff_date=norm_cutoff_date,
    )

    # audit F17: the phantom-transition auto-reset filter (sac_trainer) assumes
    # Gymnasium >=1.0 NEXT_STEP autoreset semantics. Under 0.29.x (SAME_STEP) it
    # silently drops the first real transition of each episode and bootstraps from
    # the reset obs -> quiet training corruption. Fail loudly instead of silently.
    _parts = (gym.__version__.split(".") + ["0", "0"])[:2]
    _gv = tuple(int(p) if p.isdigit() else 0 for p in _parts)
    assert _gv >= (1, 0), (
        f"Gymnasium >=1.0 required for correct vectorized auto-reset handling "
        f"(got {gym.__version__}); see audit F17 / sac_trainer phantom-transition filter."
    )

    if use_sync:
        # SyncVectorEnv: all envs run in main process. No pipes, no FD issues.
        # Slower but reliable on containers with restricted ulimits.
        env = gym.vector.SyncVectorEnv(
            [env_factory for _ in range(num_envs)],
        )
    else:
        # AsyncVectorEnv for production training (parallel data loading)
        env = gym.vector.AsyncVectorEnv(
            [env_factory for _ in range(num_envs)],
            context="spawn",
            shared_memory=False,
        )

    return env
