# Synapse Crypto 1H — Docker Image
# Extends synapse_v9 base with CCXT, WebSocket, and crypto dependencies
# Timezone: UTC (critical for funding rate alignment)

FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    PIP_NO_CACHE_DIR=1 \
    TZ=UTC

WORKDIR /app

# 1. System dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    wget \
    procps \
    tzdata \
    && ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone \
    && rm -rf /var/lib/apt/lists/*

# 2. Python dependencies (cached layer)
COPY pyproject.toml /app/

RUN pip install --upgrade pip && \
    pip install \
    "numpy>=1.24,<2.0" \
    "gymnasium>=0.29" \
    "stable-baselines3>=2.3.0" \
    optuna==3.5.0 \
    pandas \
    stockstats \
    wandb \
    hmmlearn \
    matplotlib \
    vectorbt \
    scikit-learn \
    scipy \
    # Crypto-specific dependencies
    "ccxt>=4.0.0" \
    "websockets>=12.0" \
    "aiohttp>=3.9.0" \
    "python-dotenv>=1.0" \
    "pybit>=5.0.0" \
    "nest_asyncio>=1.5.0"

# 3. Install local package
COPY finrl_pro_ds /app/finrl_pro_ds
RUN pip install -e .

# 4. Copy scripts and configs
COPY configs /app/configs
COPY scripts /app/scripts

# 5. Create data directories
RUN mkdir -p /app/data/crypto_cache/bronze \
             /app/data/crypto_cache \
             /app/experiments/Synapse_Crypto

# 6. Healthcheck
HEALTHCHECK --interval=5m --timeout=30s \
    CMD python -c "import torch; import ccxt; print('OK')" || exit 1

# Default: run the crypto backtest
CMD ["python", "scripts/crypto_backtest_runner.py"]
