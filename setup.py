from setuptools import setup, find_packages

setup(
    name="finrl_pro_ds",
    version="0.1.0",
    author="Keng Lee",
    author_email="bigcan@users.noreply.github.com",
    url=" ",
    license="Apache 2.0",
    packages=find_packages(),
    install_requires=[
        'torch>=2.8.0',
        'pandas>=2.1.0',
        'pyyaml>=6.0',
        'requests==2.32.3',
        'yfinance>=0.2.40',
        'pyarrow>=14.0.1',
        'alpaca-py>=0.23.0',
        'PyWavelets>=1.5.0',
        'statsmodels>=0.14.0',
        'gymnasium>=0.29.1',
        'hmmlearn>=0.3.0',
        'scikit-learn>=1.3.0',
        'wandb>=0.16.0',
        'python-binance>=1.0.19',
        'ta>=0.11.0',
        'python-dotenv>=1.0.0',
        'kaggle>=1.5.16',
        'protobuf==3.20.1',
        'rich>=14.0.0'
    ],
    description="FinRL-Pro_DS",
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: Implementation :: CPython",
        "Programming Language :: Python :: Implementation :: PyPy",
    ],
    keywords="Deep Reinforcment Learning",
    python_requires=">=3.6",
)
