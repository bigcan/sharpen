"""Robustness metrics: Deflated Sharpe Ratio (DSR) and Probability of Backtest Overfitting (PBO).

References:
    - Bailey, D. H., & Lopez de Prado, M. (2014). The Deflated Sharpe Ratio.
    - Bailey, D. H., et al. (2016). The Probability of Backtest Overfitting.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from typing import List, Tuple, Optional, Union

def expected_max_sharpe(n_trials: int, sharpe_mean: float, sharpe_std: float) -> float:
    """
    Estimate the expected Maximum Sharpe Ratio from a set of N independent trials.
    
    Uses the approximation from Bailey & Lopez de Prado (2014):
    E[max] approx mu + sigma * ((1-gamma)*Z^{-1}[1 - 1/N] + gamma*Z^{-1}[1 - 1/(N*e)])
    where gamma is Euler-Mascheroni constant.
    """
    if n_trials <= 1:
        return sharpe_mean
        
    gamma_em = 0.5772156649015328606
    # Z^{-1} is ppf
    z_1 = norm.ppf(1 - 1.0 / n_trials)
    z_2 = norm.ppf(1 - 1.0 / (n_trials * np.e))
    
    return sharpe_mean + sharpe_std * ((1 - gamma_em) * z_1 + gamma_em * z_2)

def deflated_sharpe_ratio(
    observed_sr: float,
    sr_std: float,
    n_trials: int,
    returns_skew: float,
    returns_kurt: float,
    n_returns: int,
    periods_per_year: int = 252
) -> float:
    """
    Compute the Deflated Sharpe Ratio (DSR).
    
    DSR adjusts the Sharpe Ratio for two factors:
    1. Non-normality of returns (skewness/kurtosis) - via PSR logic
    2. Multiple testing (selection bias) - via expected max SR
    
    Args:
        observed_sr: The annualized Sharpe Ratio of the selected strategy.
        sr_std: The standard deviation of the Sharpe Ratios across all N trials (annualized).
        n_trials: Number of independent trials/configurations tested.
        returns_skew: Skewness of the selected strategy's returns.
        returns_kurt: Excess kurtosis of the selected strategy's returns.
        n_returns: Number of return observations (e.g., days) in the track record.
        periods_per_year: Frequency of observations (default 252 for daily).
        
    Returns:
        Probability (0.0 to 1.0) that the true Sharpe Ratio is positive,
        adjusted for selection bias.
    """
    # 1. Estimate Expected Maximum Sharpe (Benchmark SR) under the Null Hypothesis
    # We assume the trials are centered around 0 SR if random, but Bailey uses the actual stats of the trials.
    # Usually for DSR calculation, we test H0: SR <= SR_expected_max.
    # So we use sr_mean = 0.0 for the 'random' baseline assumption in some versions,
    # but strict DSR uses the actual mean of the trials if available. 
    # Common practice: assume mean SR of "random" strategies is 0.
    sr_benchmark = expected_max_sharpe(n_trials, 0.0, sr_std)
    
    # 2. Compute Standard Error of SR (adjusted for non-normality)
    # Reference: Bailey & Lopez de Prado (2012)
    # sigma_SR = sqrt( (1 + 0.5*kurt*SR^2 - skew*SR) / (T-1) ) 
    # Note: observed_sr should be non-annualized for the variance term formula if T is in days?
    # Actually the formula works with annualized SR if consistency is kept, but T must be number of independent obs.
    
    # Convert annualized SR back to per-period for the standard error formula or keep annualized?
    # The standard error formula usually applies to the t-statistic.
    # Bailey (2014) Eq 11:
    # DSR = Z( (SR - E[maxSR]) / (sigma_SR) )
    # where sigma_SR is annualized standard deviation of the ESTIMATOR.
    
    # Term inside sqrt for variance of SR estimator:
    # V[SR] approx (1 / (n_returns - 1)) * (1 - skew*SR + (kurt/4)*SR^2)
    # Wait, Bailey 2012 "The Sharpe Ratio Efficient Frontier" Eq 16:
    # V[SR] = (1/T) * (1 - gamma3*SR + ((gamma4-1)/4)*SR^2)
    # This is for the *annualized* SR? No, usually per period.
    # But if we use annualized values, we must be careful.
    # The ratio (SR - SR_bench) / sigma is dimensionless.
    
    # Let's use standard implementation:
    # T is number of observations.
    
    var_numerator = 1 - returns_skew * observed_sr + ((returns_kurt) / 4.0) * (observed_sr ** 2)
    if var_numerator < 0:
        var_numerator = 1.0 # Fallback
        
    sr_sigma = np.sqrt(var_numerator / (n_returns - 1))
    
    # 3. Compute Z-score
    z = (observed_sr - sr_benchmark) / sr_sigma
    
    # 4. Return CDF
    return norm.cdf(z)

def probability_backtest_overfitting(
    matrix: Union[pd.DataFrame, np.ndarray],
    n_splits: int = 16,
    risk_free: float = 0.0
) -> Tuple[float, List[float]]:
    """
    Compute Probability of Backtest Overfitting (PBO) using Combinatorial CV (CSCV).
    
    Args:
        matrix: (T x N) matrix of returns, where T is time, N is number of strategies.
        n_splits: Number of chunks to split T into. must be even usually.
    
    Returns:
        Tuple(PBO, logits):
            PBO: Probability (0..1) of overfitting.
            logits: The list of OOS rank logits.
    """
    from itertools import combinations
    
    if isinstance(matrix, pd.DataFrame):
        data = matrix.values
    else:
        data = matrix
        
    T, N = data.shape
    if T < n_splits:
        raise ValueError(f"Not enough observations ({T}) for {n_splits} splits.")
        
    # Partition data into S chunks
    chunk_size = T // n_splits
    # We take k = n_splits / 2 for symmetric train/test size usually
    k = n_splits // 2
    
    indices = np.arange(n_splits)
    combs = list(combinations(indices, k))
    
    # Pre-calculate chunk Sharpes to speed up?
    # No, need to concatenate chunks.
    
    # Store OOS relative performance (rank logits)
    logits = []
    
    # To speed up: pre-calculate sums and squared sums for each chunk for each strategy?
    # SR = mean / std.
    # Calculating SR from scratch for each combination is expensive.
    # Optimization: N is usually small (<100 in this phase).
    
    for test_indices in combs:
        test_indices = set(test_indices)
        train_indices = [i for i in range(n_splits) if i not in test_indices]
        
        # Construct Train / Test masks
        # This is a simplified block method. 
        # Real CSCV usually respects time order within blocks but blocks can be shuffled? 
        # No, CSCV blocks are just time slices.
        
        # Assemble Train Returns
        train_rows = []
        for idx in train_indices:
            train_rows.append(data[idx*chunk_size : (idx+1)*chunk_size])
        train_data = np.vstack(train_rows)
        
        # Assemble Test Returns
        test_rows = []
        for idx in test_indices:
            test_rows.append(data[idx*chunk_size : (idx+1)*chunk_size])
        test_data = np.vstack(test_rows)
        
        # Compute IS Sharpes
        # Naive Sharpe (mean/std)
        train_means = np.mean(train_data, axis=0)
        train_stds = np.std(train_data, axis=0)
        with np.errstate(divide='ignore', invalid='ignore'):
            train_sharpes = train_means / train_stds
            train_sharpes[train_stds == 0] = 0
            
        # Identify Best IS Strategy
        best_idx = np.argmax(train_sharpes)
        
        # Compute OOS Sharpes
        test_means = np.mean(test_data, axis=0)
        test_stds = np.std(test_data, axis=0)
        with np.errstate(divide='ignore', invalid='ignore'):
            test_sharpes = test_means / test_stds
            test_sharpes[test_stds == 0] = 0
            
        # Rank of Best IS in OOS
        # We want relative rank.
        # Logit: relative rank logic from Bailey.
        # w_c = rank(R_best_oos) / (N+1)
        # PBO uses logit. 
        
        # Simple implementation: Is the IS-best below the median of OOS?
        # Bailey 2016 Eq 14: phi = relative rank.
        # lambda = log(phi / (1-phi))
        
        # Calculate rank (0 to N-1)
        # argsort gives indices that sort array.
        # argsort of argsort gives rank.
        ranks = np.argsort(np.argsort(test_sharpes))
        rank_of_best = ranks[best_idx] # 0 is worst, N-1 is best
        
        # Normalize rank to (0, 1)
        # relative_rank = (rank_of_best + 1) / (N + 1)
        relative_rank = (rank_of_best + 1) / (N + 1)
        
        # Logit
        # clip to avoid inf
        relative_rank = max(1e-6, min(1 - 1e-6, relative_rank))
        logit = np.log(relative_rank / (1 - relative_rank))
        logits.append(logit)
        
    # PBO = proportion of logits < 0 (which implies rank < 0.5 i.e. below median)
    pbo = np.mean([1.0 if l < 0 else 0.0 for l in logits])
    
    return float(pbo), logits
