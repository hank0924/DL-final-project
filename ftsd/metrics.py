"""Fidelity metrics of Sec. 5.2 and Appendix E.3."""
import warnings

import numpy as np
from scipy import stats


def log_returns(logp):
    return np.diff(np.asarray(logp, dtype=float), axis=-1)


def distribution_tests(real, syn):
    """Two-sample goodness-of-fit between return samples.

    KS, AD and ES are p-values (higher = closer), which is what the bounds quoted in Table 1
    (KS in [0, 1], AD in [0.001, 0.25] in scipy) indicate. W is the Wasserstein distance (lower = closer).
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ks = stats.ks_2samp(real, syn).pvalue
        ad = stats.anderson_ksamp([real, syn]).pvalue
        try:
            es = stats.epps_singleton_2samp(real, syn).pvalue
        except (ValueError, np.linalg.LinAlgError):
            es = np.nan
    return {"KS": ks, "AD": ad, "ES": es, "W": stats.wasserstein_distance(real, syn)}


def performance_ratios(r, periods=252):
    """Annualized Sharpe, Sortino and Calmar ratios of daily log returns (risk-free rate 0)."""
    r = np.asarray(r, dtype=float)
    mu, sd = r.mean(), r.std()
    downside = np.sqrt(np.mean(np.minimum(r, 0.0) ** 2))
    cum = np.concatenate([[0.0], np.cumsum(r)])
    max_dd = 1 - np.exp(-np.max(np.maximum.accumulate(cum) - cum))
    cagr = np.exp(mu * periods) - 1
    return {"SR": mu / sd * np.sqrt(periods), "SoR": mu / downside * np.sqrt(periods),
            "CR": cagr / max_dd if max_dd > 0 else np.nan}


def acf(x, lags):
    x = np.asarray(x, dtype=float) - np.mean(x)
    v = np.dot(x, x)
    return np.array([np.dot(x[:-k], x[k:]) / v for k in lags])


def stylized_facts(r, lags=(1, 5, 10, 20, 50)):
    """Heavy tails and volatility clustering summary."""
    r = np.asarray(r, dtype=float)
    a = acf(np.abs(r), lags)
    out = {"std": r.std(), "skew": stats.skew(r), "kurtosis": stats.kurtosis(r)}
    out.update({f"acf|r|@{k}": v for k, v in zip(lags, a)})
    out["acf r@1"] = acf(r, [1])[0]
    return out
