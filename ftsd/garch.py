"""GJR-GARCH(1,1)-X with relative trading volume as the exogenous variance driver.

    r_t = mu + eps_t,   eps_t = sigma_t * z_t
    sigma_t^2 = omega + (a + g * 1[eps_{t-1} < 0]) * eps_{t-1}^2 + b * sigma_{t-1}^2 + d * x_{t-1}

x_t is the relative volume exp(log V_t - mean of log V over the previous `window` days).
Returns are handled in percent for numerical stability.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.signal import lfilter


def log_relative_volume(volume, window=20):
    """log V_t minus the trailing mean of log V over the previous `window` days (0 when unknown)."""
    lv = np.log(np.maximum(np.asarray(volume, dtype=float), 1.0))
    c = np.concatenate([[0.0], np.cumsum(lv)])
    t = np.arange(len(lv))
    lo = np.maximum(0, t - window)
    cnt = t - lo
    ma = np.where(cnt > 0, (c[t] - c[lo]) / np.maximum(cnt, 1), lv)
    out = lv - ma
    out[lv <= 0] = 0.0
    return out


class GJRGARCHX:
    def __init__(self, use_volume=True):
        self.use_volume = use_volume

    def _variance(self, params, eps, x, s0):
        mu, omega, a, g, b, d = params
        u = omega + (a + g * (eps < 0)) * eps ** 2 + d * x
        return lfilter([1.0], [1.0, -b], np.concatenate([[s0], u[:-1]]))

    def _nll(self, params, r, x):
        mu, omega, a, g, b, d = params
        if a + g / 2 + b >= 0.999:
            return 1e10
        eps = r - mu
        s2 = np.maximum(self._variance(params, eps, x, r.var()), 1e-8)
        return 0.5 * np.sum(np.log(s2) + eps ** 2 / s2)

    def fit(self, returns, lrv=None):
        r = 100 * np.asarray(returns, dtype=float)
        x = np.exp(lrv) if (self.use_volume and lrv is not None) else np.zeros_like(r)
        v = r.var()
        x0 = [r.mean(), 0.05 * v, 0.03, 0.08, 0.88, 0.01 * v if self.use_volume else 0.0]
        bounds = [(None, None), (1e-6, None), (0.0, 1.0), (0.0, 1.0), (0.0, 0.999),
                  (0.0, None) if self.use_volume else (0.0, 0.0)]
        res = minimize(self._nll, x0, args=(r, x), method="L-BFGS-B", bounds=bounds)
        self.params = res.x
        self.success = res.success
        self.s0 = r.var()
        self.mean_x = x.mean()
        self.s2_max = np.max(self._variance(self.params, r - self.params[0], x, self.s0))
        return self

    def fit_constant(self, returns):
        """Constant-volatility special case (a = g = b = d = 0), used for ablations."""
        r = 100 * np.asarray(returns, dtype=float)
        self.use_volume = False
        self.params = np.array([r.mean(), r.var(), 0.0, 0.0, 0.0, 0.0])
        self.success, self.s0, self.mean_x, self.s2_max = True, r.var(), 0.0, r.var()
        return self

    def filter(self, returns, lrv=None):
        """Conditional standard deviation sigma_t (in return units) and standardized residuals z_t."""
        r = 100 * np.asarray(returns, dtype=float)
        x = np.exp(lrv) if (self.use_volume and lrv is not None) else np.zeros_like(r)
        eps = r - self.params[0]
        s2 = np.maximum(self._variance(self.params, eps, x, self.s0), 1e-8)
        sigma = np.sqrt(s2)
        return sigma / 100, eps / sigma

    def step(self, s2_prev, eps_prev, x_prev, cap=2.0):
        """Next conditional variance (percent^2), capped at `cap` times the largest in-sample
        conditional variance so that a run of extreme simulated residuals cannot explode."""
        mu, omega, a, g, b, d = self.params
        x_prev = x_prev if self.use_volume else 0.0
        s2 = omega + (a + g * (eps_prev < 0)) * eps_prev ** 2 + b * s2_prev + d * x_prev
        return min(s2, cap * self.s2_max)

    def summary(self):
        mu, omega, a, g, b, d = self.params
        return (f"mu={mu:.4f}% omega={omega:.4f} alpha={a:.3f} gamma={g:.3f} beta={b:.3f} "
                f"delta(volume)={d:.4f} persistence={a + g / 2 + b:.3f}")
