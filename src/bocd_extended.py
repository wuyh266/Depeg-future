import numpy as np
from scipy.special import logsumexp
from scipy.stats import t as student_t
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class ConstantHazard:
    def __init__(self, lam):
        self.lam = lam

    def log_hazard(self, r):
        return -np.log(self.lam) * np.ones_like(r)

    def log_1m_hazard(self, r):
        return np.log(1 - 1/self.lam) * np.ones_like(r)


class StudentTModel:
    def __init__(self, mu0=0, kappa0=1, alpha0=1, beta0=1):
        self.mu0 = mu0
        self.kappa0 = kappa0
        self.alpha0 = alpha0
        self.beta0 = beta0
        self.reset()

    def reset(self):
        self.mu = np.array([self.mu0])
        self.kappa = np.array([self.kappa0])
        self.alpha = np.array([self.alpha0])
        self.beta = np.array([self.beta0])

    def log_predictive(self, x):
        scale = np.sqrt(
            self.beta * (self.kappa + 1) /
            (self.alpha * self.kappa)
        )
        df = 2 * self.alpha
        return student_t.logpdf(x, df=df, loc=self.mu, scale=scale)

    def update(self, x):
        mu_new = np.concatenate([
            [self.mu0],
            (self.kappa * self.mu + x) / (self.kappa + 1)
        ])

        kappa_new = np.concatenate([
            [self.kappa0],
            self.kappa + 1
        ])

        alpha_new = np.concatenate([
            [self.alpha0],
            self.alpha + 0.5
        ])

        beta_new = np.concatenate([
            [self.beta0],
            self.beta + (self.kappa * (x - self.mu)**2) / (2*(self.kappa + 1))
        ])

        self.mu = mu_new
        self.kappa = kappa_new
        self.alpha = alpha_new
        self.beta = beta_new

    def truncate(self, R):
        self.mu = self.mu[:R]
        self.kappa = self.kappa[:R]
        self.alpha = self.alpha[:R]
        self.beta = self.beta[:R]

    def get_posterior_mean(self):
        return self.mu[0]

    def get_posterior_std(self):
        return np.sqrt(self.beta[0] / self.alpha[0])


class BOCD:
    def __init__(self, hazard, model, truncate_threshold=1e-4):
        self.hazard = hazard
        self.model = model
        self.truncate_threshold = truncate_threshold
        self.reset()

    def reset(self):
        self.model.reset()
        self.log_R = np.array([0.0])
        self.cp_probs = []
        self.entropy = []
        self.log_evidence = []

    def update(self, x):
        T = len(self.log_R)

        log_pred = self.model.log_predictive(x)

        r = np.arange(T)
        log_growth = (
            self.log_R
            + log_pred
            + self.hazard.log_1m_hazard(r)
        )

        log_cp = logsumexp(
            self.log_R
            + log_pred
            + self.hazard.log_hazard(r)
        )

        new_log_R = np.concatenate([[log_cp], log_growth])

        log_norm = logsumexp(new_log_R)
        new_log_R -= log_norm

        self.log_evidence.append(log_norm)

        cp_prob = np.exp(new_log_R[0])
        self.cp_probs.append(cp_prob)

        entropy = -np.sum(np.exp(new_log_R) * new_log_R)
        self.entropy.append(entropy)

        self.log_R = new_log_R

        self.model.update(x)

        self._truncate()

    def _truncate(self):
        R = len(self.log_R)
        probs = np.exp(self.log_R)
        cumsum = np.cumsum(probs)

        idx = np.searchsorted(cumsum, 1 - self.truncate_threshold) + 1

        if idx < R:
            self.log_R = self.log_R[:idx]
            self.model.truncate(idx)

    def get_cp_probs(self):
        return np.array(self.cp_probs)

    def get_entropy(self):
        return np.array(self.entropy)

    def get_run_length_posterior(self):
        return np.exp(self.log_R)


class ExtendedBOCD:
    def __init__(self, hazard_rate=200, truncate_threshold=1e-4):
        self.hazard = ConstantHazard(hazard_rate)
        self.model = StudentTModel(mu0=0, kappa0=1, alpha0=1, beta0=1)
        self.bocd = BOCD(self.hazard, self.model, truncate_threshold)
        
        self.deviation_scores = []
        self.surprise_scores = []
        self.prediction_errors = []
        self.posterior_means = []
        self.posterior_stds = []

    def reset(self):
        self.bocd.reset()
        self.deviation_scores = []
        self.surprise_scores = []
        self.prediction_errors = []
        self.posterior_means = []
        self.posterior_stds = []

    def update(self, x):
        posterior_mean_before = self.model.get_posterior_mean()
        posterior_std_before = self.model.get_posterior_std()
        
        self.bocd.update(x)
        
        pred_mean = self.model.mu[0]
        pred_std = np.sqrt(self.model.beta[0] * (self.model.kappa[0] + 1) / 
                          (self.model.alpha[0] * self.model.kappa[0]))
        
        z_score = abs(x - pred_mean) / (pred_std + 1e-8)
        self.deviation_scores.append(z_score)
        
        prediction_error = abs(x - posterior_mean_before)
        self.prediction_errors.append(prediction_error)
        
        surprise = -self.bocd.log_R[0] if len(self.bocd.log_R) > 0 else 0
        self.surprise_scores.append(surprise)
        
        self.posterior_means.append(self.model.get_posterior_mean())
        self.posterior_stds.append(self.model.get_posterior_std())

    def get_features(self):
        return {
            'cp_prob': np.array(self.bocd.cp_probs),
            'deviation': np.array(self.deviation_scores),
            'surprise': np.array(self.surprise_scores),
            'prediction_error': np.array(self.prediction_errors),
            'entropy': np.array(self.bocd.entropy),
            'posterior_mean': np.array(self.posterior_means),
            'posterior_std': np.array(self.posterior_stds),
            'run_length': self.bocd.get_run_length_posterior()
        }

    def get_latest_features(self):
        return {
            'cp_prob': self.bocd.cp_probs[-1] if self.bocd.cp_probs else 0,
            'deviation': self.deviation_scores[-1] if self.deviation_scores else 0,
            'surprise': self.surprise_scores[-1] if self.surprise_scores else 0,
            'prediction_error': self.prediction_errors[-1] if self.prediction_errors else 0,
            'entropy': self.bocd.entropy[-1] if self.bocd.entropy else 0,
            'posterior_mean': self.posterior_means[-1] if self.posterior_means else 0,
            'posterior_std': self.posterior_stds[-1] if self.posterior_stds else 0,
        }


def process_coin_with_bocd(price_series, hazard_rate=200):
    returns = np.diff(np.log(price_series))
    returns = returns[~np.isnan(returns)]
    returns = returns[~np.isinf(returns)]
    
    bocd = ExtendedBOCD(hazard_rate=hazard_rate)
    
    for x in returns:
        bocd.update(x)
    
    return bocd.get_features()


if __name__ == '__main__':
    import pandas as pd
    
    df = pd.read_csv('data/usdc/prices.csv', parse_dates=['timestamp'])
    prices = df['price'].values
    
    features = process_coin_with_bocd(prices)
    print("Sample deviation scores:", features['deviation'][-10:])
    print("Sample cp_probs:", features['cp_prob'][-10:])
