import numpy as np
import scipy.stats
from tqdm import tqdm
from scipy.stats import norm


class DGP:

    def __init__(self, seed: int, true_statistics: dict = {}):
        np.random.seed(seed)
        self.true_statistics = true_statistics

    def sample(self, sample_size: int, nr_samples: int = 1) -> np.array:
        raise NotImplementedError()

    def get_true_value(self, statistic_name):
        return self.true_statistics[statistic_name]


class DGPNorm(DGP):

    def __init__(self, seed: int, loc: float, scale: float, true_statistics: dict = {}):
        super(DGPNorm, self).__init__(seed, true_statistics)
        self.loc = loc
        self.scale = scale
        self.true_statistics['mean'] = loc
        self.true_statistics['median'] = loc
        self.true_statistics['std'] = scale
        self.true_statistics['5_percentil'] = loc - 1.645 * scale
        self.true_statistics['95_percentil'] = loc + 1.645 * scale

    def sample(self, sample_size: int, nr_samples: int = 1) -> np.array:
        return np.random.normal(self.loc, self.scale, size=(nr_samples, sample_size))


class DGPExp(DGP):

    def __init__(self, seed: int, scale: float, true_statistics: dict = {}):
        super(DGPExp, self).__init__(seed, true_statistics)
        self.scale = scale                      # 1/lambda
        self.true_statistics['mean'] = scale
        self.true_statistics['median'] = scale * np.log(2)
        self.true_statistics['std'] = scale ** 2
        self.true_statistics['5_percentil'] = scale * np.log(20/19)
        self.true_statistics['95_percentil'] = scale * np.log(20)

    def sample(self, sample_size: int, nr_samples: int = 1) -> np.array:
        return np.random.exponential(self.scale, size=(nr_samples, sample_size))


class DGPBeta(DGP):

    def __init__(self, seed: int, alpha: float, beta: float, true_statistics: dict = {}):
        super(DGPBeta, self).__init__(seed, true_statistics)
        self.alpha = alpha
        self.beta = beta
        self.true_statistics['mean'] = alpha / (alpha + beta)
        self.true_statistics['std'] = alpha * beta / (alpha + beta) ** 2 / (alpha + beta + 1)
        self.true_statistics['5_percentil'] = scipy.stats.beta.ppf(0.05, alpha, beta)
        self.true_statistics['95_percentil'] = scipy.stats.beta.ppf(0.95, alpha, beta)
        self.true_statistics['median'] = scipy.stats.beta.ppf(0.5, alpha, beta)

    def sample(self, sample_size: int, nr_samples: int = 1) -> np.array:
        return np.random.beta(self.alpha, self.beta, size=(nr_samples, sample_size))


class DGPLogNorm(DGP):

    def __init__(self, seed: int, mean: float, sigma: float, true_statistics: dict = {}):
        super(DGPLogNorm, self).__init__(seed, true_statistics)
        self.mean = mean
        self.sigma = sigma
        self.true_statistics['mean'] = np.exp(mean + (sigma ** 2) / 2)
        self.true_statistics['median'] = np.exp(mean)
        self.true_statistics['std'] = (np.exp(2 * mean + sigma ** 2) * (np.exp(sigma ** 2) - 1)) ** 0.5
        self.true_statistics['5_percentil'] = np.exp(mean - 1.645 * sigma)
        self.true_statistics['95_percentil'] = np.exp(mean + 1.645 * sigma)

    def sample(self, sample_size: int, nr_samples: int = 1) -> np.array:
        return np.random.lognormal(self.mean, self.sigma, size=(nr_samples, sample_size))


class DGPLaplace(DGP):

    def __init__(self, seed: int, loc: float, scale: float, true_statistics: dict = {}):
        super(DGPLaplace, self).__init__(seed, true_statistics)
        self.loc = loc
        self.scale = scale
        self.true_statistics['mean'] = loc
        self.true_statistics['median'] = loc
        self.true_statistics['std'] = scale * 2**0.5
        self.true_statistics['5_percentil'] = loc + scale * np.log(0.1)
        self.true_statistics['95_percentil'] = loc - scale * np.log(0.1)

    def sample(self, sample_size: int, nr_samples: int = 1) -> np.array:
        return np.random.laplace(self.loc, self.scale, size=(nr_samples, sample_size))


class DGPBernoulli(DGP):

    def __init__(self, seed: int, p: float, true_statistics: dict = {}):
        super(DGPBernoulli, self).__init__(seed, true_statistics)
        self.p = p
        self.true_statistics['mean'] = p
        if p == 0.5:
            self.true_statistics['median'] = 0.5
        elif p < 0.5:
            self.true_statistics['median'] = 0
        else:
            self.true_statistics['median'] = 1
        self.true_statistics['std'] = (p * (1 - p)) ** 0.5
        self.true_statistics['5_percentil'] = p >= 0.95
        self.true_statistics['95_percentil'] = p >= 0.05

    def sample(self, sample_size: int, nr_samples: int = 1) -> np.array:
        return np.random.binomial(1, self.p, size=(nr_samples, sample_size))


class DGPCategorical(DGP):

    def __init__(self, seed: int, pvals: np.array, true_statistics: dict = {}):
        super(DGPCategorical, self).__init__(seed, true_statistics)
        self.pvals = pvals
        self.true_statistics['mean'] = pvals
        self.true_statistics['median'] = (pvals > 0.5).astype(float)
        self.true_statistics['median'][pvals == 0.5] = 0.5
        self.true_statistics['std'] = (pvals * (1 - pvals)) ** 0.5
        self.true_statistics['5_percentil'] = (pvals >= 0.95).astype(int)
        self.true_statistics['95_percentil'] = (pvals >= 0.05).astype(int)

    def sample(self, sample_size: int, nr_samples: int = 1) -> np.array:
        return np.random.multinomial(1, self.pvals, size=(nr_samples, sample_size))


class DGPBiNorm(DGP):

    def __init__(self, seed: int, mean: np.array, cov: np.array, true_statistics: dict = {}):
        super(DGPBiNorm, self).__init__(seed, true_statistics)
        self.mean = mean        # means of both variables, 1D array of length 2
        self.cov = cov          # covariance matrix, 2D array (2x2)
        self.true_statistics['mean'] = mean
        self.true_statistics['median'] = mean
        self.true_statistics['std'] = np.diag(cov) ** 0.5
        # self.true_statistics['5_percentil'] = loc - 1.645 * scale     # TODO do we need percentiles??
        # self.true_statistics['95_percentil'] = loc + 1.645 * scale
        self.true_statistics['corr'] = cov[0, 1] / (cov[0, 0] * cov[1, 1]) ** 0.5   # TODO extend it to multiple??

    def sample(self, sample_size: int, nr_samples: int = 1) -> np.array:
        return np.random.multivariate_normal(self.mean, self.cov, size=(nr_samples, sample_size))

