import numpy as np
import scipy.stats
from tqdm import tqdm
from scipy.stats import norm


class DGP:

    def __init__(self, seed: int):
        np.random.seed(seed)

    def sample(self, sample_size: int, nr_samples: int = 1) -> np.array:
        pass


class DGPNorm(DGP):

    def __init__(self, seed: int, loc: float, scale: float):
        super(DGPNorm, self).__init__(seed)
        self.loc = loc
        self.scale = scale

    def sample(self, sample_size: int, nr_samples: int = 1) -> np.array:
        return np.random.normal(self.loc, self.scale, size=(nr_samples, sample_size))


class DGPExp(DGP):

    def __init__(self, seed: int, scale: float):
        super(DGPExp, self).__init__(seed)
        self.scale = scale                      # 1/lambda

    def sample(self, sample_size: int, nr_samples: int = 1) -> np.array:
        return np.random.exponential(self.scale, size=(nr_samples, sample_size))


class DGPBeta(DGP):

    def __init__(self, seed: int, alpha: float, beta: float):
        super(DGPBeta, self).__init__(seed)
        self.alpha = alpha
        self.beta = beta

    def sample(self, sample_size: int, nr_samples: int = 1) -> np.array:
        return np.random.beta(self.alpha, self.beta, size=(nr_samples, sample_size))


class DGPLogNorm(DGP):

    def __init__(self, seed: int, mean: float, sigma: float):
        super(DGPLogNorm, self).__init__(seed)
        self.mean = mean
        self.sigma = sigma

    def sample(self, sample_size: int, nr_samples: int = 1) -> np.array:
        return np.random.lognormal(self.mean, self.sigma, size=(nr_samples, sample_size))


class DGPLaplace(DGP):

    def __init__(self, seed: int, loc: float, scale: float):
        super(DGPLaplace, self).__init__(seed)
        self.loc = loc
        self.scale = scale

    def sample(self, sample_size: int, nr_samples: int = 1) -> np.array:
        return np.random.laplace(self.loc, self.scale, size=(nr_samples, sample_size))


class DGPBernoulli(DGP):

    def __init__(self, seed: int, p: float):
        super(DGPBernoulli, self).__init__(seed)
        self.p = p

    def sample(self, sample_size: int, nr_samples: int = 1) -> np.array:
        return np.random.binomial(1, self.p, size=(nr_samples, sample_size))


class DGPCategorical(DGP):

    def __init__(self, seed: int, pvals: list):
        super(DGPCategorical, self).__init__(seed)
        self.pvals = pvals

    def sample(self, sample_size: int, nr_samples: int = 1) -> np.array:
        return np.random.multinomial(1, self.pvals, size=(nr_samples, sample_size))


class DGPBiNorm(DGP):

    def __init__(self, seed: int, mean: np.array, cov: np.array):
        super(DGPBiNorm, self).__init__(seed)
        self.mean = mean        # means of both variables, 1D array of length 2
        self.cov = cov          # covariance matrix, 2D array (2x2)

    def sample(self, sample_size: int, nr_samples: int = 1) -> np.array:
        return np.random.multivariate_normal(self.mean, self.cov, size=(nr_samples, sample_size))

