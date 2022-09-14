import itertools

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
        if statistic_name not in self.true_statistics:
            raise ValueError(f"True value of {statistic_name} is not known. You should specify it at DGP "
                             f"initialization.")
        return self.true_statistics[statistic_name]

    def describe(self):
        return type(self).__name__


class DGPNorm(DGP):

    def __init__(self, seed: int, loc: float = 0, scale: float = 1, true_statistics: dict = {}):
        super(DGPNorm, self).__init__(seed, true_statistics)
        self.loc = loc
        self.scale = scale
        self.true_statistics['mean'] = loc
        self.true_statistics['median'] = loc
        self.true_statistics['std'] = scale
        self.true_statistics['percentile_5'] = loc - 1.645 * scale
        self.true_statistics['percentile_95'] = loc + 1.645 * scale

    def sample(self, sample_size: int, nr_samples: int = 1) -> np.array:
        size = (nr_samples, sample_size) if nr_samples != 1 else sample_size
        return np.random.normal(self.loc, self.scale, size=size)

    def describe(self):
        return type(self).__name__ + '_' + str(self.loc) + '_' + str(self.scale)


class DGPExp(DGP):

    def __init__(self, seed: int, scale: float = 1, true_statistics: dict = {}):
        super(DGPExp, self).__init__(seed, true_statistics)
        self.scale = scale                      # 1/lambda
        self.true_statistics['mean'] = scale
        self.true_statistics['median'] = scale * np.log(2)
        self.true_statistics['std'] = scale
        self.true_statistics['percentile_5'] = scale * np.log(20/19)
        self.true_statistics['percentile_95'] = scale * np.log(20)

    def sample(self, sample_size: int, nr_samples: int = 1) -> np.array:
        size = (nr_samples, sample_size) if nr_samples != 1 else sample_size
        return np.random.exponential(self.scale, size=size)

    def describe(self):
        return type(self).__name__ + '_' + str(self.scale)


class DGPBeta(DGP):

    def __init__(self, seed: int, alpha: float = 1, beta: float = 1, true_statistics: dict = {}):
        super(DGPBeta, self).__init__(seed, true_statistics)
        self.alpha = alpha
        self.beta = beta
        self.true_statistics['mean'] = alpha / (alpha + beta)
        self.true_statistics['std'] = np.sqrt(alpha * beta / (alpha + beta) ** 2 / (alpha + beta + 1))
        self.true_statistics['percentile_5'] = scipy.stats.beta.ppf(0.05, alpha, beta)
        self.true_statistics['percentile_95'] = scipy.stats.beta.ppf(0.95, alpha, beta)
        self.true_statistics['median'] = scipy.stats.beta.ppf(0.5, alpha, beta)

    def sample(self, sample_size: int, nr_samples: int = 1) -> np.array:
        size = (nr_samples, sample_size) if nr_samples != 1 else sample_size
        return np.random.beta(self.alpha, self.beta, size=size)

    def describe(self):
        return type(self).__name__ + '_' + str(self.alpha) + '_' + str(self.beta)


class DGPLogNorm(DGP):

    def __init__(self, seed: int, mean: float, sigma: float, true_statistics: dict = {}):
        super(DGPLogNorm, self).__init__(seed, true_statistics)
        self.mean = mean
        self.sigma = sigma
        self.true_statistics['mean'] = np.exp(mean + (sigma ** 2) / 2)
        self.true_statistics['median'] = np.exp(mean)
        self.true_statistics['std'] = (np.exp(2 * mean + sigma ** 2) * (np.exp(sigma ** 2) - 1)) ** 0.5
        self.true_statistics['percentile_5'] = np.exp(mean - 1.645 * sigma)
        self.true_statistics['percentile_95'] = np.exp(mean + 1.645 * sigma)

    def sample(self, sample_size: int, nr_samples: int = 1) -> np.array:
        size = (nr_samples, sample_size) if nr_samples != 1 else sample_size
        return np.random.lognormal(self.mean, self.sigma, size=size)

    def describe(self):
        return type(self).__name__ + '_' + str(self.mean) + '_' + str(self.sigma)


class DGPLaplace(DGP):

    def __init__(self, seed: int, loc: float, scale: float, true_statistics: dict = {}):
        super(DGPLaplace, self).__init__(seed, true_statistics)
        self.loc = loc
        self.scale = scale
        self.true_statistics['mean'] = loc
        self.true_statistics['median'] = loc
        self.true_statistics['std'] = scale * 2**0.5
        self.true_statistics['percentile_5'] = loc + scale * np.log(0.1)
        self.true_statistics['percentile_95'] = loc - scale * np.log(0.1)

    def sample(self, sample_size: int, nr_samples: int = 1) -> np.array:
        size = (nr_samples, sample_size) if nr_samples != 1 else sample_size
        return np.random.laplace(self.loc, self.scale, size=size)

    def describe(self):
        return type(self).__name__ + '_' + str(self.loc) + '_' + str(self.scale)


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
        self.true_statistics['percentile_5'] = p >= 0.95
        self.true_statistics['percentile_95'] = p >= 0.05

    def sample(self, sample_size: int, nr_samples: int = 1) -> np.array:
        size = (nr_samples, sample_size) if nr_samples != 1 else sample_size
        return np.random.binomial(1, self.p, size=size).astype(float)

    def describe(self):
        return type(self).__name__ + '_' + str(self.p)


class DGPCategorical(DGP):

    def __init__(self, seed: int, pvals: np.array, true_statistics: dict = {}):
        super(DGPCategorical, self).__init__(seed, true_statistics)
        self.pvals = pvals
        self.true_statistics['mean'] = np.sum(pvals * np.array(range(len(pvals))))
        self.true_statistics['median'] = np.where(np.cumsum(pvals) > 0.5)[0][0]
        # self.true_statistics['std'] = (pvals * (1 - pvals)) ** 0.5 ???
        self.true_statistics['percentile_5'] = np.where(np.cumsum(pvals) > 0.05)[0][0]
        self.true_statistics['percentile_95'] = np.where(np.cumsum(pvals) > 0.95)[0][0]

    def sample(self, sample_size: int, nr_samples: int = 1) -> np.array:
        size = (nr_samples, sample_size) if nr_samples != 1 else sample_size
        return np.array([np.argmax(c, axis=-1) for c in np.random.multinomial(1, self.pvals, size=size)]).astype(float)

    def describe(self):
        return type(self).__name__ + '_' + str(self.pvals)


class DGPBiNorm(DGP):

    def __init__(self, seed: int, mean: np.array, cov: np.array, true_statistics: dict = {}):
        super(DGPBiNorm, self).__init__(seed, true_statistics)
        self.mean = mean        # means of both variables, 1D array of length 2
        self.cov = cov          # covariance matrix, 2D array (2x2)
        self.true_statistics['mean'] = mean                         # TODO do we need any of these (for now just corr)
        self.true_statistics['median'] = mean
        self.true_statistics['std'] = np.diag(cov) ** 0.5
        # self.true_statistics['5_percentile'] = loc - 1.645 * scale     # TODO do we need percentiles??
        # self.true_statistics['95_percentile'] = loc + 1.645 * scale
        self.true_statistics['corr'] = cov[0, 1] / (cov[0, 0] * cov[1, 1]) ** 0.5   # extend it to multiple??

    def sample(self, sample_size: int, nr_samples: int = 1) -> np.array:
        size = (nr_samples, sample_size) if nr_samples != 1 else sample_size
        return np.random.multivariate_normal(self.mean, self.cov, size=size)

    def describe(self):
        return type(self).__name__ + '_'.join([str(par) for par in [self.mean[0], self.mean[1], self.cov[0, 0],
                                                                    self.cov[0, 1], self.cov[1, 1]]])


class DGPRandEff(DGP):

    def __init__(self, seed: 0, mean: float, stds: list, true_statistics: dict = {}):
        super(DGPRandEff, self).__init__(seed, true_statistics)
        self.group_indices = []
        self.mean = mean
        self.stds = stds
        self.true_statistics['mean'] = mean
        self.true_statistics['median'] = mean
        self.true_statistics['std'] = stds  # TODO this depends on strategy, what to do?

    def sample(self, group_sizes: list, nr_samples: int = 1) -> np.array:
        # TODO check if group_sizes has same depth as stds
        counter = itertools.count()

        def get_indices(sizes, cnt):
            if isinstance(sizes, int):
                return [next(cnt) for _ in range(sizes)]
            else:
                return [get_indices(sizes[j], cnt) if isinstance(sizes[j], int) else
                        get_indices(sizes[j], cnt) for j in range(len(sizes))]

        self.group_indices = get_indices(group_sizes, counter)

        sample_size = next(counter)
        size = (nr_samples, sample_size) if nr_samples != 1 else sample_size
        data = np.zeros(size) + self.mean

        def add_error(indices, data, depth, err):
            if isinstance(indices, int):
                data[indices] += err
            else:
                for inds in indices:
                    add_error(inds, data, depth + 1, err + np.random.normal(0, self.stds[depth]))

        add_error(self.group_indices, data, 0, 0)
        return data


if __name__ == '__main__':
    dgp = DGPRandEff(0, 0, [100, 10, 1, 0.1])
    print(dgp.sample([[[3, 4, 5], [2, 3]], [[3, 4, 5], [2, 3]]], 1))
