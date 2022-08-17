import time

import numpy as np
import scipy
from tqdm import tqdm
from scipy.stats import norm
from scipy.stats import bootstrap as boot_sci
from arch.bootstrap import IIDBootstrap as boot_arch
import matplotlib.pyplot as plt
from collections import defaultdict

from ci_methods import Bootstrap
from generators import DGP, DGPNorm, DGPExp, DGPBeta, DGPBiNorm, DGPLogNorm, DGPLaplace, DGPBernoulli, DGPCategorical
from R_functions import psignrank_range


class CompareIntervals:

    def __init__(self, statistic: callable, methods: list[str], data_generator: DGP, n: int, b: int,
                 alphas: list[float]):
        self.statistic = statistic
        self.methods = methods
        self.dgp = data_generator
        self.n = n
        self.b = b
        self.alphas = np.array(alphas)  # we are interested in one sided intervals, two sided can be computed from them
        self.computed_intervals = {m: {a: [] for a in alphas} for m in methods}  # add all computed intervals
        self.inverse_cdf = {}
        self.times = defaultdict(list)
        self.coverages = {}
        self.distances_from_exact = {}
        self.lengths = {}

    def compute_bootstrap_intervals(self, data: np.array):
        # initialize and sample so we will have the same bootstrap samples for all bootstrap methods
        bts = Bootstrap(data, self.statistic)
        bts.sample(self.b)
        bts.evaluate_statistic()
        for method in self.methods:
            # TODO: avoid new computations for each alpha?
            if method not in bts.implemented_methods:
                continue
            for alpha in self.alphas:
                # ce zelimo dodatne (klasicne) metode en if stavek tuki
                t = time.time()
                ci = bts.ci(coverage=alpha, side='one', method=method)
                self.times[method].append(time.time() - t)
                self.computed_intervals[method][alpha].append(ci[0])
            # print('finished', method)
        return bts

    def compute_non_bootstrap_intervals(self, data: np.array):
        """

        :param data: one sample
        :return:
        """
        ci = defaultdict(list)
        times = defaultdict(list)
        new_methods = {'mean': ['ttest', 'wilcoxon', 'sign'],
                       'median': ['wilcoxon', 'ci_quant_param', 'ci_quant_nonparam', 'maritz-jarrett', 'sign'],
                       'std': ['chi_sq'], 'percentile': ['ci_quant_param', 'ci_quant_nonparam', 'maritz-jarrett'],
                       'corr': ['ci_corr_pearson', 'ci_corr_spearman']}
        if self.statistic.__name__[:10] not in ['mean', 'median', 'std', 'percentile', 'corr']:
            print(f'No known non-bootstrap methods to use for statistic {self.statistic.__name__}.')
            new_methods[self.statistic.__name__] = []

        for method in new_methods[self.statistic.__name__[:10]]:
            if method == 'ttest':
                stat = self.statistic(data)
                se = np.std(data) / np.sqrt(self.n)
                ci[method] = scipy.stats.t.ppf(self.alphas, df=self.n - 1, loc=stat, scale=se)

            elif method == 'sign':
                sorted_data = sorted(data) + [np.inf]  # inf for taking all points?
                p = 0.5  # if self.statistic.__name__ == 'median' else np.mean(data > self.statistic(data))  # TODO ??
                # returning open intervals (-inf, alpha)
                possible_intervals = scipy.stats.binom.cdf(range(self.n + 1), self.n, p)
                # TODO interpolate (/use quantile function?) -> usklajevanje z ostalimi!!!
                ci[method] = [sorted_data[int(np.sum(possible_intervals < a))] for a in self.alphas]

            elif method == 'wilcoxon':
                # signed_ranks = scipy.stats.rankdata(data) * np.sign(data)
                m = int(self.n * (self.n + 1) / 2)
                k = int(m/2)                             # looking only at half points for faster calculation
                conf_half = psignrank_range(k, self.n)
                conf = np.concatenate([conf_half, (1 - conf_half)[::-1]])
                w_sums = sorted([(data[i] + data[j]) / 2 for i in range(self.n) for j in range(i, self.n)])
                from_start = [sum(conf < a) for a in self.alphas]
                ci[method] = [w_sums[f] for f in from_start]

            elif method in ['ci_quant_param', 'ci_quant_nonparam']:
                quant = 0.5 if self.statistic.__name__ == 'median' else int(self.statistic.__name__.split('_')[1])/100

                if method == 'ci_quant_param':
                    m = np.mean(data)
                    s = np.std(data)
                    z = (np.quantile(data, quant, method='inverted_cdf') - m) / s
                    nc = -z * np.sqrt(self.n)
                    ci[method] = m - scipy.stats.nct.ppf(1 - self.alphas, nc=nc, df=self.n - 1) * s / np.sqrt(n)

                elif method == 'ci_quant_nonparam':
                    sorted_data = np.array(sorted(data) + [np.inf])
                    ci[method] = sorted_data[scipy.stats.binom.ppf(self.alphas, self.n, quant).astype(int)]

            elif method == 'maritz-jarrett':
                quant = 0.5 if self.statistic.__name__ == 'median' else int(self.statistic.__name__.split('_')[1])/100
                ci[method] = [scipy.stats.mstats.mquantiles_cimj(data, prob=quant, alpha=abs(2*a-1))[int(a > 0.5)][0]
                              for a in self.alphas]

            elif method == 'chi_sq':
                s = np.std(data)
                qchisq = scipy.stats.chi2.ppf(1 - self.alphas, self.n - 1)
                ci[method] = np.sqrt((self.n - 1) * s ** 2 / qchisq)

            elif method[:7] == 'ci_corr':
                if method == 'ci_corr_pearson':
                    in1 = data[:, 0]
                    in2 = data[:, 1]
                elif method == 'ci_corr_spearman':
                    in1 = scipy.stats.rankdata(data[:, 0])
                    in2 = scipy.stats.rankdata(data[:, 1])

                res = scipy.stats.pearsonr(in1, in2, alternative='less')
                ci[method] = [res.confidence_interval(a).high for a in self.alphas]

            else:
                raise ValueError('Wrong method!!')      # should never happen, just here as a code check

        self.methods.extend(ci.keys())
        for m in ci.keys():
            if m not in self.computed_intervals:
                self.computed_intervals[m] = {a: [c] for a, c in zip(self.alphas, ci[m])}
            else:
                for i in range(len(self.alphas)):
                    self.computed_intervals[m][self.alphas[i]].append(ci[m][i])
            self.times[m].append(times[m])

    def exact_interval_simulation(self, repetitions: int):
        stat_values = np.empty(repetitions)
        data = self.dgp.sample(sample_size=self.n, nr_samples=repetitions)
        for r in range(repetitions):
            stat_values[r] = self.statistic(data[r])

        distribution = stat_values - self.dgp.get_true_value(self.statistic.__name__)

        # inverse cdf that is used for exact interval calculation
        self.inverse_cdf = {a: np.quantile(distribution, 1 - a) for a in self.alphas}

    def compare_intervals(self, repetitions, length=None):
        true_statistic_value = self.dgp.get_true_value(self.statistic.__name__)
        self.exact_interval_simulation(10000)

        stat_original = []
        data = self.dgp.sample(sample_size=self.n, nr_samples=repetitions)

        # calculation with different bootstrap methods
        for r in tqdm(range(repetitions)):
            bts, times = self.compute_bootstrap_intervals(data[r])
            stat_original.append(bts.original_statistic_value)
        stat_original = np.array(stat_original)

        # calculation with non-bootstrap methods
        self.compute_non_bootstrap_intervals(data)

        # exact intervals
        self.computed_intervals['exact'] = {a: stat_original - self.inverse_cdf[a] for a in self.alphas}

        self.coverages = {method: {alpha: np.mean(np.array(self.computed_intervals[method][alpha][-repetitions:]) >
                                                  true_statistic_value) for alpha in self.alphas}
                          for method in self.methods}

        self.distances_from_exact = {method: {alpha: np.array(self.computed_intervals[method][alpha][-repetitions:]) -
                                                     self.computed_intervals['exact'][alpha] for alpha in self.alphas}
                                     for method in self.methods}

        distance_from_exact_stats = {method: {alpha: {'mean': np.mean(self.distances_from_exact[method][alpha]),
                                                      'std': np.std(self.distances_from_exact[method][alpha])}
                                              for alpha in self.alphas} for method in self.methods}

        if length is not None:
            low_alpha, high_alpha = [(1 - length) / 2, (length + 1) / 2]
            if low_alpha not in self.alphas or high_alpha not in self.alphas:
                raise ValueError(f"Length of {length} CI can't be calculated, because we don't have calculations for"
                                 f"corresponding alphas.")
        else:
            low_alpha, high_alpha = [min(self.alphas), max(self.alphas)]
            print(f'Calculating lengths of {high_alpha - low_alpha} CI, from {low_alpha} to {high_alpha} quantiles.')

        self.lengths = {method: np.array(self.computed_intervals[method][high_alpha][-repetitions:]) -
                                np.array(self.computed_intervals[method][low_alpha][-repetitions:])
                        for method in self.methods}

        length_stats = {method: {'mean': np.mean(self.lengths[method]), 'std': np.std(self.lengths[method])}
                        for method in self.methods}

        shapes = {method: (np.array(self.computed_intervals[method][high_alpha][-repetitions:]) - stat_original) /
                          (stat_original - np.array(self.computed_intervals[method][low_alpha][-repetitions:]))
                  for method in self.methods}

        shape_stats = {method: {'mean': np.mean(shapes[method]), 'std': np.std(shapes[method])}
                       for method in self.methods}

        return self.coverages, length_stats, shape_stats, distance_from_exact_stats

    def draw_intervals(self, alphas_to_draw: list[float]):
        data = self.dgp.sample(sample_size=self.n)
        self.alphas = np.union1d(self.alphas, alphas_to_draw)

        # compute bootstrap intervals
        bts = self.compute_bootstrap_intervals(data)

        # compute non-bootstrap intervals
        self.compute_non_bootstrap_intervals(data)

        print({m: [self.computed_intervals[m][a][-1] for a in alphas_to_draw] for m in self.methods})

        # exact intervals calculation
        if not np.array([a in self.inverse_cdf for a in alphas_to_draw]).all():
            self.exact_interval_simulation(10000)
        exact_intervals = [self.statistic(data) - self.inverse_cdf[a] for a in alphas_to_draw]

        # plotting
        colors = iter(plt.cm.jet(np.linspace(0.05, 0.95, len(self.methods))))
        plt.hist(bts.statistic_values, bins=30, label='statistic')
        if 'smoothed' in self.methods:
            plt.hist(bts.statistic_values_noise, bins=30, label='smoothed stat.', alpha=0.3)
        for method in self.methods:
            col = next(colors)
            for alpha in alphas_to_draw:
                if alpha == alphas_to_draw[0]:  # label only the first line of a method to avoid duplicates in legend
                    plt.axvline(self.computed_intervals[method][alpha][-1], linestyle='--', label=method, color=col,
                                alpha=0.75)
                else:
                    plt.axvline(self.computed_intervals[method][alpha][-1], linestyle='-.', color=col, alpha=0.75)

        # draw exact intervals
        for e in exact_intervals:
            if e == exact_intervals[0]:
                plt.axvline(e, linestyle=':', label='exact', color='black', alpha=0.75)
            else:
                plt.axvline(e, linestyle=':', color='black', alpha=0.75)

        plt.title(f'Interval {alphas_to_draw} for {self.statistic.__name__}, n = {self.n}, B = {self.b}')
        plt.legend()
        plt.show()


def compare_bootstraps_with_library_implementations(data, statistic, methods, B, alpha):
    for method in methods:
        print(method)

        # initializations
        b_arch = boot_arch(data)
        if method == 'standard':
            ma = 'norm'
        elif method == 'smoothed':
            # change to percentile because smoothed is not implemented
            ma = 'percentile'
        else:
            ma = method
        ci_arch = b_arch.conf_int(statistic, B, method=ma, size=alpha)
        print(f'Arch: {ci_arch[:, 0]}')

        if method in ['basic', 'percentile', 'bca']:
            # only these are implemented in scipy
            b_sci = boot_sci((data,), statistic, n_resamples=B, confidence_level=alpha, method=method,
                             vectorized=False)
            ci_sci = b_sci.confidence_interval
            print(f'Scipy: {[ci_sci.low, ci_sci.high]}')

        our = Bootstrap(data, statistic)
        our_ci = our.ci(coverage=alpha, nr_bootstrap_samples=B, method=method)  # , seed=0)
        # our.plot_bootstrap_distribution()
        print(f'Our: {our_ci}')

        print('_____________________________________________________________________')


def corr(data):
    c = np.corrcoef(data, rowvar=False)
    return c[0, 1]


def percentile_5(data):
    return np.quantile(data, 0.05)


def percentile_95(data):
    return np.quantile(data, 0.95)


if __name__ == '__main__':

    statistic = np.median
    n = 500
    B = 1000
    alpha = 0.9
    seed = 0
    alphas = [0.05, 0.1, 0.5, 0.9, 0.95]

    methods = ['percentile', 'basic']  # , 'bca', 'bc', 'standard', 'smoothed', 'double']
    # compare_bootstraps_with_library_implementations(data, statistic, methods, B, alpha)

    # jackknife-after-bootstrap
    # our
    # = Bootstrap(data, statistic)
    # our_ci = our.ci(coverage=alpha, nr_bootstrap_samples=B, method='percentile', seed=0)
    # our.jackknife_after_bootstrap([lambda x: np.quantile(x, 0.05), lambda x: np.quantile(x, 0.1), np.mean, np.median,
    #                                lambda x: np.quantile(x, 0.9), lambda x: np.quantile(x, 0.95)])

    # INTERVAL COMPARISON

    dgp = DGPNorm(seed, 0, 1)
    # dgp = DGPBiNorm(seed, np.array([1, 20]), np.array([[3, 2], [2, 2]]))

    comparison = CompareIntervals(statistic, methods, dgp, n, B, alphas)
    # for c in comparison.compare_intervals(repetitions=100):
    #     print(c)

    comparison.draw_intervals([0.1, 0.9])

    # comparison = CompareIntervals(statistic, methods, dgp, 15, B, [0.025, 0.975] + alphas)
    # comparison.compute_non_bootstrap_intervals(np.array([3.7, -4.7, 6.5, -6.9, -7.8, 8.7, 9.1, 10.1, 10.8, 13.6, 14.4, 15.6, 20.2, 22.4, 23.5]))
    # print(comparison.computed_intervals)
    # print({m: [comparison.computed_intervals[m][a][-1] for a in [0.025, 0.975]] for m in comparison.methods})

