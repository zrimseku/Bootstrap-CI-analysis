import numpy as np
import scipy.stats
from tqdm import tqdm
from scipy.stats import norm
from scipy.stats import bootstrap as boot_sci
from arch.bootstrap import IIDBootstrap as boot_arch
import matplotlib.pyplot as plt

from ci_methods import Bootstrap


class CompareIntervals:

    def __init__(self, statistic, methods, data_generator, n, b, alphas):
        self.statistic = statistic
        self.methods = methods
        self.dgp = data_generator
        self.n = n
        self.b = b
        self.alphas = alphas        # we are interested in one sided intervals, two sided can be computed from them
        self.computed_intervals = {m: {a: [] for a in alphas} for m in methods}  # add all computed intervals

    def compute_intervals(self, data):
        # initialize and sample so we will have the same bootstrap samples for all bootstrap methods
        bts = Bootstrap(data, self.statistic)
        bts.sample(self.b)
        bts.evaluate_statistic()
        for method in self.methods:
            # TODO: avoid new computations for each alpha?
            for alpha in self.alphas:
                # ce zelimo dodatne (klasicne) metode en if stavek tuki
                self.computed_intervals[method][alpha].append(bts.ci(coverage=alpha, side='one', method=method)[0])
            # print('finished', method)
        return bts

    def exact_interval_simulation(self, repetitions):
        # Framework should get this info, not compute it. WRONG intervals, should be around theta hat.
        stat_values = []
        for r in range(repetitions):
            data = self.dgp(self.n)
            stat_values.append(self.statistic(data))

        self.computed_intervals['exact'] = {a: [np.quantile(stat_values, a)] for a in self.alphas}
        return np.mean(stat_values)

    def draw_intervals(self, alphas_to_draw):
        data = self.dgp(self.n)
        self.alphas = np.union1d(self.alphas, alphas_to_draw)
        bts = self.compute_intervals(data)

        # plotting
        colors = iter(plt.cm.jet(np.linspace(0, 0.95, len(self.methods))))
        plt.hist(bts.statistic_values, bins=30, label='statistic')
        if 'smoothed' in self.methods:
            plt.hist(bts.statistic_values_noise, bins=30, label='smoothed statistic', alpha=0.3)
        for method in self.methods:
            col = next(colors)
            for alpha in alphas_to_draw:
                if alpha == alphas_to_draw[0]:  # label only the first line of a method to avoid duplicates in legend
                    plt.axvline(self.computed_intervals[method][alpha][-1], linestyle='--', label=method, color=col,
                                alpha=0.8)
                else:
                    plt.axvline(self.computed_intervals[method][alpha][-1], linestyle='--', color=col, alpha=0.8)

        plt.legend()
        plt.show()

    def compare_intervals(self, repetitions, true_statistic_value=None, length=None):
        if true_statistic_value is None:
            # compute true statistic value from exact simulation, if it is not given
            true_statistic_value = self.exact_interval_simulation(repetitions)

        stat_original = []
        for _ in tqdm(range(repetitions)):
            data = self.dgp(self.n)
            bts = self.compute_intervals(data)
            stat_original.append(bts.original_statistic_value)
        stat_original = np.array(stat_original)

        coverages = {method: {alpha: np.mean(np.array(self.computed_intervals[method][alpha][-repetitions:]) >
                                             true_statistic_value) for alpha in self.alphas} for method in self.methods}

        if length is not None:
            low_alpha, high_alpha = [(1-length)/2, (length+1)/2]
            if low_alpha not in self.alphas or high_alpha not in self.alphas:
                raise ValueError(f"Length of {length} CI can't be calculated, because we don't have calculations for"
                                 f"corresponding alphas.")
        else:
            low_alpha, high_alpha = [min(self.alphas), max(self.alphas)]
            print(f'Calculating lengths of {high_alpha - low_alpha} CI, from {low_alpha} to {high_alpha} quantiles.')

        lengths = {method: np.array(self.computed_intervals[method][high_alpha][-repetitions:]) -
                           np.array(self.computed_intervals[method][low_alpha][-repetitions:])
                   for method in self.methods}

        length_stats = {method: {'mean': np.mean(lengths[method]), 'std': np.std(lengths[method])}
                        for method in self.methods}

        shapes = {method: (np.array(self.computed_intervals[method][high_alpha][-repetitions:]) - stat_original) /
                          (stat_original - np.array(self.computed_intervals[method][low_alpha][-repetitions:]))
                  for method in self.methods}

        shape_stats = {method: {'mean': np.mean(shapes[method]), 'std': np.std(shapes[method])}
                       for method in self.methods}

        return coverages, length_stats, shape_stats


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
        print(f'Arch: {ci_arch[:,0]}')

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


if __name__ == '__main__':
    # data generation
    np.random.seed(0)
    n = 100
    # NORMAL
    # data = np.random.normal(5, 10, 100)
    # GAMMA
    data = np.random.gamma(shape=2, scale=3, size=n)

    # other settings
    statistic = np.median
    B = 1000
    alpha = 0.9
    print(f'Original data statistic value: {statistic(data)}')

    # exact intervals calculation:
    # GAMMA
    exact_simulation = [statistic(np.random.gamma(shape=2, scale=3, size=n)) for _ in range(100000)]
    print(f'Simulated value: {statistic(exact_simulation)}, '
          f'exact simulated CI: {np.quantile(exact_simulation, [(1 - alpha) / 2, 0.5 + alpha / 2])}')

    # print('Exact theoretical CI', np.array(scipy.stats.gamma.interval(alpha=alpha, a=2 * n, scale=3)) / n) only mean

    print('SKEW:', scipy.stats.skew(data))

    methods = ['basic', 'percentile', 'bca', 'bc', 'standard', 'smoothed']#, 'double']
    # compare_bootstraps_with_library_implementations(data, statistic, methods, B, alpha)

    # jackknife-after-bootstrap
    # our = Bootstrap(data, statistic)
    # our_ci = our.ci(coverage=alpha, nr_bootstrap_samples=B, method='percentile', seed=0)
    # our.jackknife_after_bootstrap([lambda x: np.quantile(x, 0.05), lambda x: np.quantile(x, 0.1), np.mean, np.median,
    #                                lambda x: np.quantile(x, 0.9), lambda x: np.quantile(x, 0.95)])

    # INTERVAL COMPARISON
    alphas = [0.05, 0.1, 0.5, 0.9, 0.95]

    statistic = np.median
    true_stat_value = 3
    par2 = 2

    def dgp_norm(x):
        return np.random.normal(true_stat_value, par2, size=x)

    def dgp_gamma(x):
        return np.random.gamma(shape=true_stat_value/par2, scale=par2, size=x)

    comparison = CompareIntervals(statistic, methods, dgp_gamma, n, B, alphas)
    for c in comparison.compare_intervals(repetitions=100, true_statistic_value=true_stat_value):
        print(c)

    comparison.draw_intervals([0.1, 0.9])

