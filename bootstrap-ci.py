import numpy as np
import scipy.stats
from tqdm import tqdm
from scipy.stats import norm
from scipy.stats import bootstrap as boot_sci
from arch.bootstrap import IIDBootstrap as boot_arch
import matplotlib.pyplot as plt


class Bootstrap:

    def __init__(self, data: np.array, statistic: callable):
        self.original_sample = data
        self.original_statistic_value = statistic(data)
        self.statistic = statistic
        self.n = data.shape[0]
        self.b = 0
        self.bootstrap_indices = np.empty(0)
        self.statistic_values = np.empty(0)
        self.statistic_values_noise = np.empty(0)
        # sampling method?
        # CI method?

    def sample(self, nr_bootstrap_samples: int = 1000, seed: int = None, sampling: str = 'nonparametric'):
        """
        Draws bootstrap samples from original dataset.
        :param nr_bootstrap_samples: TODO
        :param seed:
        :param sampling:
        """
        # TODO: include semi-parametric and parametric sampling
        if seed is not None:
            np.random.seed(seed)
        self.b = nr_bootstrap_samples
        self.bootstrap_indices = np.random.choice(range(self.n), size=[nr_bootstrap_samples, self.n])

    def evaluate_statistic(self, noise: np.array = None, sampling: str = 'nonparametric'):
        """Evaluates statistic on bootstrapped datasets"""
        self.statistic_values = np.zeros(self.b)
        if noise is not None:
            # save statistic values with noise separately, so we don't override the original ones when calling smoothed
            self.statistic_values_noise = np.zeros(self.b)
            statistic_input_noise = self.original_sample[self.bootstrap_indices]
            statistic_input_noise += noise
        for i in range(self.b):
            # do we want to call statistic in a vectorized way, to avoid the for loop? Pomojem ne, za splošnost?
            self.statistic_values[i] = self.statistic(self.original_sample[self.bootstrap_indices][i, :])
            if noise is not None:
                self.statistic_values_noise[i] = self.statistic(statistic_input_noise[i, :])

        # TODO: parametric sampling?

    # TODO? For nested we can input sample and indices into evaluate_statistic, to have all evaluations in the same
    #  place? Put "nested" parameter into sampling, to return indices instead of saving them (should we just always
    #  return them?) Polepšaj kodo uglavnem

    def ci(self, coverage: float, side: str = 'two', method: str = 'bca', nr_bootstrap_samples: int = None,
           seed: int = None, sampling: str = 'nonparametric',
           sampling_args: dict = {'kernel': 'norm', 'width': 1}) -> np.array:
        """
        Returns confidence intervals.
        :param coverage: TODO
        :param side:
        :param method:
        :param nr_bootstrap_samples:
        :param seed:
        :param sampling:
        :param sampling_args:
        :return:
        """

        if nr_bootstrap_samples is not None:        # we will sample again, otherwise reuse previously sampled data
            self.sample(nr_bootstrap_samples, seed, sampling)

        if len(self.statistic_values) == 0:
            if method != 'smoothed':
                self.evaluate_statistic()

        quantile = []
        if side == 'two':
            quantile = [(1-coverage)/2, 0.5 + coverage/2]
        elif side == 'one':
            quantile = [coverage]
        else:
            assert ValueError("Choose between 'one' and 'two'-sided intervals when setting parameter side.")

        if method == 'percentile':
            return np.quantile(self.statistic_values, quantile)

        elif method == 'standard':
            sd = np.std(self.statistic_values)
            return self.original_statistic_value + sd * norm.ppf(quantile)

        elif method == 'basic':
            if len(quantile) == 2:
                tails = np.quantile(self.statistic_values, quantile[::-1])
            else:
                # a se v primeru enostranskega basic 90% gleda 2 * ocena - 10%?
                tails = np.quantile(self.statistic_values, 1 - np.array(quantile))

            return 2 * self.original_statistic_value - tails

        elif method[:2] == 'bc':
            bias = np.mean(self.statistic_values < self.original_statistic_value)
            # print('bias', bias)
            a = 0   # for BC method

            if method == 'bca':
                jackknife_values = [self.statistic(self.original_sample[np.arange(self.n) != i]) for i in range(self.n)]
                jack_dot = np.mean(jackknife_values)
                u = (self.n - 1) * (jack_dot - np.array(jackknife_values))
                a = np.sum(u**3) / (6 * np.sum(u**2) ** 1.5)
                # print('a', a)

            z_alpha = norm.ppf(quantile)
            corrected = norm.cdf(norm.ppf(bias) + (norm.ppf(bias) + z_alpha) / (1 - a * (norm.ppf(bias) + z_alpha)))
            # print('quant', z_alpha)
            # print('norm ppf bias', norm.ppf(bias))
            # print('corrected', corrected)
            return np.quantile(self.statistic_values, corrected)

        elif method == 'studentized':   # bootstrap-t, add more possible names for all that have multiple names?
            standard_errors = self.studentized_error_calculation()
            t_samples = (self.statistic_values - self.original_statistic_value) / standard_errors
            se = np.std(self.statistic_values)      # tole naj bi bil se na original podatkih, kako to dobiš avtomatsko?
            # print(self.original_statistic_value)
            # print(np.quantile(t_samples, quantile) * se)
            # print((self.original_statistic_value + np.quantile(t_samples, quantile) * se)) boljši rezultati, ni isto
            return (self.original_statistic_value - np.quantile(t_samples, quantile) * se)[::-1]

        elif method == 'smoothed':
            # TODO automatic setting of parameters (too big width breaks results)
            input_shape = self.original_sample[self.bootstrap_indices].shape
            if sampling_args['kernel'] == 'uniform':
                noise = np.random.uniform(-sampling_args['width'] / 2, sampling_args['width'] / 2, input_shape)
            elif sampling_args['kernel'] == 'norm':
                noise = np.random.normal(0, sampling_args['width'], input_shape)
            else:
                noise = np.zeros(input_shape)
                print(f"Unknown kernel: {sampling_args['kernel']}, using percentile method.")

            self.evaluate_statistic(noise)
            return np.quantile(self.statistic_values_noise, quantile)

        else:
            implemented_methods = ['basic', 'standard', 'percentile', 'bc', 'bca', 'studentized', 'smoothed']
            assert ValueError(f'This method is not supported, choose between {implemented_methods}.')

    def studentized_error_calculation(self):
        # TODO
        #  a bi blo bols dat to funkcijo ven
        #  a je okej da je nested, kaj so druge opcije? lahko delta method
        #  pomojem bi blo kul nastimat, da uporabnik sam poda, ce noce nested delat...
        #  paralelizacija

        # NESTED BOOTSTRAP:
        standard_errors = np.zeros(self.b)
        for i in range(self.b):
            new_indices = np.random.choice(self.bootstrap_indices[i, :], size=[self.b, self.n])
            new_values = np.zeros(self.b)
            for j in range(self.b):
                new_values[j] = self.statistic(self.original_sample[new_indices[j, :]])
            standard_errors[i] = np.std(new_values)

        return standard_errors

    def calibration(self):
        # a bo to posebej, a dodamo v ci, a sploh?
        pass

    def plot_bootstrap_distribution(self):
        """Draws distribution of statistic values on all bootstrap samples."""
        plt.hist(self.statistic_values, bins=30)
        plt.show()

    # DIAGNOSTICS - should they be in separate class?
    def jackknife_after_bootstrap(self, bootstrap_statistics=[np.mean]):
        # would it help to add exact to jackknife - this wouldn't be possible in real case, maybe we can find something
        # from it + draw different percentiles for different methods, see which are less effected by single points
        jk_indices = {i: [j for j in range(self.b) if i not in self.bootstrap_indices[j, :]]
                      for i in range(self.n)}   # gives indices of samples where the point is not included
        # jk_values = {i: [self.statistic(self.original_sample[self.bootstrap_indices[j, :]]) for j in jk_indices[i]]
        #              for i in range(self.n)}    # computes statistic value on those samples

        jk_stat_values = {i: self.statistic_values[jk_indices[i]] for i in range(self.n)}   # statistic on those samples

        # point influences on mean, a je prav da se to gleda? v članku mi ne zgleda, ampak more bit isti influence za
        # vse statistike
        jk_values = np.array([np.mean(jk_stat_values[i]) for i in range(self.n)])
        jk_influence = (np.mean(jk_values) - jk_values) * (self.n - 1)
        point_order = np.argsort(jk_influence)
        # we can also compute influence on length and shape, if we will need it (Efron's JAB SE and Influence fn.)

        min_val = self.original_statistic_value
        for bs in bootstrap_statistics:
            jk_values_bs = np.array([bs(jk_stat_values[i]) for i in range(self.n)])

            plt.plot(jk_influence, jk_values_bs, 'o', color='black')
            plt.plot(jk_influence[point_order], jk_values_bs[point_order], color='black')
            plt.axhline(bs(self.statistic_values), linestyle='--', color='black')

            min_val = min(min_val, min(jk_values_bs))

        for point in range(self.n):
            if abs(jk_influence[point]) > 1:
                plt.text(jk_influence[point], min_val - (self.original_statistic_value - min_val) / 10, point)

        plt.show()


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
            for alpha in self.alphas:
                # ce zelimo dodatne (klasicne) metode en if stavek tuki
                self.computed_intervals[method][alpha].append(bts.ci(coverage=alpha, side='one', method=method)[0])
            # print('finished', method)
        return bts

    def exact_interval_simulation(self, repetitions):
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
                assert ValueError(f"Length of {length} CI can't be calculated, because we don't have calculations for"
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
    n = 10
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

    methods = ['basic', 'percentile', 'bca', 'bc', 'standard', 'smoothed']
    # compare_bootstraps_with_library_implementations(data, statistic, methods, B, alpha)

    # jackknife-after-bootstrap
    # our = Bootstrap(data, statistic)
    # our_ci = our.ci(coverage=alpha, nr_bootstrap_samples=B, method='percentile', seed=0)
    # our.jackknife_after_bootstrap([lambda x: np.quantile(x, 0.05), lambda x: np.quantile(x, 0.1), np.mean, np.median,
    #                                lambda x: np.quantile(x, 0.9), lambda x: np.quantile(x, 0.95)])

    # INTERVAL COMPARISON
    alphas = [0.05, 0.1, 0.5, 0.9, 0.95]

    statistic = np.mean
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

