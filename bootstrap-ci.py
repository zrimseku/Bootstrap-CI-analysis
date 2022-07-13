import numpy as np
import scipy.stats
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
        # sampling method?
        # CI method?

    def sample(self, nr_bootstrap_samples: int = 1000, seed: int = None, sampling: str = 'nonparametric'):
        """
        Draws bootstrap samples from original dataset.
        :param nr_bootstrap_samples: TODO
        :param seed:
        :param sampling:
        """
        # TODO: include semi-parametric and parametric sampling -> in evaluate?
        if seed is not None:
            np.random.seed(seed)
        self.b = nr_bootstrap_samples
        self.bootstrap_indices = np.random.choice(range(self.n), size=[nr_bootstrap_samples, self.n])

    def evaluate_statistic(self, noise: np.array = None, sampling: str = 'nonparametric'):
        """Evaluates statistic on bootstrapped datasets"""
        self.statistic_values = np.zeros(self.b)
        statistic_input = self.original_sample[self.bootstrap_indices]
        if noise is not None:
            statistic_input += noise
        for i in range(self.b):
            # do we want to call statistic in a vectorized way, to avoid the for loop? Pomojem ne, za splošnost?
            self.statistic_values[i] = self.statistic(statistic_input[i, :])

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
            return np.quantile(self.statistic_values, quantile)

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

    # methods = ['basic', 'percentile', 'bca', 'bc', 'standard', 'studentized']
    # compare_bootstraps_with_library_implementations(data, statistic, methods, B, alpha)

    # jackknife-after-bootstrap
    our = Bootstrap(data, statistic)
    our_ci = our.ci(coverage=alpha, nr_bootstrap_samples=B, method='percentile', seed=0)
    our.jackknife_after_bootstrap([lambda x: np.quantile(x, 0.05), lambda x: np.quantile(x, 0.1), np.mean, np.median,
                                   lambda x: np.quantile(x, 0.9), lambda x: np.quantile(x, 0.95)])

