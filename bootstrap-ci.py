import numpy as np
import scipy.stats
from scipy.stats import norm
from scipy.stats import bootstrap as boot_sci
from arch.bootstrap import IIDBootstrap as boot_arch


class Bootstrap:

    def __init__(self, data: np.array, statistic: callable):
        self.original_sample = data
        self.original_statistic_value = statistic(data)
        self.statistic = statistic
        self.n = data.shape[0]
        self.b = 0
        self.bootstrap_indices = np.empty(0)
        self.statistic_values = np.empty(0)

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

    def evaluate_statistic(self):
        """Evaluates statistic on bootstrapped datasets"""
        # do we want to call statistic in a vectorized way, to avoid the for loop?
        self.statistic_values = np.zeros(self.b)
        for i in range(self.b):
            self.statistic_values[i] = self.statistic(self.original_sample[self.bootstrap_indices[i, :]])

    # TODO? For nested we can input sample and indices into evaluate_statistic, to have all evaluations in the same
    #  place? Put "nested" parameter into sampling, to return indices instead of saving them (should we just always
    #  return them?) Polepšaj kodo uglavnem

    def ci(self, coverage: float, side: str = 'two', method: str = 'bca', nr_bootstrap_samples: int = None,
           seed: int = None, sampling: str = 'nonparametric') -> np.array:
        """
        Returns confidence intervals.
        :param coverage: TODO
        :param side:
        :param method:
        :param nr_bootstrap_samples:
        :param seed:
        :param sampling:
        :return:
        """

        if nr_bootstrap_samples is not None:        # we will sample again, otherwise reuse previously sampled data
            self.sample(nr_bootstrap_samples, seed, sampling)
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
            print(self.original_statistic_value)
            print(np.quantile(t_samples, quantile) * se)
            print((self.original_statistic_value + np.quantile(t_samples, quantile) * se))
            return (self.original_statistic_value - np.quantile(t_samples, quantile) * se)[::-1]

        elif method == 'tilted':
            return

        else:
            implemented_methods = ['basic', 'standard', 'percentile', 'bc', 'bca', 'studentized', 'tilted']
            assert ValueError(f'This method is not supported, choose between {implemented_methods}.')

    def studentized_error_calculation(self):
        # a bi blo bols dat to funkcijo ven
        # je okej da je nested, kaj so druge opcije? lahko delta method
        # pomojem bi blo kul nastimat, da uporabnik sam poda, ce noce nested delat...

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


if __name__ == '__main__':
    # data generation
    np.random.seed(0)
    # data = np.random.normal(5, 10, 100)
    # print(data)
    data = np.random.gamma(2, 3, 100)
    print(scipy.stats.skew(data))

    # for method in ['basic', 'percentile', 'bca', 'bc', 'studentized', 'standard']:
    for method in ['percentile', 'studentized']:
        print(method)
        # other settings
        statistic = np.mean
        # method = "percentile"
        B = 1000
        confidence = 0.9

        print(f'Original data statistic value: {statistic(data)}')

        # initializations
        b_arch = boot_arch(data)
        if method == 'standard':
            ma = 'norm'
        else:
            ma = method
        ci_arch = b_arch.conf_int(statistic, B, method=ma, size=confidence)
        print(f'Arch: {ci_arch[:,0]}')

        if method in ['basic', 'percentile', 'bca']:
            b_sci = boot_sci((data,), statistic, n_resamples=B, confidence_level=confidence, method=method,
                             vectorized=False)
            ci_sci = b_sci.confidence_interval
            print(f'Scipy: {[ci_sci.low, ci_sci.high]}')

        our = Bootstrap(data, statistic)
        our_ci = our.ci(coverage=confidence, nr_bootstrap_samples=B, method=method)
        print(f'Our: {our_ci}')

        print('_____________________________________________________________________')
