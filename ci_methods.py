import itertools

import numpy as np
import scipy.stats
from numba import njit
from tqdm import tqdm
from scipy.stats import norm
from scipy.stats import bootstrap as boot_sci
from arch.bootstrap import IIDBootstrap as boot_arch
import matplotlib.pyplot as plt


class Bootstrap:

    def __init__(self, data: np.array, statistic: callable, use_jit: bool = False, group_indices: list = None):
        self.original_sample = data
        self.original_statistic_value = statistic(data)
        self.statistic = statistic
        self.n = data.shape[0]
        self.b = 0
        self.bootstrap_indices = np.empty(0)
        self.statistic_values = np.empty(0)
        self.statistic_values_noise = np.empty(0)
        self.implemented_methods = ['basic', 'standard', 'percentile', 'bc', 'bca', 'studentized', 'smoothed', 'double']
        self.use_jit = use_jit
        self.group_indices = group_indices      # hierarchical structure needed for hierarchical bootstrap

    def sample(self, nr_bootstrap_samples: int = 1000, seed: int = None, sampling: str = 'nonparametric',
               sampling_args: dict = None):
        """
        Draws bootstrap samples from original dataset.
        :param nr_bootstrap_samples: TODO
        :param seed:
        :param sampling:
        :param sampling_args:
        """
        # TODO: include semi-parametric and parametric sampling
        if seed is not None:
            np.random.seed(seed)
        self.b = nr_bootstrap_samples

        if sampling == 'nonparametric':
            self.bootstrap_indices = np.random.choice(range(self.n), size=[nr_bootstrap_samples, self.n])

        elif sampling == 'hierarchical':
            method = sampling_args['method']            # cases / random-effect / residuals?
            strategy = sampling_args['strategy']        # with (True) or without (False) replacement on each level

            if method == 'cases':
                groups_n = [[self.group_indices.copy()] for _ in range(nr_bootstrap_samples)]
                for s in strategy:
                    if s:       # with replacement
                        gr_indices_n = [[np.random.randint(len(g), size=len(g)) for g in groups] for groups in groups_n]
                        groups_n = [list(itertools.chain.from_iterable([[g[i] for i in ids]
                                                                        for ids, g in zip(gr_indices, groups)]))
                                    for gr_indices, groups in zip(gr_indices_n, groups_n)]
                    else:       # without replacement
                        groups_n = [list(itertools.chain.from_iterable(groups)) for groups in groups_n]

                self.bootstrap_indices = groups_n  # can't be numpy array as lengths can be different

            else:
                raise ValueError(f'Method {method} of hierarchical sampling is not implemented. Choose between cases'
                                 f'and random-effect.')



        else:
            raise ValueError(f'{sampling} sampling is not implemented. Choose between nonparametric and hierarchical.')

    def evaluate_statistic(self, noise: np.array = None, sampling: str = 'nonparametric'):
        """Evaluates statistic on bootstrapped datasets"""
        if np.size(self.original_statistic_value) == 1:
            self.statistic_values = np.zeros(self.b)
        else:
            self.statistic_values = np.zeros((self.b, np.size(self.original_statistic_value)))

        if noise is not None:
            # save statistic values with noise separately, so we don't override the original ones when calling smoothed
            if np.size(self.original_statistic_value) == 1:
                self.statistic_values_noise = np.zeros(self.b)
            else:
                self.statistic_values_noise = np.zeros((self.b, np.size(self.original_statistic_value)))

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

    def ci(self, coverages: np.array, side: str = 'two', method: str = 'bca', nr_bootstrap_samples: int = None,
           seed: int = None, sampling: str = 'nonparametric', quantile_type: str = 'median_unbiased',
           sampling_args: dict = {'kernel': 'norm', 'width': None}) -> np.array:
        """
        Returns confidence intervals.
        :param coverages: TODO
        :param side:
        :param method:
        :param nr_bootstrap_samples:
        :param seed:
        :param sampling:
        :param quantile_type:
        :param sampling_args:
        :return:
        """

        if nr_bootstrap_samples is not None:        # we will sample again, otherwise reuse previously sampled data
            self.sample(nr_bootstrap_samples, seed, sampling)

        if len(self.statistic_values) == 0:
            # if method != 'smoothed':  calculate even for smoothed, to get the rule-of-thumb estimation of bandwidth
            self.evaluate_statistic()
        quantiles = []
        if side == 'two':
            quantiles = np.array([(1-coverages)/2, 0.5 + coverages/2])
        elif side == 'one':
            quantiles = np.array(coverages)
        else:
            assert ValueError("Choose between 'one' and 'two'-sided intervals when setting parameter side.")

        if method == 'percentile':
            return np.quantile(self.statistic_values, quantiles, method=quantile_type)

        elif method == 'standard':
            sd = np.std(self.statistic_values)
            return self.original_statistic_value + sd * norm.ppf(quantiles)

        elif method == 'basic':
            tails = np.quantile(self.statistic_values, 1 - quantiles, method=quantile_type)

            return 2 * self.original_statistic_value - tails

        elif method[:2] == 'bc':
            bias = np.mean(self.statistic_values < self.original_statistic_value)

            if bias == 1 or bias == 0:
                e = np.empty(quantiles.shape)
                e[:] = np.nan
                return e

            a = 0   # for BC method
            if method == 'bca':
                jackknife_values = [self.statistic(self.original_sample[np.arange(self.n) != i]) for i in range(self.n)]
                jack_dot = np.mean(jackknife_values)
                u = (self.n - 1) * (jack_dot - np.array(jackknife_values))
                if np.sum(u**2) == 0:
                    u += 1e-10                      # hack for u = 0
                a = np.sum(u**3) / (6 * np.sum(u**2) ** 1.5)
                # print('a', a)

            z_alpha = norm.ppf(quantiles)
            # print(self.n, self.b, self.statistic.__name__)
            corrected = norm.cdf(norm.ppf(bias) + (norm.ppf(bias) + z_alpha) / (1 - a * (norm.ppf(bias) + z_alpha)))
            # print('quant', z_alpha)
            # print('norm ppf bias', norm.ppf(bias))
            # print('corrected', corrected)
            return np.quantile(self.statistic_values, corrected, method=quantile_type)

        elif method == 'studentized':   # bootstrap-t, add more possible names for all that have multiple names?
            standard_errors = self.studentized_error_calculation()
            t_samples = (self.statistic_values - self.original_statistic_value) / standard_errors
            se = np.std(self.statistic_values)      # tole naj bi bil se na original podatkih, kako to dobiš avtomatsko?
            # TODO nanquantiles, skips standard errors that would be 0 - OK?
            return self.original_statistic_value - np.nanquantile(t_samples, 1 - quantiles, method=quantile_type) * se

        elif method == 'smoothed':
            input_shape = self.original_sample[self.bootstrap_indices].shape
            if sampling_args['width'] is None:
                # rule of thumb width selection (we can improve it with AMISE/MISE approximation if needed)
                iqr = np.quantile(self.statistic_values, 0.75, method=quantile_type) - \
                      np.quantile(self.statistic_values, 0.25, method=quantile_type)
                h = 0.9 * min(np.std(self.statistic_values), iqr / 1.34) * (self.n ** -0.2)
            else:
                h = sampling_args['width']
            if sampling_args['kernel'] == 'uniform':
                noise = np.random.uniform(-h, h, input_shape)
            elif sampling_args['kernel'] == 'norm':
                noise = np.random.normal(0, h, input_shape)
            else:
                noise = np.zeros(input_shape)
                print(f"Unknown kernel: {sampling_args['kernel']}, using percentile method.")

            self.evaluate_statistic(noise)
            return np.quantile(self.statistic_values_noise, quantiles, method=quantile_type)

        elif method == 'double':
            # get percentiles of original value in inner bootstrap samples
            nested_btsp_values = self.nested_bootstrap(self.b)
            sample_quantiles = np.mean(nested_btsp_values < self.original_statistic_value, axis=1)

            if side == 'two':
                t = abs(0.5 - sample_quantiles)     # change quantiles to one parameter to get symmetric interval
                t_quantile = np.quantile(t, coverages)
                new_quantiles = [0.5 - t_quantile, 0.5 + t_quantile]
            else:
                new_quantiles = np.quantile(sample_quantiles, quantiles, method=quantile_type)
            # TODO kasneje: ta coverage iz drugih metod, ne samo percentile - lahko naredimo cel bootstrap iterativen?
            # TODO iterative bootstrap (možno več iteracij)
            # print('New quantiles: ', new_quantiles, f'({quantile})')
            return np.quantile(self.statistic_values, new_quantiles, method=quantile_type)

        else:
            raise ValueError(f'This method is not supported, choose between {self.implemented_methods}.')

    def studentized_error_calculation(self):
        # TODO
        #  a bi blo bols dat to funkcijo ven
        #  a je okej da je nested, kaj so druge opcije? lahko delta method
        #  pomojem bi blo kul nastimat, da uporabnik sam poda, ce noce nested delat...
        #  paralelizacija

        # NESTED BOOTSTRAP:
        nested_btsp_values = self.nested_bootstrap(self.b)
        standard_errors = np.std(nested_btsp_values, axis=1)
        if 0 in standard_errors:
            standard_errors = [s if s != 0 else np.nan for s in standard_errors]

        return standard_errors

    def nested_bootstrap(self, b_inner):

        if self.b >= 500 or b_inner >= 500 or self.n > 100 or self.use_jit:
            stat_njit = {'mean': wrapped_mean, 'median': wrapped_median, 'std': wrapped_std,
                 'percentile_5': wrapped_percentile_5, 'percentile_95': wrapped_percentile_95,
                 'corr': wrapped_corr}[self.statistic.__name__]
            new_values = nested_bootstrap_jit(self.b, b_inner, self.n, self.original_sample, stat_njit)
        else:
            new_values = np.zeros([self.b, b_inner])
            for i in range(self.b):
                new_indices = np.random.choice(self.bootstrap_indices[i, :], size=[b_inner, self.n])
                for j in range(b_inner):
                    new_values[i, j] = self.statistic(self.original_sample[new_indices[j, :]])

        return new_values

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


# NUMBA functions, used to significantly speed up nested bootstrap calculation
@njit()
def nested_bootstrap_jit(b, b_inner, n, original_sample, statistic):
    bootstrap_indices = np.random.choice(np.arange(n), size=(b, n))
    new_values = np.zeros((b, b_inner))
    for i in range(b):
        new_indices = np.random.choice(bootstrap_indices[i, :], size=(b_inner, n))
        for j in range(b_inner):
            new_values[i, j] = statistic(original_sample[new_indices[j, :]])

    return new_values


@njit()
def wrapped_median(val):
    return np.median(val)


@njit()
def wrapped_mean(val):
    return np.mean(val)


@njit()
def wrapped_std(val):
    return np.std(val)


@njit()
def wrapped_corr(data):
    c = np.corrcoef(data, rowvar=False)
    return c[0, 1]


@njit()
def wrapped_percentile_5(data):
    # TODO: can't use median unbiased method in numba, implement it if needed
    return np.quantile(data, 0.05)  # , method='median_unbiased')


@njit()
def wrapped_percentile_95(data):
    return np.quantile(data, 0.95)  # , method='median_unbiased')



if __name__ == '__main__':

    b = Bootstrap(np.random.normal(0, 1, size=10), statistic=np.mean,
                  group_indices=[[[0, 1], [2, 3, 4]], [[5, 6], [7, 8, 9]]])

    b.sample(10, 1, sampling='hierarchical', sampling_args={'method': 'cases', 'strategy': [False, True, True]})

    print(b.bootstrap_indices)