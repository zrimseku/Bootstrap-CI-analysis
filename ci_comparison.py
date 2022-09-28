import itertools
import os
import time
from itertools import repeat
from multiprocessing import Pool

import numpy as np
import pandas as pd
import scipy
from tqdm import tqdm
from scipy.stats import norm
from scipy.stats import bootstrap as boot_sci
from arch.bootstrap import IIDBootstrap as boot_arch
import matplotlib.pyplot as plt
from collections import defaultdict
import seaborn as sns
import matplotlib.pyplot as plt

from ci_methods import Bootstrap
from generators import DGP, DGPNorm, DGPExp, DGPBeta, DGPBiNorm, DGPLogNorm, DGPLaplace, DGPBernoulli, DGPCategorical, \
    DGPRandEff
from R_functions import psignrank_range

# TODO set correct R folder
# os.environ['R_HOME'] = "C:/Users/ursau/AppData/Local/Programs/Anaconda/envs/bootstrap/lib/R"        # doma
os.environ['R_HOME'] = "C:/Anaconda3/envs/bootstrapci/lib/R"                                        # lab

import rpy2
import rpy2.robjects as robjects
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()


class CompareIntervals:

    def __init__(self, statistic: callable, methods: list[str], data_generator: DGP, n: int, b: int,
                 alphas: list[float], quantile_type='median_unbiased', use_jit: bool = True,
                 sampling: str = 'nonparametric', sampling_args_to_compare=None):
        self.statistic = statistic
        self.methods = methods
        self.methods_hierarchical = []
        self.dgp = data_generator
        self.n = n
        self.b = b
        self.alphas = np.array(alphas)  # we are interested in one sided intervals, two sided can be computed from them
        self.quantile_type = quantile_type
        self.computed_intervals = {m: defaultdict(list) for m in methods}       # will add all computed intervals
        self.inverse_cdf = {}
        self.times = defaultdict(list)
        self.coverages = {}
        self.distances_from_exact = {}
        self.lengths = {}
        self.use_jit = use_jit
        self.sampling = sampling
        self.sampling_args = sampling_args_to_compare
        max_groups_l1 = np.random.randint(2, min(5, int(self.n / 2) + 1))  # TODO smarter selection, 3 levels
        self.max_group_sizes = [max_groups_l1, 2 * self.n / max_groups_l1]

    def compute_bootstrap_intervals(self, data: np.array):
        # initialize and sample so we will have the same bootstrap samples for all bootstrap methods
        t = time.time()
        bts = Bootstrap(data, self.statistic, self.use_jit)
        bts.sample(self.b)
        bts.evaluate_statistic()
        ts = time.time() - t            # time needed for sampling (will add it to all methods)
        for method in self.methods:
            if method not in bts.implemented_methods:
                # skip non-bootstrap methods during calculation of bootstrap CI
                continue
            t = time.time()
            cis = bts.ci(coverages=self.alphas, side='one', method=method, quantile_type=self.quantile_type)
            for a, ci in zip(self.alphas, cis):
                self.computed_intervals[method][a].append(ci)
            self.times[method].append(time.time() - t + ts)
            # print('finished', method)
        return bts

    def compute_bootstrap_intervals_hierarchical(self, data: np.array, group_indices: list):
        btss = []
        for sampling_method in ['cases', 'random-effect']:
            for strategy in itertools.product([0, 1], repeat=len(self.dgp.stds)):
                strategy_bool = [bool(i) for i in strategy]
                strategy_str = '_' + ''.join([str(s) for s in strategy]) if sampling_method == 'cases' else ''
                if any(strategy):
                    if sampling_method == 'random-effect':
                        # only run random-effect bootstrap once
                        continue
                else:
                    if sampling_method == 'cases':
                        # skip if we sample without replacement on all levels, as this would just be original set
                        continue
                t = time.time()
                bts = Bootstrap(data, self.statistic, self.use_jit, group_indices=group_indices)
                bts.sample(self.b, sampling='hierarchical',
                           sampling_args={'method': sampling_method, 'strategy': strategy_bool})
                bts.evaluate_statistic(sampling='hierarchical',  sampling_args={'method': sampling_method,
                                                                                'strategy': strategy_bool})
                ts = time.time() - t            # time needed for sampling (will add it to all methods)
                for method in self.methods:
                    if method not in bts.implemented_methods:
                        # skip non-bootstrap methods during calculation of bootstrap CI
                        continue
                    method_str = sampling_method + strategy_str + '_' + method
                    if method_str not in self.methods_hierarchical:
                        self.methods_hierarchical.append(method_str)
                    t = time.time()
                    cis = bts.ci(coverages=self.alphas, side='one', method=method, quantile_type=self.quantile_type,
                                 sampling='hierarchical')
                    if method_str not in self.computed_intervals:
                        self.computed_intervals[method_str] = defaultdict(list)
                    for a, ci in zip(self.alphas, cis):
                        self.computed_intervals[method_str][a].append(ci)
                    self.times[method_str].append(time.time() - t + ts)
                    # print('finished', method_str)
                btss.append(bts)
        return btss

    def compute_non_bootstrap_intervals(self, data: np.array):
        """
        Computes CI with non-bootstrap methods, that can be applied to the statistic in use.
        :param data: array containing one sample
        """
        ci = defaultdict(list)
        new_methods = {'mean': ['wilcoxon', 'ttest'],
                       'median': ['wilcoxon', 'ci_quant_param', 'ci_quant_nonparam', 'maritz-jarrett'],
                       'std': ['chi_sq'], 'percentile': ['ci_quant_param', 'ci_quant_nonparam', 'maritz-jarrett'],
                       'corr': ['ci_corr_pearson', 'ci_corr_spearman']}
        if self.statistic.__name__[:10] not in ['mean', 'median', 'std', 'percentile', 'corr']:
            print(f'No known non-bootstrap methods to use for statistic {self.statistic.__name__}.')
            new_methods[self.statistic.__name__] = []

        for method in new_methods[self.statistic.__name__[:10]]:
            if method == 'ttest':
                t = time.time()
                stat = self.statistic(data)
                se = np.std(data) / np.sqrt(self.n)
                ci[method] = scipy.stats.t.ppf(self.alphas, df=self.n - 1, loc=stat, scale=se)
                self.times[method].append(time.time() - t)

            elif method == 'wilcoxon':
                t = time.time()
                for a in self.alphas:
                    f = robjects.r('''f <- function(data, a) {
                                        res <- wilcox.test(data, conf.int = T, conf.level = a, alternative='less')
                                        b <- lapply(res, attributes)$conf.int$conf.level[1]
                                        list(ci=res$conf.int[2], cl=b)
                                        }
                                   ''')
                    r_f = robjects.globalenv['f']
                    ci_r, achieved_a = r_f(data, a)

                    if abs(achieved_a[0] - a) > a/2:        # R criteria for saying that conf level is not achievable
                        # TODO get correct criteria, decide what to do
                        ci_r[0] = np.nan
                    ci[method].append(ci_r[0])
                self.times[method].append(time.time() - t)

            elif method in ['ci_quant_param', 'ci_quant_nonparam']:
                t = time.time()
                quant = 0.5 if self.statistic.__name__ == 'median' else int(self.statistic.__name__.split('_')[1])/100

                if method == 'ci_quant_param':
                    m = np.mean(data)
                    s = np.std(data)
                    z = (np.quantile(data, quant, method=self.quantile_type) - m) / s
                    nc = -z * np.sqrt(self.n)
                    ci[method] = m - scipy.stats.nct.ppf(1 - self.alphas, nc=nc, df=self.n - 1) * s / np.sqrt(self.n)

                elif method == 'ci_quant_nonparam':
                    t = time.time()
                    sorted_data = np.array(sorted(data) + [np.nan])
                    ci[method] = sorted_data[scipy.stats.binom.ppf(self.alphas, self.n, quant).astype(int)]

                self.times[method].append(time.time() - t)

            elif method == 'maritz-jarrett':
                t = time.time()
                quant = 0.5 if self.statistic.__name__ == 'median' else int(self.statistic.__name__.split('_')[1])/100
                # can't choose 0.5
                ci[method] = [scipy.stats.mstats.mquantiles_cimj(data, prob=quant, alpha=abs(2*a-1))[int(a > 0.5)][0]
                              for a in self.alphas]
                self.times[method].append(time.time() - t)

            elif method == 'chi_sq':
                t = time.time()
                s = np.std(data)
                qchisq = scipy.stats.chi2.ppf(1 - self.alphas, self.n - 1)
                ci[method] = np.sqrt((self.n - 1) * s ** 2 / qchisq)
                self.times[method].append(time.time() - t)

            elif method[:7] == 'ci_corr':
                t = time.time()
                if method == 'ci_corr_pearson':
                    in1 = data[:, 0]
                    in2 = data[:, 1]
                elif method == 'ci_corr_spearman':
                    in1 = scipy.stats.rankdata(data[:, 0])
                    in2 = scipy.stats.rankdata(data[:, 1])

                res = scipy.stats.pearsonr(in1, in2, alternative='less')
                ci[method] = [res.confidence_interval(a).high for a in self.alphas]
                self.times[method].append(time.time() - t)

            else:
                raise ValueError('Wrong method!!')      # should never happen, just here as a code check

        for m in ci.keys():
            if m not in self.methods:
                self.methods.append(m)
            if m not in self.computed_intervals:
                self.computed_intervals[m] = defaultdict(list, {a: [c] for a, c in zip(self.alphas, ci[m])})
            else:
                for i in range(len(self.alphas)):
                    self.computed_intervals[m][self.alphas[i]].append(ci[m][i])

    def exact_interval_simulation(self, repetitions: int):
        true_val = self.dgp.get_true_value(self.statistic.__name__)

        if np.size(true_val) == 1:
            stat_values = np.empty(repetitions)
        else:
            stat_values = np.empty((repetitions, np.size(true_val)))

        if self.sampling == 'hierarchical':
            data = self.dgp.sample(max_group_sizes=self.max_group_sizes, nr_samples=repetitions)
        else:
            data = self.dgp.sample(sample_size=self.n, nr_samples=repetitions)

        for r in range(repetitions):
            stat_values[r] = self.statistic(data[r])

        distribution = stat_values - true_val

        # inverse cdf that is used for exact interval calculation
        self.inverse_cdf = {a: np.quantile(distribution, 1 - a, method=self.quantile_type) for a in self.alphas}

    def compare_intervals(self, repetitions, length=None):
        true_statistic_value = self.dgp.get_true_value(self.statistic.__name__)
        self.exact_interval_simulation(100000)

        stat_original = []

        if self.sampling == 'hierarchical':
            data = self.dgp.sample(max_group_sizes=self.max_group_sizes, nr_samples=repetitions)
            group_indices = self.dgp.group_indices
        else:
            data = self.dgp.sample(sample_size=self.n, nr_samples=repetitions)

        slow_methods = []                   # hack for speeding up in double or studentized method
        # calculation with different bootstrap methods
        for r in range(repetitions):
            # calculation with non-bootstrap methods
            # TODO include again after fixing DGP (hierarchical)
            self.compute_non_bootstrap_intervals(data[r])
            # calculation with different bootstrap methods
            if r == 10000:
                slow_methods = [m for m in self.methods if m in ['double', 'studentized']]
                print(f"Leaving out methods {slow_methods} during repetitions over 10000.")
                self.methods = [m for m in self.methods if m not in ['double', 'studentized']]

            if self.sampling == 'hierarchical':
                btss = self.compute_bootstrap_intervals_hierarchical(data[r], group_indices)
                stat_original.append(btss[0].original_statistic_value)      # original doesn't depend on sampling
            else:
                bts = self.compute_bootstrap_intervals(data[r])
                stat_original.append(bts.original_statistic_value)
        stat_original = np.array(stat_original)
        self.methods = self.methods + slow_methods

        # exact intervals
        self.computed_intervals['exact'] = {a: stat_original - self.inverse_cdf[a] for a in self.alphas}

        # compute coverages for all methods and alphas
        methods_to_compare = self.methods_hierarchical if self.sampling == 'hierarchical' else self.methods
        self.coverages = {method: {alpha: np.mean(np.array(self.computed_intervals[method][alpha][-repetitions:]) >
                                                  true_statistic_value) for alpha in self.alphas}
                          for method in methods_to_compare}

        self.distances_from_exact = {method: {alpha: np.array(self.computed_intervals[method][alpha][-repetitions:]) -
                                                     self.computed_intervals['exact'][alpha] for alpha in self.alphas}
                                     for method in methods_to_compare}

        if length is not None:
            low_alpha, high_alpha = [round((1 - length) / 2, 5), round((length + 1) / 2, 5)]
            if low_alpha not in self.alphas or high_alpha not in self.alphas:
                raise ValueError(f"Length of {length} CI can't be calculated, because we don't have calculations for "
                                 f"the corresponding alphas.")
        else:
            low_alpha, high_alpha = [min(self.alphas), max(self.alphas)]
            print(f'Calculating lengths of {round(high_alpha - low_alpha, 5)} CI, '
                  f'from {low_alpha} to {high_alpha} quantiles.')

        self.lengths = {method: np.array(self.computed_intervals[method][high_alpha][-repetitions:]) -
                                np.array(self.computed_intervals[method][low_alpha][-repetitions:])
                        for method in methods_to_compare}

        times_stats = {method: {'mean': np.mean(self.times[method]), 'std': np.std(self.times[method])}
                       for method in methods_to_compare}

        return self.coverages, times_stats

    def draw_intervals(self, alphas_to_draw: list[float], show=False):
        # only implemented for nonparametric sampling, I think we don't need it for hierarchical
        data = self.dgp.sample(sample_size=self.n)
        self.alphas = np.union1d(self.alphas, alphas_to_draw)

        # compute bootstrap intervals
        bts = self.compute_bootstrap_intervals(data)

        # compute non-bootstrap intervals
        self.compute_non_bootstrap_intervals(data)

        # print({m: [self.computed_intervals[m][a][-1] for a in alphas_to_draw] for m in self.methods})

        # exact intervals calculation
        if not np.array([a in self.inverse_cdf for a in alphas_to_draw]).all():
            self.exact_interval_simulation(10000)
        exact_intervals = [self.statistic(data) - self.inverse_cdf[a] for a in alphas_to_draw]

        # plotting
        colors = iter(plt.cm.jet(np.linspace(0.05, 0.95, len(self.methods))))
        plt.hist(bts.statistic_values, bins=30, label='statistic')
        if 'smoothed' in self.methods:
            if np.nan in bts.statistic_values_noise or np.inf in bts.statistic_values_noise:
                print('skipped drawing of smoothed values because of nan values.')      # TODO why are they here?
            else:
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

        ci = round((alphas_to_draw[1] - alphas_to_draw[0]) * 100)
        plt.title(f'{ci}CI for {self.statistic.__name__} of {type(self.dgp).__name__} (n = {self.n}, B = {self.b})')
        plt.legend()
        if show:
            plt.show()
        else:
            plt.savefig(f'images/{ci}CI_{self.statistic.__name__}_{type(self.dgp).__name__}_n{self.n}_B{self.b}.png')
            plt.close()

    def plot_results(self, repetitions, length=0.9, show=False):
        res = self.compare_intervals(repetitions, length)

        folder = 'images_hierarchical' if self.sampling == 'hierarchical' else 'images'

        title = f'{self.statistic.__name__} of {type(self.dgp).__name__} (n = {self.n}, B = {self.b})'

        # building long dataframes for seaborn use
        methods_to_use = self.methods_hierarchical if self.sampling == 'hierarchical' else self.methods
        cov_methods = list(methods_to_use)*len(self.alphas)
        cov_alphas = np.repeat(self.alphas, len(methods_to_use))
        coverage_df = pd.DataFrame({'method': cov_methods, 'alpha': cov_alphas,
                                    'coverage': [self.coverages[m][a] for m, a in zip(cov_methods, cov_alphas)]})

        sns.barplot(x="alpha", hue="method", y="coverage", data=coverage_df, hue_order=methods_to_use)

        plt.axhline(y=self.alphas[0], xmin=0, xmax=1 / len(self.alphas), linestyle='--', color='gray', label='expected')
        for i in range(1, len(self.alphas)):
            plt.axhline(y=self.alphas[i], xmin=i/len(self.alphas), xmax=(i + 1)/len(self.alphas), linestyle='--',
                        color='gray')

        plt.legend(loc='upper left')
        plt.title('True coverages of ' + title)
        if show:
            plt.show()
        else:
            plt.savefig(f'{folder}/coverages_{self.statistic.__name__}_{type(self.dgp).__name__}_n{self.n}_B{self.b}.png')
            plt.close()

        df_distance = pd.DataFrame({'method': np.repeat(cov_methods, repetitions),
                                    'alpha': np.repeat(cov_alphas, repetitions),
                                    'distance': np.concatenate([self.distances_from_exact[m][a][-repetitions:]
                                                                for m, a in zip(cov_methods, cov_alphas)])})

        plt.figure(figsize=(15, 6))

        # if np.inf in df_distance.values:      TODO inf vals in 0.975 for ci_quant_nonparam, change this??
        #     print()

        sns.boxplot(x="alpha", hue="method", y="distance", data=df_distance, hue_order=methods_to_use)
        plt.axhline(y=0, linestyle='--', color='gray')

        plt.legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)
        plt.title('Distance from exact intervals for' + title)
        plt.tight_layout()
        if show:
            plt.show()
        else:
            plt.savefig(f'{folder}/distance_{self.statistic.__name__}_{type(self.dgp).__name__}_n{self.n}_B{self.b}.png')
            plt.close()

        df_length = pd.DataFrame({m: v[-repetitions:] for m, v in zip(self.lengths.keys(), self.lengths.values())})
        df_times = pd.DataFrame({m: v[-repetitions:] for m, v in zip(self.times.keys(), self.times.values())})

        ax = sns.boxplot(data=df_length, order=methods_to_use)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
        plt.title(f'Lengths of {length} CI for ' + title)
        plt.tight_layout()
        if show:
            plt.show()
        else:
            plt.savefig(f'{folder}/length{int(length*100)}_{self.statistic.__name__}_{type(self.dgp).__name__}_n{self.n}_'
                        f'B{self.b}.png')
            plt.close()

        # times are for all alphas combined TODO: divide them?
        ax = sns.boxplot(data=df_times, order=methods_to_use)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
        plt.title('Calculation times for ' + title)
        plt.tight_layout()
        if show:
            plt.show()
        else:
            plt.savefig(f'{folder}/times_{self.statistic.__name__}_{type(self.dgp).__name__}_n{self.n}_B{self.b}.png')
            plt.close()

        true_val = self.dgp.get_true_value(self.statistic.__name__)
        df_intervals = pd.DataFrame([{'method': m, 'alpha': a, 'predicted': v, 'true_value': true_val}
                                     for m in self.computed_intervals.keys() for a in self.computed_intervals[m].keys()
                                     for v in self.computed_intervals[m][a]])

        return res, coverage_df, df_length, df_times, df_distance, df_intervals


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
        our_ci = our.ci(coverages=[alpha], nr_bootstrap_samples=B, method=method)  # , seed=0)
        # our.plot_bootstrap_distribution()
        print(f'Our: {our_ci}')

        print('_____________________________________________________________________')


def run_comparison(dgps, statistics, ns, Bs, methods, alphas, repetitions, alphas_to_draw=[0.05, 0.95], length=0.9,
                   append=True, nr_processes=24, dont_repeat=False, sampling='nonparametric'):

    names = ['coverage', 'length', 'times', 'distance', 'intervals']
    if sampling == 'nonparametric':
        all_methods = ['percentile', 'basic', 'bca', 'bc', 'standard',  'smoothed', 'double', 'studentized', 'ttest',
                       'wilcoxon', 'ci_quant_param', 'ci_quant_nonparam', 'maritz-jarrett', 'chi_sq', 'ci_corr_pearson',
                       'ci_corr_spearman']
    else:
        # TODO spremeni ko veš prave metode, ki se bodo uporabljale -> generiraj glede na level?
        all_methods = ['cases_01_percentile', 'cases_01_bc', 'cases_10_percentile', 'cases_10_bc',
                       'cases_11_percentile', 'cases_11_bc', 'random-effect_percentile', 'random-effect_bc']
    cols = {'coverage': ['method', 'alpha', 'coverage', 'dgp', 'statistic', 'n', 'B', 'repetitions'],
            'length': ['CI', 'dgp', 'statistic', 'n', 'B', 'repetitions'] + all_methods,
            'distance': ['method', 'alpha', 'distance', 'dgp', 'statistic', 'n', 'B', 'repetitions'],
            'times': ['dgp', 'statistic', 'n', 'B', 'repetitions'] + all_methods,
            'intervals': ['method', 'alpha', 'predicted', 'true_value', 'dgp', 'statistic', 'n', 'B', 'repetitions']}

    folder = 'results_hierarchical' if sampling == 'hierarchical' else 'results'

    if not append:
        # need to use this (append=False) for running first time to set header!!
        print('Will delete all results - ARE YOU SURE???')
        time.sleep(60)
        for name in names:
            pd.DataFrame(columns=cols[name]).to_csv(f'{folder}/' + name + '.csv', index=False)

    if dont_repeat:
        cov = pd.read_csv(f'{folder}/coverage.csv')

    params = []
    nr_skipped = 0
    for dgp in dgps:
        for statistic in statistics:
            if (statistic.__name__ == 'corr' and type(dgp).__name__ != 'DGPBiNorm') or \
                    (type(dgp).__name__ == 'DGPBiNorm' and statistic.__name__ != 'corr') or \
                    (type(dgp).__name__ == 'DGPCategorical' and statistic.__name__ == 'std'):
                continue
            for n in ns:
                for B in Bs:
                    # not needed anymore because of numba?
                    # if B > 1000:
                    #     methods_par = [m for m in methods if m not in ['double', 'studentized']]
                    # else:

                    if dont_repeat:
                        same_pars = cov[(cov['dgp'] == dgp.describe()) & (cov['statistic'] == statistic.__name__) &
                                        (cov['n'] == n) & (cov['B'] == B) & (cov['repetitions'] == repetitions)]
                        if same_pars.shape[0] > 0:
                            # TODO: check if we have calculations for all methods and alphas if needed
                            nr_skipped += 1
                            continue
                        else:
                            print('Adding: ', dgp.describe(), statistic.__name__, n, B, repetitions)

                    methods_par = methods.copy()

                    params.append((statistic, methods_par, dgp, n, B, alphas.copy()))

    if dont_repeat:
        print(f'Skipped {nr_skipped} combinations, because we already have results.')

    pool = Pool(processes=nr_processes)
    for dfs in tqdm(pool.imap_unordered(multiprocess_run_function, zip(params, repeat(repetitions), repeat(length),
                                                                       repeat(alphas_to_draw), repeat(sampling)),
                                        chunksize=1), total=len(params)):
        for i in range(len(dfs)):

            dfs[i] = pd.concat([pd.DataFrame(columns=cols[names[i]]), dfs[i]])      # setting right order of columns

            dfs[i].to_csv(f'{folder}/{names[i]}.csv', header=False, mode='a', index=False)


def multiprocess_run_function(param_tuple):
    pars, repetitions, length, alphas_to_draw, sampling = param_tuple
    statistic, methods, dgp, n, B, alphas = pars
    use_jit = (repetitions >= 100)
    comparison = CompareIntervals(*pars, use_jit=use_jit, sampling=sampling)
    _, coverage_df, df_length, df_times, df_distance, df_intervals = comparison.plot_results(repetitions=repetitions,
                                                                                             length=length)
    dfs = [coverage_df, df_length, df_times, df_distance, df_intervals]

    if sampling != 'hierarchical' and dgp.describe()[:9] != 'DGPBiNorm':
        # 3D histogram not implemented and hierarchical data are not properly shown on histogram
        comparison.draw_intervals(alphas_to_draw)
    df_length['CI'] = length
    for i in range(len(dfs)):
        dfs[i]['dgp'] = dgp.describe()
        dfs[i]['statistic'] = statistic.__name__
        dfs[i]['n'] = n
        dfs[i]['B'] = B
        dfs[i]['repetitions'] = repetitions

    return dfs


def corr(data):
    c = np.corrcoef(data, rowvar=False)
    return c[0, 1]


def percentile_5(data):
    return np.quantile(data, 0.05, method='median_unbiased')


def percentile_95(data):
    return np.quantile(data, 0.95, method='median_unbiased')


if __name__ == '__main__':

    # warnings.showwarning = warn_with_traceback

    # statistic = np.median
    # n = 10
    # B = 2000
    # alpha = 0.9
    seed = 0
    alphas = [0.025, 0.05, 0.25, 0.75, 0.95, 0.975]
    methods = ['percentile', 'bc']
    # methods = ['percentile', 'basic', 'bca', 'bc', 'standard', 'smoothed', 'double', 'studentized']

    dgps = [DGPNorm(seed, 0, 1), DGPExp(seed, 1), DGPBeta(seed, 1, 1), DGPBeta(seed, 10, 2), DGPBernoulli(seed, 0.5),
            DGPBernoulli(seed, 0.95), DGPLaplace(seed, 0, 1), DGPLogNorm(seed, 0, 1),
            DGPBiNorm(seed, np.array([1, 1]), np.array([[2, 0.5], [0.5, 1]]))]
    # dgps = [DGPNorm(seed, 0, 1)]
    statistics = [np.mean, np.median, np.std, percentile_5, percentile_95, corr]
    # statistics = [np.mean, np.median]

    ns = [4, 8, 16, 32, 64, 128, 256]
    Bs = [10, 100, 1000]
    repetitions = 10000
    run_comparison(dgps, statistics, ns, Bs, methods, alphas, repetitions, nr_processes=24, dont_repeat=True,
                   append=False)

