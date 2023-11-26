import numpy as np
import scipy as sp
import pandas as pd

from collections import defaultdict


def compare_variances():
    intervals = pd.read_csv('results_hierarchical/intervals.csv')
    intervals['var_ratio'] = intervals['mean_var'] / intervals['gt_variance']

    intervals = intervals[intervals['method'] != 'exact']
    intervals['strat'] = intervals['method'].apply(lambda x: x.split('_')[1])
    res_int = intervals[['strat', 'var_ratio']].groupby('strat').mean().sort_values(by='var_ratio')

    coverage = pd.read_csv('results_hierarchical/coverage.csv')
    # cov1 = cov[::2]
    coverage['strat'] = coverage['method'].apply(lambda x: x.split('_')[1])
    res_cov = coverage[['strat', 'var_coverage']].groupby('strat').mean().sort_values(by='var_coverage')

    return res_int, res_cov


def kl(p, q):
    """K-L divergence."""
    part1 = np.where(p == 0, 0, p * np.log2(p / q))
    part2 = np.where(p == 1, 0, (1 - p) * np.log2((1 - p) / (1 - q)))
    return part1 + part2


def aggregate_results(result_folder, methods=None, combined_with='mean', withnans=True, onlybts=True, r=10000):
    # reading and filtering coverage table
    results = pd.read_csv(f'{result_folder}/results_from_intervals_{combined_with}{["", "_bts"][int(onlybts)]}'
                          f'{["", "_withnans"][int(withnans)]}.csv')

    # coverage = coverage[coverage['B'] == 1000]  # if you want to filter results
    if onlybts:
        if methods is None:
            methods = ['percentile', 'standard', 'basic', 'bc', 'bca', 'double', 'smoothed', 'studentized']
        results = results[results['method'].isin(methods)]

    # calculations for table of closeness to the best method
    results['difference'] = results['coverage'] - results['alpha']
    results['abs_difference'] = abs(results['difference'])
    min_distances = results[['alpha', 'coverage', 'abs_difference', 'dgp', 'statistic', 'n']] \
        .sort_values('abs_difference').groupby(['alpha', 'dgp', 'statistic', 'n']).first()
    results['min_difference'] = results.apply(
        lambda row: min_distances.loc[row['alpha'], row['dgp'], row['statistic'], row['n']]['abs_difference'], axis=1)
    results['best_coverage'] = results.apply(
        lambda row: min_distances.loc[row['alpha'], row['dgp'], row['statistic'], row['n']]['coverage'], axis=1)
    results['std'] = np.sqrt(results['best_coverage'] * (1 - results['best_coverage']) / results['repetitions'])
    results['near_best'] = abs(results['best_coverage'] - results['coverage']) < results['std']

    # calculation for KL-divergence
    results['kl_div'] = kl(results['coverage'], results['alpha'])

    # calculation for ranks
    results['rank'] = results[['alpha', 'abs_difference', 'dgp', 'statistic', 'n']].groupby(
        ['alpha', 'dgp', 'statistic', 'n']).rank()
    results['rank_kl'] = results[['alpha', 'kl_div', 'dgp', 'statistic', 'n']].groupby(
        ['alpha', 'dgp', 'statistic', 'n']).rank()

    def agregate_n_stat(df, column, fun, asc=True):
        agg = fun(df[['method', column]].groupby(['method']))

        agg_n = fun(results[['method', column, 'n']].groupby(['method', 'n'])).unstack()
        agg_n.columns = agg_n.columns.droplevel()
        agg = agg.join(agg_n)

        agg_stat = fun(results[['method', column, 'statistic']].groupby(['method', 'statistic'])).unstack()
        agg_stat.columns = agg_stat.columns.droplevel()
        agg = agg.join(agg_stat).sort_values(by=column, ascending=asc)

        return agg

    # tables
    # first comparison by near best in avg rank, distance, nans
    near_best = agregate_n_stat(results, 'near_best', lambda x: x.sum(), asc=False)
    avg_rank = agregate_n_stat(results, 'rank', lambda x: x.mean())

    dist_table = agregate_n_stat(results, 'avg_distance', lambda x: x.median())
    nans = agregate_n_stat(results, 'nans', lambda x: x.mean())

    nans_a = results[['method', 'nans', 'alpha']].groupby(['method', 'alpha']).mean().unstack()
    nans_a.columns = nans_a.columns.droplevel()
    nans = nans.join(nans_a)

    nans = nans[nans['nans'] > 0].sort_values(by='nans', ascending=False)

    # normalization
    for m in avg_rank.index:
        near_best.loc[m, 'near_best'] /= results[results['method'] == m].shape[0]
        for n in results['n'].unique():
            near_best.loc[m, n] /= results[(results['method'] == m) & (results['n'] == n)].shape[0]
        for stat in results['statistic'].unique():
            near_best.loc[m, stat] /= results[(results['method'] == m) & (results['statistic'] == stat)].shape[0]

    # check for differences in method ranking
    results['rank_dist'] = results[['alpha', 'avg_distance', 'dgp', 'statistic', 'n']].groupby(
        ['alpha', 'dgp', 'statistic', 'n']).rank()

    # rank diff -> mean, median, kok kerih eksperimentov jih je med tistimi k majo > n razliko

    # gather all nans
    nans_all = results[results['method'] == 'ci_quant_nonparam'][['method', 'nans', 'statistic', 'repetitions',
                                                                  'n', 'alpha']].groupby(['method', 'statistic',
                                                                                          'n', 'alpha']).mean()
    nans_all = nans_all[nans_all['nans'] > 0]['nans'].sort_values(ascending=False)

    # t = results[results['method'].isin(['double', 'standard'])] for finding experiment for histogram

    # current comparison by kl div
    kl_div = agregate_n_stat(results, 'kl_div', lambda x: x.mean())
    kl_div_se = agregate_n_stat(results, 'kl_div', lambda x: x.sem())
    kl_div_med = agregate_n_stat(results, 'kl_div', lambda x: x.median())
    kl_div_rank = agregate_n_stat(results, 'rank_kl', lambda x: x.mean())
    kl_rank_se = agregate_n_stat(results, 'rank_kl', lambda x: x.sem())

    def significantly_worse(df, se_df, values=False, experiment_se=None):
        # subtracting method differences and standard errors, to know which are significantly worse (positive ones)
        idx_min = df.idxmin()
        df_significant = df - df.min() - se_df
        for col in df_significant.columns:
            # subtracting se of the best method
            df_significant[col] -= se_df.loc[idx_min[col], col]
        df_significant = df_significant.loc[df.index, :]

        return df_significant if values else (df_significant >= 0)

    kl_div_significant = significantly_worse(kl_div, kl_div_se, experiment_se='kl')
    kl_rank_significant = significantly_worse(kl_div_rank, kl_rank_se, experiment_se='rank')

    return nans_all, kl_div, kl_div_med, kl_div_rank, kl_div_se, kl_rank_se, kl_div_significant, kl_rank_significant


def results_from_intervals(folder, combine_dist=np.mean, include_nan_repetitions=False, only_bts=True):
    # for wide tables (new results), skipping just replications that have nans
    # we are reading results line by line because of possibility of very large dataframes that can't fit in memory
    dist_dict = defaultdict(list)
    nans = defaultdict(int)
    coverage_dict = defaultdict(lambda: [0, 0])
    stds = {}
    bts_methods = ['percentile', 'standard', 'basic', 'bc', 'bca', 'double', 'smoothed', 'studentized']
    stat_methods = {'mean': bts_methods + ['wilcoxon', 'ttest'],
                    'median': bts_methods + ['wilcoxon', 'ci_quant_param', 'ci_quant_nonparam', 'maritz-jarrett'],
                    'std': bts_methods + ['chi_sq'],
                    'percentile': bts_methods + ['ci_quant_param', 'ci_quant_nonparam', 'maritz-jarrett'],
                    'corr': bts_methods + ['ci_corr_pearson']}
    with open(f'{folder}/intervals.csv') as f:
        keys = f.readline().strip('\n').split(',')  # header
        for line in f:
            line_dict = dict(zip(keys, line.strip('\n').split(',')))
            alpha, dgp, statistic, n, B, repetitions, true_val, exact = [line_dict[name] for name in
                                                                         ['alpha', 'dgp', 'statistic', 'n', 'B',
                                                                          'repetitions', 'true_value', 'exact']]
            alpha, n, B, repetitions, true_val, exact = float(alpha), int(n), int(B), int(repetitions), \
                                                        float(true_val), float(exact)
            if B != 1000:
                # add if you want to skip any results
                continue

            any_nan = False
            all_methods = bts_methods if only_bts else stat_methods[line_dict['statistic'][:10]]
            for method in all_methods:
                if line_dict[method] == '':
                    any_nan = True
                    nans[(method, alpha, dgp, statistic, n, repetitions)] += 1

                if (method, alpha, dgp, statistic, n, repetitions) not in dist_dict:
                    # add missing keys for dataframe iteration
                    dist_dict[(method, alpha, dgp, statistic, n, repetitions)] = []

            if (not any_nan) or include_nan_repetitions:
                for method in stat_methods[line_dict['statistic'][:10]]:
                    pred = line_dict[method]
                    if pred != '':
                        pred = float(pred)
                        dist_dict[(method, alpha, dgp, statistic, n, repetitions)].append(abs(pred - exact))
                        coverage_dict[(method, alpha, dgp, statistic, n, repetitions)][0] += int(pred >= true_val)
                        coverage_dict[(method, alpha, dgp, statistic, n, repetitions)][1] += 1

                        # aggregating results as soon as we can to save memory
                        if len(dist_dict[(method, alpha, dgp, statistic, n, repetitions)]) == repetitions:
                            # change list of distances with mean/median distance
                            stds[(method, alpha, dgp, statistic, n, repetitions)] = np.std(
                                dist_dict[(method, alpha, dgp, statistic, n, repetitions)])
                            dist_dict[(method, alpha, dgp, statistic, n, repetitions)] = combine_dist(
                                dist_dict[(method, alpha, dgp, statistic, n, repetitions)])

    results = pd.DataFrame(columns=['method', 'alpha', 'dgp', 'statistic', 'n', 'repetitions', 'avg_distance',
                                    'std', 'coverage', 'nans'])
    for i, experiment in enumerate(dist_dict.keys()):
        distances = dist_dict[experiment]
        covers, count = coverage_dict[experiment]
        if type(distances) == list:
            if len(distances) == 0:
                avg_dist = np.nan
                std = np.nan
            else:
                avg_dist = combine_dist(distances)
                std = np.std(distances)
        else:
            avg_dist = distances
            std = stds[experiment]

        results.loc[i] = [*experiment, avg_dist, std, np.nan if count == 0 else covers / count,
                          nans[experiment] / experiment[-1]]

    # normalization of distances based on the best method
    results['min_distance'] = results['avg_distance']
    results['min_distance'] = results[['alpha', 'dgp', 'statistic', 'n', 'min_distance']].groupby(
        ['alpha', 'dgp', 'statistic', 'n']).transform(lambda x: x.min())
    results['norm_distance'] = results[['alpha', 'dgp', 'statistic', 'n', 'avg_distance']].groupby(
        ['alpha', 'dgp', 'statistic', 'n']).transform(lambda x: x / x.min())

    results.to_csv(f'{folder}/results_from_intervals_{combine_dist.__name__}_wthmin_{["", "_bts"][int(only_bts)]}'
                   f'{["", "_withnans"][int(include_nan_repetitions)]}.csv', index=False)
    return results


def analyze_length(lengthA, lengthB, len_dist='ld', better_coef=1.1):
    """Compares length of two sided intervals or distance to exact intervals (input absolute distances)"""

    diff = lengthA - lengthB
    diff_mu = np.mean(diff)
    diff_se = 0 if np.all(diff == diff[0]) else np.std(diff) / np.sqrt(len(lengthA))
    diff_is_neg_prob = sp.stats.norm.cdf(0, diff_mu, diff_se)

    normalized = (lengthA - lengthB) / (lengthA + 0.0001)       # TODO is this ok??
    norm_mu = np.mean(normalized)
    norm_se = 0 if np.all(normalized == normalized[0]) else np.std(normalized) / np.sqrt(len(lengthA))

    norm_is_neg_prob = 0 if norm_se == 0 else sp.stats.norm.cdf(0, norm_mu, norm_se)

    meanA = np.mean(np.abs(lengthA))
    meanB = np.mean(np.abs(lengthB))

    result = {
        f'{len_dist}_m1': meanA,
        f'{len_dist}_m2': meanB,
        'diff_mu': diff_mu,
        'diff_q025': diff_mu + sp.stats.norm.ppf(0.025) * diff_se,
        'diff_q975': diff_mu + sp.stats.norm.ppf(0.975) * diff_se,
        f'better_{len_dist}_prob': diff_is_neg_prob,                            # probability that method A is better
        'norm_mu': norm_mu,
        'norm_q025': norm_mu + sp.stats.norm.ppf(0.025) * norm_se,
        'norm_q975': norm_mu + sp.stats.norm.ppf(0.975) * norm_se,
        f'better_{len_dist}_prob_norm': norm_is_neg_prob,
        f'better_{len_dist}_m1': meanA * better_coef < meanB,
        f'better_{len_dist}_m2': meanB * better_coef < meanA
    }

    return result


def get_error_transformed_lo(true_coverage, alpha):
    """Transform error with logit function."""
    return np.abs(sp.special.logit(true_coverage) - sp.special.logit(alpha))


def are_equal_lo(cov1, cov2, alpha, base):
    """Methods are equal in logit transformed space."""
    return np.abs(get_error_transformed_lo(cov1, alpha) - get_error_transformed_lo(cov2, alpha)) <= base


def is_better_lo(cov1, cov2, alpha, base):
    """First method is better in logit transformed space."""
    return (get_error_transformed_lo(cov1, alpha) - get_error_transformed_lo(cov2, alpha)) < -base


def is_better_kl(cov1, cov2, alpha, base):
    """First method is better if we compare K-L divergences of their coverages."""
    return kl(cov1, alpha) - kl(cov2, alpha) < -base


def analyze_coverage(covers_m1, covers_m2, target_coverage, base=sp.special.logit(0.95) - sp.special.logit(0.94),
                     basekl=kl(0.94, 0.95)):
    """Analyzes coverages of both methods. Average and confidence intervals for both, then percentage of times that the
    first method is better than the second one."""
    y = np.zeros(4)
    y[0] = np.sum((covers_m1 == 0) & (covers_m2 == 0))
    y[1] = np.sum((covers_m1 == 1) & (covers_m2 == 0))
    y[2] = np.sum((covers_m1 == 0) & (covers_m2 == 1))
    y[3] = np.sum((covers_m1 == 1) & (covers_m2 == 1))

    if sum(y) != len(covers_m1):
        if sum(y) + np.sum(covers_m1 == np.nan) + np.sum(covers_m2 == np.nan) != len(covers_m1):
            print('missing cases in y!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        else:
            print('missing cases because of nans.')

    p = sp.stats.dirichlet.rvs(y + 0.0001, size=10000)

    coverage1 = p[:, 1] + p[:, 3]
    coverage2 = p[:, 2] + p[:, 3]

    error1 = np.abs(coverage1 - target_coverage)
    error2 = np.abs(coverage2 - target_coverage)

    error_diff = error1 - error2

    # without simulation
    coverage_m1 = np.mean(covers_m1)
    coverage_m2 = np.mean(covers_m2)

    return {
        'coverage_m1_mu': np.mean(coverage1),
        'coverage_m1_q025': np.percentile(coverage1, 2.5),
        'coverage_m1_q975': np.percentile(coverage1, 97.5),
        'coverage_m2_mu': np.mean(coverage2),
        'coverage_m2_q025': np.percentile(coverage2, 2.5),
        'coverage_m2_q975': np.percentile(coverage2, 97.5),
        'errordiff_mu': np.mean(error_diff),
        'errordiff_q025': np.percentile(error_diff, 2.5),
        'errordiff_q975': np.percentile(error_diff, 97.5),
        # method one is better based on simulation and linear error:
        'better_cov_prob': np.mean(error_diff <= 0),
        # method one is better based on simulation and logistic transformation of error:
        'better_prob_m1': np.mean(is_better_lo(coverage1, coverage2, target_coverage, base)),
        # method two is better based on simulation and logistic transformation of error:
        'better_prob_m2': np.mean(is_better_lo(coverage2, coverage1, target_coverage, base)),
        # method one is better based on smaller KL divergence of it's true coverage (no simulation):
        'm1_better_kl': is_better_kl(coverage_m1, coverage_m2, target_coverage, basekl),
        # method one is better based on smaller KL divergence of it's true coverage (no simulation):
        'm2_better_kl': is_better_kl(coverage_m2, coverage_m1, target_coverage, basekl)
        # method one is better based on smaller JS distance of it's true coverage (no simulation):
    }


def analyze_coverage_m1_false(coverage_m2, true_coverage):
    """Setting method one to False, if it returns nans, so that we get results for the second method."""
    coverage_m1 = np.array([False] * len(coverage_m2))
    result = analyze_coverage(coverage_m1, coverage_m2, true_coverage)
    result.update({'coverage_m1_mu': -1, 'coverage_m1_q025': -1, 'coverage_m1_q975': -1,
                   'errordiff_mu': np.nan, 'errordiff_q025': np.nan, 'errordiff_q975': np.nan,
                   'better_cov_prob': np.nan, 'better_prob_m1': np.nan, 'better_prob_m2': np.nan})
    return result


def one_vs_others(method_one, other_methods=None, one_sided=None, two_sided=None, result_folder='results', B=1000,
                  reps=10000, better_coef=1.5):
    if one_sided is None:
        one_sided = [0.025, 0.05, 0.25, 0.75, 0.95, 0.975]
    if two_sided is None:
        two_sided = [0.5, 0.9, 0.95]
    results_df = pd.read_csv(f'{result_folder}/intervals.csv')

    # choosing only results for certain B and repetitions
    results_df = results_df.loc[(results_df['B'] == B) &
                                (results_df['repetitions'] == reps)].drop(columns=['B', 'repetitions'])

    # selecting correct methods for comparison for each statistic
    compare_to = {}
    for statistic in results_df['statistic'].unique():
        stat_cols = results_df[results_df['statistic'] == statistic].dropna(axis=1, how='all').columns.tolist()
        if other_methods is None:
            compare_to[statistic] = [m for m in stat_cols if m not in [method_one, 'dgp', 'statistic', 'n', 'alpha',
                                                                       'true_value', 'exact']]
        else:
            compare_to[statistic] = [m for m in other_methods if m in stat_cols]

    # division of results by experiments, calculating coverage and length/distance
    results_better1 = []
    results_better2 = []

    for [dgp, stat, n], df in results_df.groupby(['dgp', 'statistic', 'n']):

        # one-sided intervals
        for alpha in one_sided:
            df_a = df.loc[df['alpha'] == alpha].drop(columns='alpha')

            # we will save percentage of times that a method is unable to give predictions
            nans1 = df_a[method_one].isna().mean()


            if nans1 == 1:      # if method_one was unable to give any predictions

                # compare to all other methods for this statistic
                for m2 in compare_to[stat]:
                    nans2 = df_a[m2].isna().mean()

                    exp_dict = {'method': m2, 'alpha': alpha, 'dgp': dgp, 'stat': stat, 'n': n, 'nan_perc_m1': nans1,
                                'nan_perc_m2': nans2}

                    if nans2 == 1:      # if m2 was unable to give any predictions
                        exp_dict.update({'coverage_m1_mu': -1, 'coverage_m1_q025': -1, 'coverage_m1_q975': -1,
                                         'coverage_m2_mu': -1, 'coverage_m2_q025': -1, 'coverage_m2_q975': -1})

                    else:                               # m2 gave some predictions
                        covers_m2 = (df_a[m2] > df_a['true_value']).values
                        res_cov = analyze_coverage_m1_false(covers_m2, alpha)
                        exp_dict.update(res_cov)

                    results_better1.append(exp_dict)

            else:                                       # method_one gave some predictions for this experiment

                covers_m1 = (df_a[method_one] > df_a['true_value']).values
                distance_m1 = abs(df_a[method_one] - df_a['exact']).values

                for m2 in compare_to[stat]:             # compare to all other methods
                    covers_m2 = (df_a[m2] > df_a['true_value']).values
                    distance_m2 = abs(df_a[m2] - df_a['exact']).values

                    res_cov = analyze_coverage(covers_m1, covers_m2, alpha)
                    res_dist = analyze_length(distance_m1, distance_m2, 'dist', better_coef=better_coef)

                    nans2 = df_a[m2].isna().mean()

                    exp_dict = {'method': m2, 'alpha': alpha, 'dgp': dgp, 'stat': stat, 'n': n, 'nan_perc_m1': nans1,
                                'nan_perc_m2': nans2}
                    exp_dict.update(res_cov)
                    exp_dict.update(res_dist)

                    results_better1.append(exp_dict)

        # two-sided intervals
        for alpha in two_sided:
            au = 0.5 + alpha / 2
            al = round(0.5 - alpha / 2., 10)

            # two tables for lower and upper bounds
            df_al = df.loc[df['alpha'] == al].drop(columns='alpha')
            df_au = df.loc[df['alpha'] == au].drop(columns='alpha')

            # we will save if the method was unable to give any predictions in this experiment
            has_nans1 = df_al[method_one].isna().any() or df_au[method_one].isna().any()

            # method_one didn't give any predictions for lower or any for upper bound
            if df_al[method_one].isna().all() or df_au[method_one].isna().all():

                for m2 in compare_to[stat]:                 # compare to all other methods

                    has_nans2 = df_al[m2].isna().any() or df_au[m2].isna().any()

                    exp_dict = {'method': m2, 'alpha': alpha, 'dgp': dgp, 'stat': stat, 'n': n,
                                'has_nans_m1': has_nans1, 'has_nans_m2': has_nans2}

                    if df_al[m2].isna().all() or df_au[m2].isna().all():      # m2 gave no predictions
                        exp_dict.update({'coverage_m1_mu': -1, 'coverage_m1_q025': -1, 'coverage_m1_q975': -1,
                                         'coverage_m2_mu': -1, 'coverage_m2_q025': -1, 'coverage_m2_q975': -1})

                    else:                                   # m2 gave some predictions
                        covers_m2 = (df_au[m2] > df_au['true_value']).values & (df_al[m2] < df_al['true_value']).values

                        res_cov = analyze_coverage_m1_false(covers_m2, alpha)
                        exp_dict.update(res_cov)

                    results_better2.append(exp_dict)

            else:                                           # method one gave some predictions

                covers_m1 = (df_au[method_one] > df_au['true_value']).values & \
                            (df_al[method_one] < df_al['true_value']).values
                length_m1 = df_au[method_one].values - df_al[method_one].values

                for m2 in compare_to[stat]:                 # compare to all other methods
                    covers_m2 = (df_au[m2] > df_au['true_value']).values & (df_al[m2] < df_al['true_value']).values
                    length_m2 = df_au[m2].values - df_al[m2].values

                    res_cov = analyze_coverage(covers_m1, covers_m2, alpha)
                    res_len = analyze_length(length_m1, length_m2, 'len', better_coef=better_coef)

                    has_nans2 = df_al[m2].isna().any() or df_au[m2].isna().any()

                    exp_dict = {'method': m2, 'alpha': alpha, 'dgp': dgp, 'stat': stat, 'n': n,
                                'has_nans_m1': has_nans1, 'has_nans_m2': has_nans2}
                    exp_dict.update(res_cov)
                    exp_dict.update(res_len)

                    results_better2.append(exp_dict)

    final_df1 = pd.DataFrame(results_better1)
    final_df1.sort_values(by='better_prob_m2', ascending=False)
    final_df1.to_csv(f'{result_folder}/onesided_{method_one}_vs_others_B{B}_reps_{reps}_d{int((better_coef-1)*100)}.csv',
                     index=False)

    final_df2 = pd.DataFrame(results_better2)
    final_df2.sort_values(by='better_prob_m2', ascending=False)
    final_df2.to_csv(f'{result_folder}/twosided_{method_one}_vs_others_B{B}_reps_{reps}_d{int((better_coef-1)*100)}.csv',
                     index=False)


def analyze_experiments(method_one, name, other_methods=None, result_folder='results', statistics=None,
                        ns=None, separate_dgp='coverage', sided='onesided'):
    # USING THIS ONE FOR STEP 2
    df = pd.read_csv(f'{result_folder}/{sided}_{method_one}_{name}_nonans.csv')

    if statistics is not None:
        df = df[df['stat'].isin(statistics)]

    if ns is not None:
        df = df[df['n'].isin(ns)]

    n2 ='dist' if sided == 'onesided' else 'len'

    df_bad_coverage = df[df['m2_better_kl']]
    bad_both = df[df['m2_better_kl'] | (((df['m2_better_kl'] + df['m1_better_kl']) < 1) & (df[f'better_{n2}_m2']))]

    print(f'There are {bad_both.shape[0]} cases where a method is better than {method_one}.'
          f'Out of them {bad_both[bad_both["m2_better_kl"]].shape[0]} have better coverage.')

    def compute_bad_s(df_bad):
        bad_s = pd.DataFrame()
        bad_s['nr_better'] = df_bad[['method', 'stat']].value_counts()
        bad_s['perc_better'] = bad_s['nr_better'] / df[['method', 'stat']].value_counts()
        bad_s['se_perc'] = bad_s['perc_better'] * (1 - bad_s['perc_better']) / \
                           np.sqrt(df[['method', 'stat']].value_counts())
        bad_s['nr_exp'] = df[['method', 'stat']].value_counts()
        bad_s = bad_s.sort_values('perc_better', ascending=False)
        for method, stat in bad_s.index:
            # for alpha in sorted(df_bad['alpha'].unique()):
            for dgp in sorted(df_bad['dgp'].unique()):
                bad_s.loc[(method, stat), dgp] = ', '.join(str(n) for n in
                                                           sorted(df_bad[(df_bad['method'] == method) &
                                                                         (df_bad['stat'] == stat) &
                                                                         (df_bad['dgp'] == dgp)]['n'].unique()))
        return bad_s

    bad_s = compute_bad_s(df_bad_coverage)

    results = {'coverage': bad_s}

    bad_dist = df[(((df['m2_better_kl'] + df['m1_better_kl']) < 1) & (df[f'better_{n2}_m2']))]

    if bad_dist.shape[0] > 0:

        results['distance'] = bad_s

    # choosing which results to observe when viewing separate dgp results
    if separate_dgp == 'coverage':
        bad_separate_dgp = df_bad_coverage
        results_mag = pd.DataFrame(results[separate_dgp]['perc_better'])
        mag_se = pd.DataFrame(results[separate_dgp]['se_perc'])
        mag_n_exp = pd.DataFrame(results[separate_dgp]['nr_exp'])
    elif separate_dgp == 'distance':
        bad_separate_dgp = bad_dist
        results_mag = pd.DataFrame(results[separate_dgp]['perc_better'])
        mag_se = pd.DataFrame(results[separate_dgp]['se_perc'])
        mag_n_exp = pd.DataFrame(results[separate_dgp]['nr_exp'])
    else:
        bad_separate_dgp = bad_both
        results_mag = pd.DataFrame()
        mag_se = pd.DataFrame()
        mag_n_exp = pd.DataFrame()

    for dgp in bad_separate_dgp['dgp'].unique():
        df_bad = bad_separate_dgp[bad_separate_dgp['dgp'] == dgp]

        bad_s = pd.DataFrame()
        bad_s['nr_better'] = df_bad[['method', 'stat']].value_counts()
        bad_s['perc_better'] = bad_s['nr_better'] / df[df['dgp'] == dgp][['method', 'stat']].value_counts()
        bad_s['se_perc'] = bad_s['perc_better'] * (1 - bad_s['perc_better']) / \
                           np.sqrt(df[df['dgp'] == dgp][['method', 'stat']].value_counts())
        bad_s['nr_exp'] = df[df['dgp'] == dgp][['method', 'stat']].value_counts()
        bad_s = bad_s.sort_values('perc_better', ascending=False)

        for method, stat in bad_s.index:
            for alpha in sorted(df_bad['alpha'].unique()):
                bad_s.loc[(method, stat), alpha] = ', '.join(str(n) for n in
                                                           sorted(df_bad[(df_bad['method'] == method) &
                                                                         (df_bad['stat'] == stat) &
                                                                         (df_bad['alpha'] == alpha)]['n'].unique()))

            results_mag.loc[(method, stat), dgp] = bad_s.loc[(method, stat), 'perc_better']
            mag_se.loc[(method, stat), dgp] = bad_s.loc[(method, stat), 'se_perc']
            mag_n_exp.loc[(method, stat), dgp] = bad_s.loc[(method, stat), 'nr_exp']

        results[f'both_{dgp}'] = bad_s

    #
    results_mag = results_mag.reset_index()
    results_mag = results_mag[results_mag.columns[[1, 0]].to_list() +
                              [d for d in results_mag.columns[2:].to_list() if "BiNorm" not in d]]
                              # + ["DGPBiNorm-1_1_2.0_0.5_1.0"]]     # skipping binorm because there is no >10% methods
    results_mag = results_mag.sort_values(['stat', 'perc_better'], ascending=False)
    results_mag[results_mag.columns[2:]] = results_mag[results_mag.columns[2:]] * 100

    # mag_se = mag_se[results_mag.index, :]
    mag_se = mag_se.reset_index()
    mag_se = mag_se.sort_values(['stat', 'se_perc'], ascending=False)
    mag_se[mag_se.columns[2:]] = mag_se[mag_se.columns[2:]] * 100

    mag_n_exp = mag_n_exp.reset_index()
    mag_n_exp = mag_n_exp.sort_values(['stat', 'nr_exp'], ascending=False)

    results[f'mag_{separate_dgp}'] = results_mag
    results[f'mag_{separate_dgp}_se'] = mag_se
    results[f'mag_{separate_dgp}_nr_exp'] = mag_n_exp

    mag_n = pd.DataFrame(df_bad_coverage.value_counts('n')/df.shape[0])
    mag_n.columns = ['better coverage']
    mag_n['equal coverage'] = df[((df['m2_better_kl'] + df['m1_better_kl']) < 1)].value_counts('n') / df.shape[0]
    mag_n['better distance'] = bad_dist.value_counts('n')/df.shape[0]
    mag_n['combined better'] = bad_both.value_counts('n')/df.shape[0]
    results['mag_n'] = mag_n * 100

    for name in results:
        print(name)
        if name[:3] == 'mag':
            # if name == 'mag_n':
            #     print(results[name].sort_index().to_latex(float_format="%.2f", na_rep=0))
            # else:
            print(results[name].to_latex(index=False, float_format="%.2f").replace('NaN', ''))
        # results[name].to_csv(f'results_better/{method_one}_{name}_{sided}.csv')

    return results


if __name__ == '__main__':

    methods = ['standard', 'double', 'studentized']
    name = 'vs_others_B1000_reps_10000_d50'

    # tables = {}
    # onlybts = True
    #
    # for stat in ['mean', 'median']:
    #     print('Aggregated with ', stat)
    #
    #     for table, name in zip(aggregate_results(f'results{folder_add}', combined_with=stat, withnans=True,
    #                                              onlybts=onlybts),
    #                            ['nans', 'kl div', 'kl div med', 'kl div rank',
    #                            'kl div se', 'kl rank se', 'kl_div_significant', 'kl_rank_significant']):
    #         print(name)
    #         if 'rank' in name:
    #             ff = "%.2f"
    #         else:
    #             ff = "%.4f"
    #         print(table.to_latex(float_format=ff))
    #         tables[name] = table


    # for m in methods:
    #
    #     one_vs_others(m, B=1000, reps=10000)
    #
    #     table = pd.read_csv(f'results/twosided_{m}_{name}.csv')
    #     t_nan = table[(table['has_nans_m1'] == False) & (table['has_nans_m2'] == False)]
    #     t_nan = t_nan.drop(columns=['has_nans_m1', 'has_nans_m2'])
    #     t_nan.to_csv(f'results/twosided_{m}_{name}_nonans.csv', index=False)
    #
    #     table = pd.read_csv(f'results/onesided_{m}_{name}.csv')
    #     t_nan = table[(table['nan_perc_m1'] + table['nan_perc_m2']) == 0]
    #     t_nan = t_nan.drop(columns=['nan_perc_m1', 'nan_perc_m2'])
    #     t_nan.to_csv(f'results/onesided_{m}_{name}_nonans.csv', index=False)

    # results for step 2 of analysis
    rd = analyze_experiments('double', name=name, statistics=['mean', 'median', 'std', 'corr'], sided='onesided', separate_dgp='distance')
    rs = analyze_experiments('standard', name=name, statistics=['percentile_5', 'percentile_95'], sided='onesided', separate_dgp='distance')
    rd = analyze_experiments('double', name=name, statistics=['mean', 'median', 'std', 'corr'], sided='twosided', separate_dgp='distance')
    rs = analyze_experiments('standard', name=name, statistics=['percentile_5', 'percentile_95'], sided='twosided', separate_dgp='distance')

    # if you want to add latex table for percentage of better methods (n)
    # merged_df = pd.concat([rd['mag_n'], rs['mag_n']], axis=1, keys=['double', 'standard'])
    # print(merged_df.to_latex(float_format="%.2f", na_rep=0))


    # largest se of experiment distance estimation
    # df = pd.read_csv(f'results_final/results_from_intervals_mean_withnans.csv')
    # df['se_rel'] = df['std'] / 100 / df['avg_distance']
    # test = df.nlargest(n=50, columns=['se_rel'])
    # test_d = df[df['method'] == 'double'].nlargest(n=50, columns=['se_rel'])
    # test_s = df[df['method'] == 'standard'].nlargest(n=50, columns=['se_rel'])


