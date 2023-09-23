import numpy as np
import scipy as sp
import pandas as pd


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


def kl(p, q):
    """K-L divergence."""
    part1 = 0 if p == 0 else p * np.log2(p / q)
    part2 = 0 if p == 1 else (1 - p) * np.log2((1 - p) / (1 - q))
    return part1 + part2


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
                  reps=10000):
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

            if (dgp == 'DGPNorm_0_1') & (alpha == 0.025) & (stat == 'mean') & (n == 4):
                stop = True     # TODO delete (just for testing)

            if nans1 == 1:      # if method_one was unable to give any predictions

                # compare to all other methods for this statistic
                for m2 in compare_to[stat]:
                    nans2 = df_a[m2].isna().mean()

                    exp_dict = {'method': m2, 'alpha': alpha, 'dgp': dgp, 'stat': stat, 'n': n, 'nan_perc_m1': nans1,
                                'nan_perc_m2': nans2}

                    if nans2 == 1:      # if m2 was unable to give any predictions
                        exp_dict.update({'coverage_m1_mu': -1, 'coverage_m1_q025': -1, 'coverage_m1_q975': -1,
                                         'coverage_m2_mu': -1, 'coverage_m2_q025': -1, 'coverage_m2_q975': -1})

                    else:                               # m2 gave some predictions (TODO do we want to know how many?)
                        covers_m2 = (df_a[m2] > df_a['true_value']).values
                        res_cov = analyze_coverage_m1_false(covers_m2, alpha)
                        exp_dict.update(res_cov)

                    results_better1.append(exp_dict)

            else:                                       # method_one gave some predictions for this experiment

                covers_m1 = (df_a[method_one] > df_a['true_value']).values
                distance_m1 = abs(df_a[method_one] - df_a['exact']).values

                for m2 in compare_to[stat]:             # compare to all other methods
                    # TODO do we want to separate those without predictions?
                    covers_m2 = (df_a[m2] > df_a['true_value']).values
                    distance_m2 = abs(df_a[m2] - df_a['exact']).values

                    res_cov = analyze_coverage(covers_m1, covers_m2, alpha)
                    res_dist = analyze_length(distance_m1, distance_m2, 'dist')

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

                for m2 in compare_to[stat]:                 # compare to all other methods (TODO check predictions?)
                    covers_m2 = (df_au[m2] > df_au['true_value']).values & (df_al[m2] < df_al['true_value']).values
                    length_m2 = df_au[m2].values - df_al[m2].values

                    res_cov = analyze_coverage(covers_m1, covers_m2, alpha)
                    res_len = analyze_length(length_m1, length_m2, 'len')

                    has_nans2 = df_al[m2].isna().any() or df_au[m2].isna().any()

                    exp_dict = {'method': m2, 'alpha': alpha, 'dgp': dgp, 'stat': stat, 'n': n,
                                'has_nans_m1': has_nans1, 'has_nans_m2': has_nans2}
                    exp_dict.update(res_cov)
                    exp_dict.update(res_len)

                    results_better2.append(exp_dict)

    final_df1 = pd.DataFrame(results_better1)
    final_df1.sort_values(by='better_prob_m2', ascending=False)
    final_df1.to_csv(f'{result_folder}/onesided_{method_one}_vs_others_B{B}_reps_{reps}_d10.csv', index=False)

    final_df2 = pd.DataFrame(results_better2)
    final_df2.sort_values(by='better_prob_m2', ascending=False)
    final_df2.to_csv(f'{result_folder}/twosided_{method_one}_vs_others_B{B}_reps_{reps}_d10.csv', index=False)


def one_vs_other_analysis(method_one, other_methods=None, one_sided=None, two_sided=None, result_folder='results', B=1000,
                  reps=10000, skip_nans=True):
    # if one_sided is None:
    #     one_sided = [0.025, 0.05, 0.25, 0.75, 0.95, 0.975]
    # if two_sided is None:
    #     two_sided = [0.5, 0.9, 0.95]
    results_df = pd.read_csv(f'{result_folder}/onesided_{method_one}_vs_others_B{B}_reps_{reps}.csv')

    df = results_df[results_df['nan_perc_m1'] + results_df['nan_perc_m2'] == 0] if skip_nans else results_df

    # counting ns/methods that occur more than 50%
    df_bad = df[df['better_cov_prob'] < 0.5]
    df_bad['method'].value_counts()

    # mean probability by grouping
    df_mean = df[['stat', 'n', 'method', 'better_cov_prob']].groupby(['stat', 'n', 'method']).mean()        # df_bad?

    bad_n = df_bad[['method', 'n']].value_counts()
    bad_d = df_bad[['method', 'dgp']].value_counts()
    bad_s = df_bad[['method', 'stat']].value_counts()
    bad_ns = df_bad[['method', 'n', 'stat']].value_counts()


def analyze_experiments(method_one, other_methods=None, result_folder='results', B=1000, reps=10000, statistics=None,
                        ns=None):

    df = pd.read_csv(f'{result_folder}/onesided_{method_one}_vs_others_B{B}_reps_{reps}_lsd10_nonans.csv')
    # df_bad = df[df['m2_better_kl'] | (((df['m2_better_kl'] + df['m1_better_kl']) < 1) & (df['better_dist_prob']<0.1))]

    if statistics is not None:
        df = df[df['stat'].isin(statistics)]

    if ns is not None:
        df = df[df['n'].isin(ns)]

    df_bad_coverage = df[df['m2_better_kl']]
    bad_both = df[df['m2_better_kl'] | (((df['m2_better_kl'] + df['m1_better_kl']) < 1) & (df['better_dist_m2']))]

    print(f'There are {bad_both.shape[0]} cases where a method is better than {method_one}.'
          f'Out of them {bad_both[bad_both["m2_better_kl"]].shape[0]} have better coverage.')

    bad_s = pd.DataFrame()
    bad_s['nr_better'] = df_bad_coverage[['method', 'stat']].value_counts()
    bad_s['perc_better'] = bad_s['nr_better'] / df[['method', 'stat']].value_counts()
    bad_s = bad_s.sort_values('perc_better', ascending=False)
    for method, stat in bad_s.index:
        # for alpha in sorted(df_bad['alpha'].unique()):
        for dgp in sorted(df_bad_coverage['dgp'].unique()):
            bad_s.loc[(method, stat), dgp] = ', '.join(str(n) for n in
                                                         sorted(df_bad_coverage[(df_bad_coverage['method'] == method) &
                                                                       (df_bad_coverage['stat'] == stat) &
                                                                       (df_bad_coverage['dgp'] == dgp)]['n'].unique()))

    results = {'coverage': bad_s}

    bad_dist = df[(((df['m2_better_kl'] + df['m1_better_kl']) < 1) & (df['better_dist_m2']))]

    bad_s = pd.DataFrame()
    bad_s['nr_better'] = bad_dist[['method', 'stat']].value_counts()
    bad_s['perc_better'] = bad_s['nr_better'] / df[['method', 'stat']].value_counts()
    bad_s = bad_s.sort_values('perc_better', ascending=False)
    for method, stat in bad_s.index:
        # for alpha in sorted(df_bad['alpha'].unique()):
        for dgp in sorted(bad_dist['dgp'].unique()):
            bad_s.loc[(method, stat), dgp] = ', '.join(str(n) for n in
                                                         sorted(bad_dist[(bad_dist['method'] == method) &
                                                                       (bad_dist['stat'] == stat) &
                                                                       (bad_dist['dgp'] == dgp)]['n'].unique()))

    results['distance'] = bad_s

    for dgp in bad_both['dgp'].unique():
        df_bad = bad_both[bad_both['dgp'] == dgp]

        bad_s = pd.DataFrame()
        bad_s['nr_better'] = df_bad[['method', 'stat']].value_counts()
        bad_s['perc_better'] = bad_s['nr_better'] / df[df['dgp'] == dgp][['method', 'stat']].value_counts()
        bad_s = bad_s.sort_values('perc_better', ascending=False)

        bad_n = pd.DataFrame(df_bad[['method', 'n']].value_counts())
        bad_d = pd.DataFrame(df_bad[['method', 'dgp']].value_counts())
        bad_ns = pd.DataFrame(df_bad[['method', 'n', 'stat']].value_counts())

        for method, stat in bad_s.index:
            for alpha in sorted(df_bad['alpha'].unique()):
                bad_s.loc[(method, stat), alpha] = ', '.join(str(n) for n in
                                                           sorted(df_bad[(df_bad['method'] == method) &
                                                                         (df_bad['stat'] == stat) &
                                                                         (df_bad['alpha'] == alpha)]['n'].unique()))

        results[f'both_{dgp}'] = bad_s

    for name in results:
        results[name].to_csv(f'results_better/{method_one}_{name}.csv')



if __name__ == '__main__':

    # one_vs_others('double', B=1000, reps=10000)
    # one_vs_others('bca', B=1000, reps=10000)
    #
    # # script used to save them without any experiment with nans:
    # for name in ['twosided_bca_vs_others_B1000_reps_10000_lt_kl', 'twosided_double_vs_others_B1000_reps_10000_lt_kl']:
    #     table = pd.read_csv(f'results/{name}.csv')
    #     t_nan = table[(table['has_nans_m1'] == False) & (table['has_nans_m2'] == False)]
    #     t_nan = t_nan.drop(columns=['has_nans_m1', 'has_nans_m2'])
    #     t_nan.to_csv(f'results/{name}_nonans.csv', index=False)
    #
    # for name in ['onesided_bca_vs_others_B1000_reps_10000_lt_kl', 'onesided_double_vs_others_B1000_reps_10000_lt_kl']:
    #     table = pd.read_csv(f'results/{name}.csv')
    #     t_nan = table[(table['nan_perc_m1'] + table['nan_perc_m2']) == 0]
    #     t_nan = t_nan.drop(columns=['nan_perc_m1', 'nan_perc_m2'])
    #     t_nan.to_csv(f'results/{name}_nonans.csv', index=False)

    analyze_experiments('studentized', statistics=['percentile_5', 'percentile_95'])

