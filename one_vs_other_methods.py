import numpy as np
import scipy as sp
import pandas as pd


def analyze_length(lengthA, lengthB, len_dist='ld'):
    """Compares length of two sided intervals or distance to exact intervals (input absolute distances)"""

    diff = lengthA - lengthB
    diff_mu = np.mean(diff)
    diff_se = 0 if np.all(diff == diff[0]) else np.std(diff) / np.sqrt(len(lengthA))
    diff_is_neg_prob = sp.stats.norm.cdf(0, diff_mu, diff_se)

    normalized = (lengthA - lengthB) / (lengthA + 0.0001)       # TODO is this ok??
    norm_mu = np.mean(normalized)
    norm_se = 0 if np.all(normalized == normalized[0]) else np.std(normalized) / np.sqrt(len(lengthA))

    norm_is_neg_prob = 0 if norm_se == 0 else sp.stats.norm.cdf(0, norm_mu, norm_se)

    result = {
        'diff_mu': diff_mu,
        'diff_q025': diff_mu + sp.stats.norm.ppf(0.025) * diff_se,
        'diff_q975': diff_mu + sp.stats.norm.ppf(0.975) * diff_se,
        f'better_{len_dist}_prob': diff_is_neg_prob,
        'norm_mu': norm_mu,
        'norm_q025': norm_mu + sp.stats.norm.ppf(0.025) * norm_se,
        'norm_q975': norm_mu + sp.stats.norm.ppf(0.975) * norm_se,
        f'better_{len_dist}_prob_norm': norm_is_neg_prob
    }

    return result


def analyze_coverage(coverage_m1, coverage_m2, target_coverage):
    """Analyzes coverages of both methods. Average and confidence intervals for both, then percentage of times that the
    first method is better than the second one."""
    y = np.zeros(4)
    y[0] = np.sum((coverage_m1 == 0) & (coverage_m2 == 0))
    y[1] = np.sum((coverage_m1 == 1) & (coverage_m2 == 0))
    y[2] = np.sum((coverage_m1 == 0) & (coverage_m2 == 1))
    y[3] = np.sum((coverage_m1 == 1) & (coverage_m2 == 1))

    if sum(y) != len(coverage_m1):
        if sum(y) + np.sum(coverage_m1 == np.nan) + np.sum(coverage_m2 == np.nan) != len(coverage_m1):
            print('missing cases in y!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        else:
            print('missing cases because of nans.')

    alpha = y + 0.0001
    p = sp.stats.dirichlet.rvs(alpha, size=10000)

    coverageA = p[:, 1] + p[:, 3]
    coverageB = p[:, 2] + p[:, 3]

    errorA = np.abs(coverageA - target_coverage)
    errorB = np.abs(coverageB - target_coverage)

    error_diff = errorA - errorB

    return {
        'coverage_m1_mu': np.mean(coverageA),
        'coverage_m1_q025': np.percentile(coverageA, 2.5),
        'coverage_m1_q975': np.percentile(coverageA, 97.5),
        'coverage_m2_mu': np.mean(coverageB),
        'coverage_m2_q025': np.percentile(coverageB, 2.5),
        'coverage_m2_q975': np.percentile(coverageB, 97.5),
        'errordiff_mu': np.mean(error_diff),
        'errordiff_q025': np.percentile(error_diff, 2.5),
        'errordiff_q975': np.percentile(error_diff, 97.5),
        'better_cov_prob': np.mean(error_diff <= 0)            # TODO < ali <=
    }


def analyze_coverage_m1_false(coverage_m2, true_coverage):
    """Setting method one to False, if it returns nans, so that we get results for the second method."""
    coverage_m1 = np.array([False] * len(coverage_m2))
    result = analyze_coverage(coverage_m1, coverage_m2, true_coverage)
    result.update({'coverage_m1_mu': -1, 'coverage_m1_q005': -1, 'coverage_m1_q095': -1,
                   'errordiff_mu': np.nan, 'errordiff_q025': np.nan, 'errordiff_q975': np.nan,
                   'better_cov_prob': np.nan})
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
            if (dgp == 'DGPNorm_0_1') & (alpha == 0.025) & (stat == 'mean') & (n == 4):
                stop = True

            if df_a[method_one].isna().all():      # if method_one was unable to give any predictions

                # compare to all other methods for this statistic
                for m2 in compare_to[stat]:
                    exp_dict = {'method': m2, 'alpha': alpha, 'dgp': dgp, 'stat': stat, 'n': n}

                    if df_a[m2].isna().all():      # if m2 was unable to give any predictions
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
                    # TODO do we want to separate  those without predictions?
                    covers_m2 = (df_a[m2] > df_a['true_value']).values
                    distance_m2 = abs(df_a[m2] - df_a['exact']).values

                    res_cov = analyze_coverage(covers_m1, covers_m2, alpha)
                    res_dist = analyze_length(distance_m1, distance_m2, 'dist')

                    exp_dict = {'method': m2, 'alpha': alpha, 'dgp': dgp, 'stat': stat, 'n': n}
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

            # method_one didn't give any predictions for lower or any for upper bound
            if df_al[method_one].isna().all() or df_au[method_one].isna().all():

                for m2 in compare_to[stat]:                 # compare to all other methods
                    exp_dict = {'method': m2, 'alpha': alpha, 'dgp': dgp, 'stat': stat, 'n': n}

                    if np.all(df_al[m2] == np.nan) or np.all(df_au[m2] == np.nan):      # m2 gave no predictions
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

                    exp_dict = {'method': m2, 'alpha': alpha, 'dgp': dgp, 'stat': stat, 'n': n}
                    exp_dict.update(res_cov)
                    exp_dict.update(res_len)

                    results_better2.append(exp_dict)

    final_df1 = pd.DataFrame(results_better1)
    final_df1.to_csv(f'{result_folder}/onesided_{method_one}_vs_others_B{B}_reps_{reps}.csv', index=False)

    final_df2 = pd.DataFrame(results_better2)
    final_df2.to_csv(f'{result_folder}/twosided_{method_one}_vs_others_B{B}_reps_{reps}.csv', index=False)


one_vs_others('double', B=1000, reps=10000)
one_vs_others('bca', B=1000, reps=10000)

