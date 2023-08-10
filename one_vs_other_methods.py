import numpy as np
import scipy as sp
import pandas as pd


def analyze_length(lengthA, lengthB, len_dist='ld'):
    # compares length of two sided intervals or distance to exact intervals (input absolute distances)

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
        'diff_q005': diff_mu + sp.stats.norm.ppf(0.05) * diff_se,
        'diff_q095': diff_mu + sp.stats.norm.ppf(0.95) * diff_se,
        f'better_{len_dist}_prob': diff_is_neg_prob,
        'norm_mu': norm_mu,
        'norm_q005': norm_mu + sp.stats.norm.ppf(0.05) * norm_se,
        'norm_q095': norm_mu + sp.stats.norm.ppf(0.95) * norm_se,
        f'better_{len_dist}_prob_norm': norm_is_neg_prob
    }

    return result


def analyze_coverage(coverage_m1, coverage_m2, target_coverage):
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
        'coverage_m1_q005': np.percentile(coverageA, 5),
        'coverage_m1_q095': np.percentile(coverageA, 95),
        'coverage_m2_mu': np.mean(coverageB),
        'coverage_m2_q005': np.percentile(coverageB, 5),
        'coverage_m2_q095': np.percentile(coverageB, 95),
        'errordiff_mu': np.mean(error_diff),
        'errordiff_q005': np.percentile(error_diff, 5),
        'errordiff_q095': np.percentile(error_diff, 95),
        'better_cov_prob': np.mean(error_diff <= 0)            # TODO < ali <=
    }


def one_vs_others(method_one, other_methods=None, one_sided=None, two_sided=None, result_folder='results', B=1000,
                  reps=10000):
    if one_sided is None:
        one_sided = [0.025, 0.05, 0.25, 0.75, 0.95, 0.975]
    if two_sided is None:
        two_sided = [0.5, 0.9, 0.95]
    results_df = pd.read_csv(f'{result_folder}/intervals.csv')

    results_df = results_df.loc[(results_df['B'] == B) &
                                (results_df['repetitions'] == reps)].drop(columns=['B', 'repetitions'])

    results_better1 = []
    results_better2 = []

    for [dgp, stat, n], df in results_df.groupby(['dgp', 'statistic', 'n']):

        print(str([dgp, stat, n]))

        df = df.dropna(axis=1, how='all')

        if other_methods is None:
            compare_to = [m for m in df.columns.tolist() if m not in [method_one, 'dgp', 'statistic', 'n', 'alpha',
                                                                      'true_value', 'exact']]
        else:
            compare_to = [m for m in other_methods if m in df.columns.tolist()]

        for alpha in one_sided:
            df_a = df.loc[df['alpha'] == alpha].drop(columns='alpha')

            if method_one not in df_a.columns:

                for m2 in compare_to:

                    exp_dict = {'method': m2, 'alpha': alpha, 'dgp': dgp, 'stat': stat, 'n': n}

                    if m2 not in df_a.columns:
                        exp_dict.update({'coverage_m1_mu': -1, 'coverage_m1_q005': -1, 'coverage_m1_q095': -1,
                                         'coverage_m2_mu': -1, 'coverage_m2_q005': -1, 'coverage_m2_q095': -1})

                    else:
                        covers_m2 = (df_a[m2] > df_a['true_value']).values
                        covers_m1 = np.array([False] * len(covers_m2))
                        res_cov = analyze_coverage(covers_m1, covers_m2, alpha)
                        res_cov.update({'coverage_m1_mu': -1, 'coverage_m1_q005': -1, 'coverage_m1_q095': -1,
                                         'errordiff_mu': np.nan, 'errordiff_q005': np.nan, 'errordiff_q095': np.nan,
                                         'better_cov_prob': np.nan})
                        exp_dict.update(res_cov)

                    results_better1.append(exp_dict)

            else:

                covers_m1 = (df_a[method_one] > df_a['true_value']).values
                distance_m1 = abs(df_a[method_one] - df_a['exact']).values

                for m2 in compare_to:
                    covers_m2 = (df_a[m2] > df_a['true_value']).values
                    distance_m2 = abs(df_a[m2] - df_a['exact']).values

                    res_cov = analyze_coverage(covers_m1, covers_m2, alpha)
                    res_dist = analyze_length(distance_m1, distance_m2, 'dist')

                    exp_dict = {'method': m2, 'alpha': alpha, 'dgp': dgp, 'stat': stat, 'n': n}
                    exp_dict.update(res_cov)
                    exp_dict.update(res_dist)

                    results_better1.append(exp_dict)

        print('Finished onesided')

        for alpha in two_sided:
            au = 0.5 + alpha / 2
            al = round(0.5 - alpha / 2., 10)

            df_al = df.loc[df['alpha'] == al].drop(columns='alpha')
            df_au = df.loc[df['alpha'] == au].drop(columns='alpha')

            if method_one not in df_al.columns or 'double' not in df_au.columns:

                for m2 in compare_to:

                    exp_dict = {'method': m2, 'alpha': alpha, 'dgp': dgp, 'stat': stat, 'n': n}

                    if m2 not in df_al.columns or m2 not in df_au.columns:
                        exp_dict.update({'coverage_m1_mu': -1, 'coverage_m1_q005': -1, 'coverage_m1_q095': -1,
                                         'coverage_m2_mu': -1, 'coverage_m2_q005': -1, 'coverage_m2_q095': -1})

                    else:
                        covers_m2 = (df_au[m2] > df_au['true_value']).values & (df_al[m2] < df_al['true_value']).values
                        covers_m1 = np.array([False] * len(covers_m2))
                        res_cov = analyze_coverage(covers_m1, covers_m2, alpha)
                        res_cov.update({'coverage_m1_mu': -1, 'coverage_m1_q005': -1, 'coverage_m1_q095': -1,
                                        'errordiff_mu': np.nan, 'errordiff_q005': np.nan, 'errordiff_q095': np.nan,
                                        'better_cov_prob': np.nan})
                        exp_dict.update(res_cov)

                    results_better2.append(exp_dict)

            else:

                covers_m1 = (df_au[method_one] > df_au['true_value']).values & \
                            (df_al[method_one] < df_al['true_value']).values
                length_m1 = df_au[method_one].values - df_al[method_one].values

                for m2 in compare_to:
                    covers_m2 = (df_au[m2] > df_au['true_value']).values & (df_al[m2] < df_al['true_value']).values
                    length_m2 = df_au[m2].values - df_al[m2].values

                    res_cov = analyze_coverage(covers_m1, covers_m2, alpha)
                    res_len = analyze_length(length_m1, length_m2, 'len')

                    exp_dict = {'method': m2, 'alpha': alpha, 'dgp': dgp, 'stat': stat, 'n': n}
                    exp_dict.update(res_cov)
                    exp_dict.update(res_len)

                    results_better2.append(exp_dict)

        print('finished ' + str([dgp, stat, n]))

    final_df1 = pd.DataFrame(results_better1)
    final_df1.to_csv(f'{result_folder}/onesided_{method_one}_vs_others_B{B}_reps_{reps}.csv', index=False)

    final_df2 = pd.DataFrame(results_better2)
    final_df2.to_csv(f'{result_folder}/twosided_{method_one}_vs_others_B{B}_reps_{reps}.csv', index=False)


one_vs_others('double', B=1000, reps=10000)
one_vs_others('bca', B=1000, reps=10000)


# import pystan

# TOY DATASET
# def m_asymptotic(x, alpha):
#     mu = np.mean(x)
#     delta = sp.stats.t.ppf(1.0 - alpha / 2, len(x) - 1) * np.std(x, ddof=1) / np.sqrt(len(x))
#     return {'lb': mu - delta, 'ub': mu + delta}
#
#
# def m_classic(x, alpha):
#     res = sp.stats.t.interval(1.0 - alpha, len(x) - 1, loc=np.mean(x), scale=np.std(x, ddof=1) / np.sqrt(len(x)))
#     return {'lb': res[0], 'ub': res[1]}
#
#
# def m_bootstrap(x, alpha):
#     n_boot = 1000
#     boot_means = np.array([np.mean(np.random.choice(x, size=len(x))) for _ in range(n_boot)])
#     lb = np.percentile(boot_means, alpha / 2 * 100)
#     ub = np.percentile(boot_means, (1 - alpha / 2) * 100)
#     return {'lb': lb, 'ub': ub}
#
#
# # Prepare toy dataset
# np.random.seed(0)
# n_rep = 1000
# n = 8
# alpha = 0.05
#
# res = []
# for i in range(1, n_rep + 1):
#     x = np.random.normal(size=n)
#
#     ciA = m_bootstrap(x, alpha)
#     ciB = m_classic(x, alpha)
#
#     new_row = {
#         'id': i,
#         'lb_A': ciA['lb'], 'ub_A': ciA['ub'],
#         'lb_B': ciB['lb'], 'ub_B': ciB['ub']
#     }
#
#     res.append(new_row)
#
# # Compute interval coverage and length
# res_df = pd.DataFrame(res)
# res_df['lengthA'] = res_df['ub_A'] - res_df['lb_A']
# res_df['lengthB'] = res_df['ub_B'] - res_df['lb_B']
# # res_df['coverageA'] = np.where((res_df['ub_A'] >= 0) & (res_df['lb_A'] <= 0), 1, 0)
# # res_df['coverageB'] = np.where((res_df['ub_B'] >= 0) & (res_df['lb_B'] <= 0), 1, 0)
# res_df['coverageA'] = sp.stats.bernoulli.rvs(p=0.95, size=len(res_df['ub_A']))
# res_df['coverageB'] = sp.stats.bernoulli.rvs(p=0.93, size=len(res_df['ub_A']))

# def analyze_coverage_stan(coverageA, coverageB, target_coverage):
#     y = np.full(len(coverageA), -1)
#
#     y[~coverageA & ~coverageB] = 1
#     y[coverageA & ~coverageB] = 2
#     y[~coverageA & coverageB] = 3
#     y[coverageA & coverageB] = 4
#
#     model = pystan.StanModel(file="./stan/analyze_coverage.stan")
#     stan_data = {'n': len(y), 'x': y, 'target_coverage': target_coverage}
#     fit = model.sampling(data=stan_data, chains=1, iter=10000, warmup=500, refresh=0)
#     smp = pd.DataFrame(fit.extract())
#
#     result = {
#         'coverageA_mu': np.mean(smp['coverageA']),
#         'coverageA_q005': np.quantile(smp['coverageA'], 0.05),
#         'coverageA_q095': np.quantile(smp['coverageA'], 0.95),
#         'coverageB_mu': np.mean(smp['coverageB']),
#         'coverageB_q005': np.quantile(smp['coverageB'], 0.05),
#         'coverageB_q095': np.quantile(smp['coverageB'], 0.95),
#         'errordiff_mu': np.mean(smp['error_diff']),
#         'errordiff_q005': np.quantile(smp['error_diff'], 0.05),
#         'errordiff_q095': np.quantile(smp['error_diff'], 0.95),
#         'A_is_better_prob': np.mean(smp['error_diff'] < 0)
#     }
#
#     return result
#
# # Analyze lengths
# length_result = analyze_length(res_df['lengthA'], res_df['lengthB'])
# print("Length Analysis:", length_result)
#
# # Analyze coverages
# target_coverage = 0.95
# coverage_result = analyze_coverage(res_df['coverageA'], res_df['coverageB'], target_coverage)
# print("Coverage Analysis:", coverage_result)
