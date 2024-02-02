import time
from collections import defaultdict
from os.path import exists

import matplotlib.cm
import scipy.stats
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib import ticker


def compare_all_cov_dis(df=None, comparing='coverage', methods=None, Bs=None, ns=None, folder_add=''):
    if df is None:
        df = pd.read_csv(f'results{folder_add}/{comparing}.csv')

    if Bs is not None:
        df = df[df['B'].isin(Bs)]

    if ns is not None:
        df = df[df['n'].isin(ns)]

    if methods is not None:
        df = df[df['method'].isin(methods)]
        methods_order = methods
    else:
        methods_order = df['method'].unique()

    distributions = df['dgp'].unique()

    fig, axs = plt.subplots(ncols=len(distributions), figsize=(20, 8))

    for i in range(len(distributions)):
        dfi = df[df['dgp'] == distributions[i]]
        name = distributions[i][3:].split('_')[0]
        sns.boxplot(x="alpha", hue="method", y=comparing, data=dfi, ax=axs[i], hue_order=methods_order, fliersize=1)
        axs[i].title.set_text(name)
        if i != len(distributions) - 1:
            axs[i].get_legend().remove()

    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.show()


def compare_alpha_cov_dis_by_n(df=None, comparing='coverage', alpha=0.95, methods=None, Bs=None, ns=None,
                               folder_add=''):
    if df is None:
        df = pd.read_csv(f'results{folder_add}/{comparing}.csv')

    df = df[df['alpha'] == alpha]

    if Bs is not None:
        df = df[df['B'].isin(Bs)]

    if methods is not None:
        df = df[df['method'].isin(methods)]
        methods_order = methods
    else:
        methods_order = df['method'].unique()

    distributions = df['dgp'].unique()

    fig, axs = plt.subplots(ncols=len(distributions), figsize=(20, 8))

    for i in range(len(distributions)):
        dfi = df[df['dgp'] == distributions[i]]
        name = distributions[i][3:].split('_')[0]
        sns.boxplot(x="n", hue="method", y=comparing, data=dfi, ax=axs[i], hue_order=methods_order, fliersize=1)
        axs[i].title.set_text(name)
        if i != len(distributions) - 1:
            axs[i].get_legend().remove()

    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.show()


def compare_cov_dis_grid(df=None, comparing='coverage', filter_by={'alpha': [0.95]}, x='n', row='statistic', col='dgp',
                         hue='method', save_add=None, title=None, ci=95, scale='linear', folder_add='', subfolder='',
                         set_ylim=False, colors=None):
    if df is None:
        df = pd.read_csv(f'results{folder_add}/{comparing}.csv')

    for key in filter_by.keys():
        df = df[df[key].isin(filter_by[key])]

    if colors is None:
        nm = df['method'].nunique()
        if nm > 10:
            cols = plt.cm.tab20(np.linspace(0.05, 0.95, df['method'].nunique()))
        else:
            cols = plt.cm.tab10(np.linspace(0.05, 0.95, df['method'].nunique()))
        colors = {m: c for (m, c) in zip(df['method'].unique(), cols)}

    g = sns.FacetGrid(df, row=row, col=col, margin_titles=True, sharex=True, sharey='row', palette=colors, aspect=2)
    if comparing == 'coverage':
        g.map_dataframe(plot_coverage_bars, colors=colors, ci=ci, scale=scale, set_ylim=set_ylim,
                        order=df[hue].unique(), hue=hue, x=x)
    else:
        g.map(sns.boxplot, x, comparing, hue, hue_order=df[hue].unique(), fliersize=0,
              whis=[(100 - ci) / 2, 50 + ci / 2],
              palette=colors)
        ylim = np.nanquantile(df['distance'], (0.01, 0.99))
        g.set(ylim=ylim)

        for axs in g.axes:
            for ax in axs:
                ax.axhline(0, linestyle='--', color='gray')

    g.add_legend(title='method')

    if title is not None:
        g.fig.subplots_adjust(top=0.95)
        g.fig.suptitle(title, fontsize=16)

    if save_add is not None:
        plt.savefig(f'images{folder_add}/comparison/{subfolder}/compare_{comparing}_{x}_{row}_{col}_{save_add}.png')
        print('saved')
        plt.close()
    # else:
    #     plt.show()


def plot_coverage_bars(data, **kwargs):
    colors = kwargs['colors']
    ci = kwargs['ci']
    scale = kwargs['scale']

    if 'cov_kind' in kwargs:                # for the possibility of plotting variance coverage with it
        cov_kind = kwargs['cov_kind']
    else:
        cov_kind = 'coverage'

    data['ci'] = np.sqrt(data[cov_kind] * (1 - data[cov_kind]) / data['repetitions'])
    if ci != 'se':
        data['ci'] *= scipy.stats.norm.ppf(0.5 + ci / 200)
    data['low'] = data[cov_kind] - data['ci']

    n_levels = len(kwargs['order'])
    group_width = 0.8
    bar_width = group_width / n_levels
    offsets = np.linspace(0, group_width - bar_width, n_levels)
    offsets -= offsets.mean()

    bar_pos = np.arange(data[kwargs['x']].nunique())
    for i, method in enumerate(kwargs['order']):
        data_m = data[data[kwargs['hue']] == method]
        offset = bar_pos + offsets[i]
        if data_m['ci'].shape[0] == 0:
            continue
        if data_m.shape[0] == 0:
            debug = True
        plt.bar(offset, data_m['ci'], bar_width, bottom=data_m[cov_kind], label=method, color=colors[method],
                ec=colors[method])
        plt.bar(offset, data_m['ci'], bar_width, bottom=data_m['low'], color=colors[method], ec=colors[method])

    for p in bar_pos[:-1]:
        plt.axvline(p + 0.5, ls=':', alpha=0.2)

    a = data['alpha'].values[0]
    if a > 0.9:
        if scale == 'logit':
            ylim = (0.8, 0.99)
        else:
            ylim = (0.8, 1)
    elif a < 0.1:
        ylim = (0, 0.2)
    else:
        ylim = (a - 0.1, a + 0.1)

    ax = plt.gca()

    ax.set_yscale(scale)

    if kwargs['set_ylim']:
        ax.set(ylim=ylim)

    ax.axhline(a, linestyle='--', color='gray')
    plt.yticks(list(plt.yticks()[0]) + [a])

    ax.set_xlabel(kwargs['x'])
    ax.set_ylabel(cov_kind)
    plt.xticks(bar_pos, sorted(data[kwargs['x']].unique()))


def main_plot_comparison(B_as_method=False, filter_by={}, additional='', scale='linear', folder_add='', set_ylim=True,
                         levels=None, stds=None):
    for comparing in ['coverage', 'distance']:
        df = pd.read_csv(f'results{folder_add}/{comparing}.csv')
        if 'method' in filter_by:
            nm = len(filter_by['method'])
        else:
            nm = df['method'].nunique()
        if nm > 10:
            cols = plt.cm.tab20(np.linspace(0.05, 0.95, df['method'].nunique()))
        else:
            cols = plt.cm.tab10(np.linspace(0.05, 0.95, df['method'].nunique()))
        colors = {m: c for (m, c) in zip(df['method'].unique(), cols)}
        for statistic in ['mean', 'median', 'std', 'percentile_5', 'percentile_95', 'corr']:
            if B_as_method:
                # we take only B=1000 for our experiments, implement if you want to compare different B sizes
                pass
            else:
                for B in [1000]:
                    df_part = df[(df['B'] == B) & (df['statistic'] == statistic)]

                    if additional == 'hierarchical':
                        for level in levels:
                            for std in stds:
                                df_part_part = df_part[(df_part['levels'] == level) & (df_part['std'] == std)]
                                if df_part_part.shape[0] == 0:
                                    print('Empty df for: ', statistic, B, level)
                                    continue
                                title = f'{comparing}s for {statistic} using B = {B}, {level} levels, std {std}'
                                compare_cov_dis_grid(df_part_part, comparing=comparing, filter_by=filter_by,
                                                     x='n_leaves', row='alpha', col='n_branches', title=title,
                                                     save_add=f'{statistic}_{B}_{level}_{std}{additional}', scale=scale,
                                                     folder_add=folder_add, set_ylim=set_ylim, colors=colors)

                    else:
                        if df_part.shape[0] == 0:
                            print('Empty df for: ', statistic, B)
                            continue
                        title = f'{comparing}s for {statistic} using B = {B}'
                        subfolder = '' if set_ylim else 'noylim'
                        compare_cov_dis_grid(df_part, comparing=comparing, filter_by=filter_by, x='n', row='alpha',
                                             col='dgp', title=title, save_add=f'{statistic}_{B}{additional}',
                                             scale=scale, folder_add=folder_add, set_ylim=set_ylim, subfolder=subfolder,
                                             colors=colors)
        del df  # clear space


def plot_times_line():
    df = pd.read_csv(f'results/times.csv')
    for stat in ['mean', 'median', 'std', 'percentile_5', 'percentile_95', 'corr']:
        filter_by = {'B': [1000], 'statistic': [stat]}
        df_part = df.copy()
        for key in filter_by.keys():
            df_part = df_part[df_part[key].isin(filter_by[key])]
            df_part[key] = np.nan
        df_na = df_part.dropna(axis=1, how='all')

        id_vars = [var for var in ['dgp', 'statistic', 'n', 'B', 'repetitions'] if var not in filter_by.keys()]
        df_long = pd.melt(df_na, id_vars=id_vars, value_name='t', var_name='method')  # .dropna()

        sns.lineplot(data=df_long, x='n', hue='method', y='t', ci=95)  # errorbar=('ci', 95)), ci is deprecated

        plt.xscale('log', base=2)
        xt = sorted(df_long['n'].unique())
        plt.xticks(xt, labels=[str(x) for x in xt])
        plt.yscale('log')
        plt.ylabel('t [s]')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        # plt.title(f'Times of CI calculation for {stat}')
        plt.tight_layout()
        plt.savefig(f'images/comparison/times_line_{stat}_ylog10.png')
        plt.close()


def plot_times_lengths_grid(comparing='times', filter_by: dict = None, title=None, save_add=None, scale='linear',
                            ci=95, folder_add='', subfolder=''):
    df = pd.read_csv(f'results{folder_add}/{comparing}.csv')

    if filter_by is not None:
        for key in filter_by.keys():
            df = df[df[key].isin(filter_by[key])]

    id_vars = ['dgp', 'statistic', 'n', 'B', 'repetitions']
    if comparing == 'length':
        id_vars.append('CI')
    df_long = pd.melt(df, id_vars=id_vars).dropna()

    nm = df_long['variable'].nunique()
    if nm > 10:
        cols = plt.cm.tab20(np.linspace(0.05, 0.95, df_long['variable'].nunique()))
    else:
        cols = plt.cm.tab10(np.linspace(0.05, 0.95, df_long['variable'].nunique()))
    colors = {m: c for (m, c) in zip(df_long['variable'].unique(), cols)}

    g = sns.FacetGrid(df_long, row='n', col='statistic', margin_titles=True, palette=colors)
    g.map(sns.boxplot, 'B', 'value', 'variable', hue_order=df_long['variable'].unique(), palette=colors, fliersize=0,
          whis=[(100 - ci) / 2, 50 + ci / 2])
    g.add_legend(title='method')

    g.set(yscale=scale)
    g.set_axis_labels(y_var=comparing)

    if title is not None:
        g.fig.subplots_adjust(top=0.95)
        g.fig.suptitle(title, fontsize=16)

    if save_add is not None:
        plt.savefig(f'images{folder_add}/comparison/{subfolder}/{comparing}_{save_add}.png')
        plt.close()
    else:
        plt.show()


def separate_experiment_plots(result_folder='results', B=1000, reps=10000, showoutliers=False):
    coverages = pd.read_csv(f'{result_folder}/coverage.csv')
    lengths = pd.read_csv(f'{result_folder}/length.csv')

    coverages = coverages[(coverages['B'] == B) & (coverages['repetitions'] == reps)]
    lengths = lengths[(lengths['B'] == B) & (lengths['repetitions'] == reps)]

    if reps < 10000:
        distances = pd.read_csv(f'{result_folder}/distance.csv')
        distances = distances[(distances['B'] == B) & (distances['repetitions'] == reps)]
    else:  # empty dataframe for final results (wasn't enough space, didn't save distances separately)
        distances = pd.read_csv(f'{result_folder}/intervals.csv')
        distances = distances[(distances['B'] == B) & (distances['repetitions'] == reps)]
        for m in ['percentile', 'basic', 'bca', 'bc', 'standard', 'smoothed', 'double', 'studentized', 'ttest',
                  'wilcoxon', 'ci_quant_param', 'ci_quant_nonparam', 'maritz-jarrett', 'chi_sq', 'ci_corr_pearson']:
            distances[m] = distances[m] - distances['exact']

    for sided in ['onesided', 'twosided']:
        if sided == 'onesided':
            df_whole = distances
        else:
            df_whole = lengths
            df_whole.rename(columns={'CI': 'alpha'}, inplace=True)

        for [dgp, stat, alpha], df in df_whole.groupby(['dgp', 'statistic', 'alpha']):
            # plotting coverage
            au = alpha if sided == 'onesided' else 0.5 + alpha / 2
            cov_df = coverages[(coverages['dgp'] == dgp) & (coverages['statistic'] == stat) &
                               (coverages['alpha'] == au)]
            if sided == 'twosided':
                cov_df2 = coverages[(coverages['dgp'] == dgp) & (coverages['statistic'] == stat) &
                                    (coverages['alpha'] == round(1 - au, 5))]
                # merge dataframes to be able to subtract coverages of same experiments
                cov_df = pd.merge(cov_df, cov_df2, on=['method', 'dgp', 'statistic', 'n', 'B', 'repetitions'],
                                  suffixes=('_au', '_al'))
                cov_df['coverage'] = cov_df['coverage_au'] - cov_df['coverage_al']
                cov_df['alpha'] = cov_df['alpha_au'] - cov_df['alpha_al']

            nm = cov_df['method'].nunique()
            if nm > 10:
                cols = plt.cm.tab20(np.linspace(0.05, 0.95, cov_df['method'].nunique()))
            else:
                cols = plt.cm.tab10(np.linspace(0.05, 0.95, cov_df['method'].nunique()))
            colors = {m: c for (m, c) in zip(cov_df['method'].unique(), cols)}
            order = cov_df['method'].unique()

            val_name = 'distance' if sided == 'onesided' else 'length'

            fig = plt.figure(figsize=(6, 12), constrained_layout=True)
            txt = fig.suptitle(f"Coverage and {val_name} for {dgp}, {stat}, {alpha}", fontsize=14)

            plt.subplot(2, 1, 1)
            plot_coverage_bars(cov_df, colors=colors, ci=95, scale='linear', set_ylim=False, order=order, hue='method',
                               x='n')
            plt.title('Coverages')

            # long dataframe for distances
            cols_here = df.dropna(axis=1, how='all').columns.tolist()
            all_methods = ["percentile", "basic", "bca", "bc", "standard", "smoothed", "double", "studentized", "ttest",
                           "wilcoxon", "ci_quant_param", "ci_quant_nonparam", "maritz-jarrett", "chi_sq",
                           "ci_corr_pearson"]
            methods_here = [m for m in all_methods if m in cols_here]
            dist_df_long = pd.melt(df, id_vars=["alpha", "dgp", "n"], value_vars=methods_here, var_name="method",
                                   value_name=val_name)

            # plotting distances
            plt.subplot(2, 1, 2)
            plt.title('Distances' if sided == 'onesided' else 'Lengths')

            sns.boxplot(x="n", y=val_name, hue="method", data=dist_df_long, hue_order=order, palette=colors,
                        showfliers=showoutliers)
            plt.axhline(y=0, color="gray", linestyle="--")
            plt.yticks(list(plt.yticks()[0]) + [0])
            plt.legend([], [], frameon=False)
            plt.yscale(['symlog', 'log'][int(sided == 'twosided')])
            plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f'{value:.2f}'))
            # plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

            handles, labels = plt.gca().get_legend_handles_labels()
            lgd = fig.legend(handles, labels, loc='center left', title="Method", bbox_to_anchor=(1, 0.5))

            plt.savefig(f'images/separate_experiments_{sided}{["", "_outliers"][int(showoutliers)]}/'
                        f'plots_{sided}_experiment_{dgp}_{stat}_{alpha}.png',
                        bbox_extra_artists=(lgd, txt), bbox_inches='tight')


def hierarchical_from_intervals(folder='results_hierarchical', bts_method='double', n_lvl=3, filenames=['intervals']):
    # for wide tables (new results), skipping just replications that have nans
    # we are reading results line by line because of possibility of very large dataframes that can't fit in memory

    def analyse_line(line_str, line_keys, method):
        line_dict = dict(zip(line_keys, line_str.strip('\n').split(',')))

        if line_dict['method'] == 'exact':
            # we don't need exact intervals in this study
            return False

        sampling, strategy, m = line_dict['method'].split('_')
        if int(line_dict['B']) != 1000 or m != method or len(strategy) != n_lvl or line_dict['statistic'] != 'mean':
            # skip the line, we are building each method's table separately, only for mean statistic
            return False

        line_dict['strategy'] = strategy
        line_dict['method'] = method

        return line_dict

    results = []

    for filename in filenames:
        with open(f'{folder}/{filename}.csv') as f:
            keys = f.readline().strip('\n').split(',')  # header
            for line in f:
                line_results = analyse_line(line, keys, bts_method)
                if not line_results:
                    # line_results == False
                    continue
                results.append(line_results)
                # df = pd.concat([df, pd.Series(line_results).to_frame().T], ignore_index=True)

    df = pd.DataFrame(results)
    df.to_csv(f'{folder}/hierarchical_nlvl{n_lvl}_{bts_method}.csv', index=False)


def separate_experiment_plots_hierarchical(result_folder='results_hierarchical', B=1000, reps=1000, std=1,
                                           method='double'):
    coverages = pd.read_csv(f'{result_folder}/coverage.csv')
    repetitions = {}
    nlvls = [2, 3, 4]
    for nlvl in nlvls:
        rdf = pd.read_csv(f'{result_folder}/hierarchical_nlvl{nlvl}_{method}.csv')
        repetitions[nlvl] = rdf[(rdf['B'] == B) & (rdf['repetitions'] == reps)]

    coverages = coverages[(coverages['B'] == B) & (coverages['repetitions'] == reps) & (coverages['std'] == std)
                          & (coverages['method'].str.split('_').str[2] == method) & (coverages['statistic'] == 'mean')]
    coverages['strategy'] = coverages['method'].str.split('_').str[1]
    coverages_a95 = coverages[(coverages['alpha'] == 0.95)]     # hack for variance coverage (always checking 95CI)

    for alpha in coverages['alpha'].unique():

        cov_df = coverages[(coverages['alpha'] == alpha)]

        fig = plt.figure(figsize=(10, 14), constrained_layout=True)
        # txt = fig.suptitle(f"Coverage and variance coverage for std {std}, {method}", fontsize=14)
        lgds = []

        for i, nlvl in enumerate(nlvls):
            cov_lvl_df = cov_df[cov_df['levels'] == nlvl]
            cov_lvl_95 = coverages_a95[coverages_a95['levels'] == nlvl]

            nm = cov_lvl_df['strategy'].nunique()
            if nm > 10:
                cols = plt.cm.tab20(np.linspace(0.05, 0.95, cov_lvl_df['strategy'].nunique()))
            else:
                cols = plt.cm.tab10(np.linspace(0.05, 0.95, cov_lvl_df['strategy'].nunique()))
            colors = {m: c for (m, c) in zip(cov_lvl_df['strategy'].unique(), cols)}
            order = cov_lvl_df['strategy'].unique()

            plt.subplot(3, 2, 2 * i + 1)
            plot_coverage_bars(cov_lvl_df, colors=colors, ci=95, scale='linear', set_ylim=False, order=order,
                               hue='strategy', x='n')
            if i == 0:
                plt.title('Accuracy')
            if i != 2:
                plt.xlabel('')

            plt.xticks(plt.xticks()[0], plt.xticks()[1], rotation=45)
            plt.ylim(max(plt.ylim()[0], 0), min(plt.ylim()[1], 1))


            # plotting distances
            ax = plt.subplot(3, 2, 2 * i + 2)
            plot_coverage_bars(cov_lvl_95, colors=colors, ci=95, scale='linear', set_ylim=False, order=order,
                               hue='strategy', x='n', cov_kind='var_coverage')
            handles, labels = ax.get_legend_handles_labels()
            lgds.append(fig.legend(handles, labels, loc='center left', title="Strategy", bbox_to_anchor=(1, (5-2*i)/6)))
            if i == 0:
                plt.title("Imitation of DGP's Variation Properties")
            if i != 2:
                plt.xlabel('')

            plt.xticks(plt.xticks()[0], plt.xticks()[1], rotation=45)
            plt.ylim(max(plt.ylim()[0], 0), min(plt.ylim()[1], 1))

        # handles, labels = plt.gca().get_legend_handles_labels()
        # lgd1 = fig.legend(handles, labels, loc='center left', title="Strategy", bbox_to_anchor=(1, 0.16))
        # lgd2 = fig.legend(handles, labels, loc='center left', title="Strategy", bbox_to_anchor=(1, 0.5))
        # lgd3 = fig.legend(handles, labels, loc='center left', title="Strategy", bbox_to_anchor=(1, 0.84))
        plt.savefig(f'images_hierarchical/separate_experiments/'
                    f'plots_experiment_{alpha}_{std}_{method}.png',
                    bbox_extra_artists=lgds, bbox_inches='tight')


if __name__ == '__main__':
    # folder_add = '_hierarchical'
    # folder_add = '_10000_reps'
    folder_add = '_final'
    subfolder = ''
    # additional = 'hierarchical'
    additional = ''
    # cov = pd.read_csv(f'results{folder_add}/coverage.csv')
    bts_methods = ['percentile', 'standard', 'basic', 'bc', 'bca', 'double', 'smoothed']
    plt.style.use('seaborn')

    # main_plot_comparison(filter_by={}, additional=additional, scale='linear', folder_add=folder_add, set_ylim=True)
    #


    plot_times_line()

    # separate_experiment_plots('results', showoutliers=False)

    # hierarchical results separation
    # for levels in [2, 3, 4]:
    #     for method in ['double', 'percentile', 'bca']:
    #         print(method, levels)
    #         hierarchical_from_intervals(folder='results_hierarchical', bts_method=method, n_lvl=levels,
    #                                     filenames=['intervals_first550experiments', 'intervals'])

    # plots for hierarchical experiments
    # separate_experiment_plots_hierarchical()


