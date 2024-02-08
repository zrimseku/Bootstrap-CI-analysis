from pathlib import Path

import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

from shiny import App, Inputs, Outputs, Session, reactive, render, req, ui

from results_visualizations import plot_coverage_bars, compare_cov_dis_grid

sns.set_theme()


df = pd.read_csv(Path(__file__).parent / "results_final/coverage.csv", na_values="NA")
df = df[df['B'] == 1000]
print(df.shape)
selectable_cols = ['Confidence level', 'Statistic', 'Distribution']
methods = df["method"].unique().tolist()
alphas1 = df["alpha"].unique().tolist()
alphas1.sort()
alphas2 = [round(b - a, 5) for a, b in zip(alphas1[:len(alphas1)//2], alphas1[len(alphas1)//2:][::-1])]
statistics = df["statistic"].unique().tolist()
distributions = df["dgp"].unique().tolist()

app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.input_radio_buttons("sided", "Confidence intervals", {"1": "One-sided", "2": "Two-sided"}),
        ui.input_selectize(
            "xgrid", "X grid", selectable_cols + ["No X grid"], selected="No X grid"
        ),
        ui.panel_conditional(
            "input.xgrid === 'Confidence level'", ui.input_checkbox_group(
                "alphas_x", "Confidence levels", alphas1, selected=alphas1
            )),
        ui.panel_conditional(
            "input.xgrid === 'Statistic'", ui.input_checkbox_group(
                "statistics_x", "Statistics", statistics, selected=statistics
            )),
        ui.panel_conditional(
            "input.xgrid === 'Distribution'", ui.input_checkbox_group(
                "distributions_x", "Distributions", distributions, selected=distributions
            )),
        ui.input_selectize(
            "ygrid", "Y grid", selectable_cols + ["No Y grid"], selected="No Y grid"
        ),
        ui.panel_conditional(
            "input.ygrid === 'Confidence level'", ui.input_checkbox_group(
                "alphas_y", "Confidence levels", alphas1, selected=alphas1
            )),
        ui.panel_conditional(
            "input.ygrid === 'Statistic'", ui.input_checkbox_group(
                "statistics_y", "Statistics", statistics, selected=statistics
            )),
        ui.panel_conditional(
            "input.ygrid === 'Distribution'", ui.input_checkbox_group(
                "distributions_y", "Distributions", distributions, selected=distributions
            )),
        ui.panel_conditional(
            "input.xgrid != 'Confidence level' & input.ygrid != 'Confidence level'", ui.input_selectize(
            "alpha", "Confidence level", alphas1, selected=0.95
            )),
        ui.panel_conditional(
            "input.xgrid != 'Statistic' & input.ygrid != 'Statistic'", ui.input_selectize(
                "statistic", "Statistic", statistics, selected=statistics[0]
            )),
        ui.panel_conditional(
            "input.xgrid != 'Distribution' & input.ygrid != 'Distribution'", ui.input_selectize(
            "distribution", "Distribution", distributions, selected=distributions[0]
            )),
        ui.input_checkbox_group(
            "methods", "Filter by methods", methods, selected=methods
        ),
        ui.hr(),
        # ui.input_switch("show_margins", "Show marginal plots", value=True),
    ),
    ui.card(
        ui.output_plot("coverages"),
    ),
)


def server(input: Inputs, output: Outputs, session: Session):

    @reactive.Effect
    def update_choices():
        """Updates possible choices based on selected values."""
        if input.sided() == '1':
            ui.update_selectize('alpha', choices=alphas1)
        else:
            ui.update_selectize('alpha', choices=alphas2)

        if 'Statistic' not in [input.xgrid(), input.ygrid()]:
            bootstrap_methods = ['percentile', 'basic', 'bca', 'bc', 'standard', 'smoothed', 'double', 'studentized']
            other_methods = {'mean': ['wilcoxon', 'ttest'], 'std': ['chi_sq'],
                             'median': ['wilcoxon', 'ci_quant_param', 'ci_quant_nonparam', 'maritz-jarrett'],
                             'percentile_5': ['ci_quant_param', 'ci_quant_nonparam', 'maritz-jarrett'],
                             'percentile_95': ['ci_quant_param', 'ci_quant_nonparam', 'maritz-jarrett'],
                             'corr': ['ci_corr_pearson']}

            possible_methods = bootstrap_methods + other_methods[input.statistic()]
            selected_methods = input.methods()
            ui.update_checkbox_group('methods', choices=possible_methods,
                                     selected=[m for m in possible_methods if m in selected_methods])

    @reactive.Calc
    def filtered_df() -> pd.DataFrame:
        """Returns a Pandas data frame that includes only the desired rows"""
        # This calculation "req"uires that at least one species is selected
        req(len(input.methods()) > 0)

        fil_df = df.copy()
        if 'Distribution' not in [input.xgrid(), input.ygrid()]:
            fil_df = fil_df[(fil_df['dgp'] == input.distribution())]
        if 'Statistic' not in [input.xgrid(), input.ygrid()]:
            fil_df = fil_df[(fil_df['statistic'] == input.statistic())]

        if input.sided() == '1':
            if 'Confidence level' not in [input.xgrid(), input.ygrid()]:
                fil_df = fil_df[(fil_df['alpha'] == float(input.alpha()))]

        else:
            al = round((1 - float(input.alpha())) / 2, 5)
            au = 1 - al
            if 'Confidence level' not in [input.xgrid(), input.ygrid()]:
                fil_df = fil_df[fil_df['alpha'] == au]

            fil_df2 = df.copy()
            if 'Distribution' not in [input.xgrid(), input.ygrid()]:
                fil_df2 = fil_df2[(fil_df2['dgp'] == input.distribution())]
            if 'Statistic' not in [input.xgrid(), input.ygrid()]:
                fil_df2 = fil_df2[(fil_df2['statistic'] == input.statistic())]
            if 'Confidence level' not in [input.xgrid(), input.ygrid()]:
                fil_df2 = fil_df2[fil_df2['alpha'] == al]

            # merge dataframes to be able to subtract coverages of same experiments
            fil_df = pd.merge(fil_df, fil_df2, on=['method', 'dgp', 'statistic', 'n', 'B', 'repetitions'],
                              suffixes=('_au', '_al'))
            fil_df['coverage'] = fil_df['coverage_au'] - fil_df['coverage_al']
            fil_df['alpha'] = fil_df['alpha_au'] - fil_df['alpha_al']

        return fil_df

    @output
    @render.plot
    def coverages():
        """Generates a plot for Shiny to display to the user"""

        # The plotting function to use depends on whether we are plotting grids
        no_grids = input.xgrid() == "No X grid" and input.ygrid() == "No Y grid"

        current_df = filtered_df()
        current_methods = [m for m in input.methods() if m in current_df['method'].unique()]

        if no_grids:

            nm = len(current_methods)
            if nm > 10:
                cols = plt.cm.tab20(np.linspace(0.05, 0.95, nm))
            else:
                cols = plt.cm.tab10(np.linspace(0.05, 0.95, nm))
            colors = {m: c for (m, c) in zip(current_methods, cols)}

            plot_coverage_bars(data=current_df, colors=colors, ci=95, scale='linear', set_ylim=True,
                               order=current_methods, hue='method', x='n')
            # plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f'{value:.2f}'))

            handles, labels = plt.gca().get_legend_handles_labels()
            plt.legend(handles, labels, loc='center left', title="Method", bbox_to_anchor=(1, 0.5))

        else:
            row_col_dict = dict(zip(selectable_cols + ["No X grid", "No Y grid"],
                                    ['alpha', 'statistic', 'dgp', None, None]))

            filters = {'method': input.methods()}
            if 'Confidence level' == input.xgrid():
                filters['alpha'] = [float(a) for a in input.alphas_x()]
            elif 'Confidence level' == input.ygrid():
                filters['alpha'] = [float(a) for a in input.alphas_y()]
            else:
                filters['alpha'] = [float(input.alpha())]
            if 'Statistic' in [input.xgrid(), input.ygrid()]:
                filters['statistic'] = [input.statistics_x() if 'Statistic' == input.xgrid() else input.statistics_y()]
            else:
                filters['statistic'] = [input.statistic()]
            if 'Distribution' in [input.xgrid(), input.ygrid()]:
                filters['distribution'] = [input.distributions_x() if 'Statistic' == input.xgrid() else input.distributions_y()]
            else:
                filters['dgp'] = [input.distribution()]

            compare_cov_dis_grid(df=current_df, comparing='coverage', filter_by=filters, x='n',
                                 row=row_col_dict[input.ygrid()], col=row_col_dict[input.xgrid()],
                                 hue='method', save_add=None, title=None, ci=95, scale='linear',
                                 set_ylim=False, colors=None)


app = App(app_ui, server)
app.run()
