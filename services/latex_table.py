import numpy as np
from scipy.stats import norm

from services.causal_inference_service import CausalInferenceService
from services.coherence_service import CoherenceService


def add_matching(row, estimate, p_value):
    z = estimate['ate'] / estimate['ate_se']
    if 2 * (1 - norm.cdf(np.abs(z))) > p_value:
        row.append(r'\color{gray} ' + f"{estimate['ate']:.3f}")
    else:
        row.append(f"{estimate['ate']:.3f}")


def add_p_value(row, estimate):
    z = estimate['ate'] / estimate['ate_se']
    p_value = 2 * (1 - norm.cdf(np.abs(z)))
    formated = f"{p_value:.4f}".replace('0.', '.')
    if formated == '.0000':
        row.append('$<.0001$')
    else:
        row.append('$' + formated + '$')


def add_corr(row, corr, p_value):
    if corr['p_value'] <= p_value:
        row.append(f"{corr['corr']:.3f}")
    else:
        row.append(r'\color{gray} ' + f"{corr['corr']:.3f}")
    formated = f"{corr['p_value']:.4f}".replace('0.', '.')
    if formated == '.0000':
        row.append('$<.0001$')
    else:
        row.append('$' + formated + '$')


def print_table(coherence_service: CoherenceService,
                matching_service: CausalInferenceService,
                p_value=0.05 / 121):
    correlations = coherence_service.correlation_analysis()
    statistics = coherence_service.get_stats()

    print(r'\begin{table}[p]')
    print(r'\centering')
    print(r'\resizebox{\columnwidth}{!}{%')
    print(r'\begin{tabular}{lrrr|rrr|cc|rr}')

    for label, independent_variable in zip(['Length', r'\#Edits', 'Track Popularity', 'Favorites', 'Collaborative'],
                                           coherence_service.independent_variables):
        if independent_variable == 'c_favorites':
            continue
        print(r"\multicolumn{11}{c}{\textbf{", label, r"}} \\ \hline")

        if independent_variable == 'c_collaborative':
            print(
                r"Coherence & \multicolumn{3}{c}{Point-biserial} & \multicolumn{3}{c}{means} & \multicolumn{4}{c}{ATE} \\ \hline")
            print(
                r" & \multicolumn{1}{c}{corr} & \multicolumn{1}{c}{P} & \multicolumn{1}{c}{N} & \multicolumn{1}{c}{$c_1$} & \multicolumn{1}{c}{$c_2$} &  "
                r"& \multicolumn{1}{c}{$c_1 \Arrow{0.8em} c_2$}  &  & \multicolumn{1}{c}{$\text{P}_{c_1 \Arrow{0.8em} c_2}$}  & \\ \hline")
        else:
            print(
                r"Coherence & \multicolumn{3}{c}{Pearson} & \multicolumn{3}{c}{means} & \multicolumn{4}{c}{ATE} \\ \hline")
            print(
                r" & \multicolumn{1}{c}{corr} & \multicolumn{1}{c}{P} & \multicolumn{1}{c}{N} & \multicolumn{1}{c}{$c_1$} & \multicolumn{1}{c}{$c_2$} & \multicolumn{1}{c}{$c_3$}  "
                r"& \multicolumn{1}{c}{$c_1 \Arrow{0.8em} c_2$}  & \multicolumn{1}{c}{$c_1 \Arrow{0.8em} c_3$}  & \multicolumn{1}{c}{$\text{P}_{c_1 \Arrow{0.8em} c_2}$}  & \multicolumn{1}{c}{$\text{P}_{c_1 \Arrow{0.8em} c_3}$} \\ \hline")

        for dependent_variable in coherence_service.dependent_variables:
            row = [f"{dependent_variable:10s}"]
            if independent_variable == 'c_collaborative':
                corr = correlations.loc[(independent_variable, dependent_variable)]
                add_corr(row, corr, p_value)
                row.append(f"{int(corr['size'] // 1000)}k")

                m1 = statistics.loc[(independent_variable, dependent_variable, 0)]['mean']
                m2 = statistics.loc[(independent_variable, dependent_variable, 1)]['mean']
                row.append(f"{m1:.3f}")
                row.append(f"{m2:.3f}")
                row.append(" ")

                estimate = matching_service.data[independent_variable][dependent_variable][0]['estimates']['weighting']
                add_matching(row, estimate, p_value)
                row.append(" ")
                add_p_value(row, estimate)
                row.append(" ")
            else:
                corr = correlations.loc[(independent_variable[2:], dependent_variable)]
                add_corr(row, corr, p_value)
                row.append(f"{int(corr['size'] // 1000)}k")

                means = [statistics.loc[(independent_variable, dependent_variable, i)]['mean'] for i in range(3)]
                row += [f"{m:.3f}" for m in means]

                estimate_0 = matching_service.data[independent_variable][dependent_variable][0]['estimates'][
                    'weighting']
                estimate_1 = matching_service.data[independent_variable][dependent_variable][1]['estimates'][
                    'weighting']
                add_matching(row, estimate_0, p_value)
                add_matching(row, estimate_1, p_value)
                add_p_value(row, estimate_0)
                add_p_value(row, estimate_1)

            print(' & '.join(row), r'\\')
        if independent_variable == 'c_collaborative':
            print(r'\hline')
        else:
            print(r'\hline \\')
    print(r'\end{tabular}')
    print('}')
    print(r'\end{table}')


def save(shuffled=False):
    coherence_service = CoherenceService(shuffled=shuffled)
    coherence_service.load_from_cache()

    causal_inference_service = CausalInferenceService(shuffled=shuffled)
    causal_inference_service.load_from_cache()
    print_table(coherence_service, causal_inference_service)
    print()


if __name__ == '__main__':
    save(shuffled=False)
    save(shuffled=True)
