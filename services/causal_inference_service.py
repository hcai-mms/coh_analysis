import os

import numpy as np
import pandas as pd
from causalinference import CausalModel
from scipy.stats import norm

from services import CACHED_PATH
from services.service import Service
from services.coherence_service import CoherenceService


def is_significant(estimator, p_value=0.05 / (11 * 2 * 3 + 11)):
    z = estimator['ate'] / estimator['ate_se']
    p = 2 * (1 - norm.cdf(np.abs(z)))
    return p_value > p


class CausalInferenceService(Service):

    def __init__(self, size=10_000, seed=42, use_cache=True, shuffled=False, estimator='weighting',
                 cached_path=CACHED_PATH):
        assert estimator in ['matching', 'weighting']
        self.estimator = estimator

        self.size = size
        if estimator == 'matching':
            filepath = os.path.join(cached_path, f'matching_service_{size}')
        else:
            filepath = os.path.join(cached_path, f'weighting_service_{size}')
        if shuffled:
            filepath += '_shuffled.pk'
        else:
            filepath += '.pk'

        self.independent_variables = ['c_length', 'c_popularity', 'c_num_edits', 'c_collaborative']
        self.prosperity_features = ['log_length', 'log_num_edits', 'popularity', 'c_collaborative', ]

        self.seed_sequence = np.random.SeedSequence(seed).generate_state(256).tolist()

        self.used_cached = use_cache
        self.shuffled = shuffled
        self.data = {}
        super().__init__(filepath)

    def causal_inference(self, data, rand_int, replacements, categorical, dependent_variable):
        assert categorical in self.independent_variables
        prosperity_features = [f for f in self.prosperity_features if categorical[2:] not in f]
        assert len(prosperity_features) == len(self.prosperity_features) - 1

        results = {}

        r0, r1, r2, r3 = replacements
        assert r1 != r3
        assert r0 != r2

        filtered = data[[categorical, dependent_variable] + prosperity_features].dropna()

        group_a = filtered[filtered[categorical] == r0]
        assert len(group_a) >= self.size

        group_b = filtered[filtered[categorical] == r2]
        assert len(group_b) >= self.size

        if r1:
            group_a = group_a.sample(self.size, random_state=next(rand_int))
            group_b = group_b.sample(90_000, random_state=next(rand_int))
        else:
            group_a = group_a.sample(90_000, random_state=next(rand_int))
            group_b = group_b.sample(self.size, random_state=next(rand_int))

        group_a.loc[:, categorical] = group_a.loc[:, categorical].replace(r0, r1)
        group_b.loc[:, categorical] = group_b.loc[:, categorical].replace(r2, r3)

        concatenated = pd.concat((group_a, group_b))

        perm = np.random.permutation(len(concatenated))

        cm = CausalModel(
            Y=concatenated[dependent_variable].values[perm],
            D=concatenated[categorical].values[perm],
            X=concatenated[prosperity_features].values[perm]
        )

        if self.estimator == 'matching':
            cm.est_via_matching(matches=1, bias_adj=True)
        else:
            cm.est_propensity_s()
            cm.est_via_weighting()

        results['size'] = (len(group_a), len(group_b))
        results['estimates'] = cm.estimates
        results['summary_stats'] = cm.summary_stats

        print(f'### {categorical:10s} {dependent_variable:10s} ###')
        print(cm.estimates)
        return results

    def load_from_data(self, coherence_service: CoherenceService):
        assert self.shuffled == coherence_service.shuffled

        rand_iter = iter(self.seed_sequence)

        if self.used_cached and os.path.exists(self.filepath):
            self.load_from_cache()

        for categorical in self.independent_variables:
            for dependent_variable in coherence_service.features:
                dict_ = self.data.setdefault(categorical, {}).setdefault(dependent_variable, {})
                if categorical == 'c_collaborative':
                    if 0 in dict_:
                        next(rand_iter)
                        next(rand_iter)
                        continue
                    dict_[0] = self.causal_inference(coherence_service.data,
                                                     rand_iter,
                                                     (0, False, 1, True),
                                                     categorical, dependent_variable)
                else:
                    replacements = [(0, False, 1, True), (0, False, 2, True)]
                    for i in range(2):
                        if i in dict_:
                            next(rand_iter)
                            next(rand_iter)
                            continue
                        dict_[i] = self.causal_inference(coherence_service.data,
                                                         rand_iter,
                                                         replacements[i],
                                                         categorical, dependent_variable)
                        if self.shuffled:
                            assert not is_significant(dict_[i]['estimates']['weighting'])
                self.save()

    def save(self):
        self._save(self.data)

    def load_from_cache(self):
        self.data = self._load_from_cache()


def save(shuffled=False):
    coherence_service = CoherenceService(shuffled=shuffled)
    coherence_service.load_from_cache()

    causal_inference_service = CausalInferenceService(shuffled=shuffled)
    causal_inference_service.load_from_data(coherence_service)
    causal_inference_service.save()
    print('Finished')


def load(shuffled=False):
    causal_inference_service = CausalInferenceService(shuffled=shuffled)
    causal_inference_service.load_from_cache()
    print('Finished')


if __name__ == '__main__':
    # save(shuffled=False)
    # save(shuffled=True)
    load(False)
