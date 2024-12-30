import math
import os
import sys
from types import NoneType

import numpy as np
import pandas as pd
from scipy import stats

from ppo.audio_feature import AudioFeature
from services import CACHED_PATH
from services.artist_variance_service import ArtistVarianceService
from services.feature_variance_service import FeatureVarianceService
from services.num_edits_service import NumEditsService
from services.playlist_service import PlaylistService
from services.popularity_service import PopularityService
from services.service import Service
from utils.variance_util import get_coherence


def categorizes(data, get_idx, set_idx=None, percentile=None):
    if set_idx is None:
        set_idx = get_idx + 1
    if percentile is None:
        values = np.array([d[get_idx] for d in data if d[get_idx] != np.nan])
        percentile = np.percentile(values, [33, 66])
    for row in data:
        value = row[get_idx]
        if value == np.nan:
            row[set_idx] = np.nan
        elif value <= percentile[0]:
            row[set_idx] = 0
        elif value <= percentile[1]:
            row[set_idx] = 1
        else:
            row[set_idx] = 2


class CoherenceService(Service):

    def __init__(self, min_samples=11, min_threshold=1e-6, shuffled=False, cached_path=CACHED_PATH):
        if shuffled:
            filepath = os.path.join(cached_path, 'coherence_service_shuffled.pk')
        else:
            filepath = os.path.join(cached_path, 'coherence_service.pk')

        self.shuffled = shuffled
        self.min_samples = min_samples
        self.min_threshold = min_threshold

        self.data: pd.DataFrame | NoneType = None
        self.features = list(AudioFeature.feature_labels) + ['artists']

        self.independent_variables = ['c_length', 'c_num_edits', 'c_popularity', 'c_collaborative']
        self.dependent_variables = ['artists', 'loudness', 'energy', 'danceability', 'acousticness', 'valence',
                                    'speechiness', 'instrumentalness', 'liveness', 'tempo', 'tonality']
        super().__init__(filepath)

    def load_from_data(self, playlist_service,
                       num_edits_service: NumEditsService,
                       popularity_service: PopularityService,
                       variance_service: FeatureVarianceService,
                       artist_variance_service: ArtistVarianceService):

        assert variance_service.shuffled == self.shuffled
        assert artist_variance_service.shuffled == self.shuffled

        columns = []
        for var_name in self.independent_variables[:-1]:
            columns.append(var_name[2:])
            if var_name != 'c_popularity':
                columns.append('log_' + var_name[2:])
            columns.append(var_name)
        columns.append('c_collaborative')
        columns += self.features

        index = []
        values = []
        for pid, playlist in playlist_service.playlists.items():
            row = []
            var_dict, s_c, p_c, track_len = variance_service.variances[pid]
            if s_c != track_len - 1 or track_len < self.min_samples:
                continue

            row.append(track_len)
            row.append(math.log(track_len))
            row.append(None)

            num_edits = num_edits_service.num_edits[pid]
            if num_edits < 1:
                continue

            row.append(num_edits)
            row.append(math.log(num_edits))
            row.append(None)

            popularity = popularity_service.popularity[pid]
            row.append(popularity)
            row.append(None)

            row.append(int(playlist.is_collaborative))

            for label, (sq_var, pl_var) in var_dict.items():
                row.append(get_coherence(sq_var, pl_var, self.min_threshold))

            if pid in artist_variance_service.variances:
                sq_var, pl_var, s_c, p_c, track_len = artist_variance_service.variances[pid]
                if s_c < self.min_samples:
                    continue
                row.append(get_coherence(sq_var, pl_var, self.min_threshold))
            else:
                row.append(float('nan'))

            index.append(pid)
            values.append(row)

        categorizes(values, 0, 2, [40, 100])  # length
        categorizes(values, 3, 5, [4, 40])  # num_edits
        categorizes(values, 6, 7)  # popularity

        self.data = pd.DataFrame(values, index=index, columns=columns)

    def check_normality(self):
        for column in self.independent_variables:
            for feature in self.features:
                if column == 'c_collaborative':
                    groups = 2
                else:
                    groups = 3
                for i in range(groups):
                    values = self.data[self.data[column] == i][feature].dropna()
                    result = stats.shapiro(values)
                    # Everything above 95 is good
                    stdout = sys.stderr if result.statistic < 0.95 else sys.stdout
                    print(f'{column} | {feature:10s} | {i} | {result}', file=stdout)

    def check_homogeneity(self):
        dependent_frame = self.data[self.dependent_variables]
        values = dependent_frame.dropna()

        data = [values[f].values for f in self.dependent_variables]
        homogeneity = stats.levene(*data)
        print(f'levene {homogeneity}')
        homogeneity = stats.bartlett(*data)
        print(f'bartlett {homogeneity}')

    def check_size(self):
        rows = []
        for independent_variable in self.independent_variables:
            for dependent_variable in self.features:
                if independent_variable == 'c_collaborative':
                    stacks = [len(self.data[self.data[independent_variable] == i][
                                      [dependent_variable, independent_variable]].dropna()) for i in [True, False]]
                    stacks.append(0)
                else:
                    stacks = [len(self.data[self.data[independent_variable] == i][
                                      [dependent_variable, independent_variable]].dropna()) for i in range(3)]
                rows.append((independent_variable, dependent_variable, *stacks))
        df = pd.DataFrame(rows, columns=['independent_variable', 'dependent_variable', '0', '1', '2'])
        print(df)

    def correlation_analysis(self):
        rows = []
        index = []
        columns = ['size', 'corr', 'p_value']
        for independent_variable in self.independent_variables:
            if independent_variable != 'c_collaborative':
                independent_variable = independent_variable[2:]
            for dependent_variable in self.features:
                data = self.data[[independent_variable, dependent_variable]].dropna()

                if independent_variable == 'collaborative':
                    corr, p_value = stats.pointbiserialr(data[independent_variable].values,
                                                         data[dependent_variable].values)
                else:
                    corr, p_value = stats.pearsonr(data[independent_variable].values, data[dependent_variable].values)
                index.append((independent_variable, dependent_variable))
                rows.append([len(data), corr, p_value])
        return pd.DataFrame(rows, index=pd.MultiIndex.from_tuples(index), columns=columns)

    def get_stats(self):
        rows = []
        index = []
        columns = ['size', 'min', 'q1', 'q2', 'q3', 'max', 'mean']

        for independent_variable in self.independent_variables:
            if independent_variable == 'c_collaborative':
                num_independents = 2
            else:
                num_independents = 3
            for dependent_variable in self.features:
                for i in range(num_independents):
                    feature_values = self.data[self.data[independent_variable] == i][dependent_variable].dropna()
                    row = [len(feature_values), feature_values.min(), feature_values.quantile(0.25),
                           feature_values.median(), feature_values.quantile(0.25), feature_values.max(),
                           feature_values.mean()]
                    index.append((independent_variable, dependent_variable, i))
                    rows.append(row)
        return pd.DataFrame(rows, index=pd.MultiIndex.from_tuples(index), columns=columns)

    def save(self):
        self._save(self.data)

    def load_from_cache(self):
        self.data = self._load_from_cache()

    def get_export_path(self, filepath):
        if self.shuffled:
            return os.path.join(filepath, 'coherence.json')
        else:
            return os.path.join(filepath, 'coherence_shuffled.json')

    def export_to_json(self, filepath='data'):
        self.data.to_json(self.get_export_path(filepath))

    def load_from_json(self, filepath='data'):
        self.data = pd.read_json(self.get_export_path(filepath))

def save(shuffled=False):
    playlist_service = PlaylistService()
    playlist_service.load_from_cache()

    popularity_service = PopularityService()
    popularity_service.load_from_cache()

    num_edits_service = NumEditsService()
    num_edits_service.load_from_cache()

    feature_variance_service = FeatureVarianceService(shuffled=shuffled)
    feature_variance_service.load_from_cache()

    artist_variance_service = ArtistVarianceService(shuffled=shuffled)
    artist_variance_service.load_from_cache()

    coherence_service = CoherenceService(shuffled=shuffled)
    coherence_service.load_from_data(playlist_service, num_edits_service, popularity_service,
                                        feature_variance_service, artist_variance_service)
    coherence_service.save()
    print('Finished')


def load(shuffled=False):
    coherence_service = CoherenceService(shuffled=shuffled)
    coherence_service.load_from_cache()
    coherence_service.check_normality()
    coherence_service.check_homogeneity()
    coherence_service.check_size()
    print('Finished')

def export_to_json(shuffled=False):
    coherence_service = CoherenceService(shuffled=shuffled)
    coherence_service.load_from_cache()
    coherence_service.export_to_json()
    print('Finished')

def load_from_json(shuffled=False):
    coherence_service = CoherenceService(shuffled=shuffled)
    coherence_service.load_from_json()
    print('Finished')

if __name__ == '__main__':
    # save(shuffled=False)
    # load(shuffled=False)
    # save(shuffled=True)
    # load(shuffled=True)
    # export_to_json(shuffled=False)
    # export_to_json(shuffled=True)
    load_from_json(shuffled=False)
