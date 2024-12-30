import concurrent
import os
from more_itertools import batched

import numpy as np
from sklearn.linear_model import LinearRegression

from services import CACHED_PATH
from services.artist_matrix_service import ArtistMatrixService
from services.artist_variance_service import embedding_similarity
from services.normalized_feature_service import NormalizedFeatureService
from services.playlist_service import PlaylistService
from services.service import Service

from services.coherence_service import CoherenceService
from utils.variance_util import tonality_distance, pair_distance

matrix = None
def calc_distances(features):
    average_distances = []

    for i, a_feature in enumerate(features[:-1]):
        if a_feature is None:
            continue

        b_feature = features[i+1]
        if b_feature is None:
            continue
        a_iter = iter(a_feature)
        b_iter = iter(b_feature)

        distances = [pair_distance(a_f, b_f) for _, a_f, b_f in zip(range(9), a_iter, b_iter)]
        distances.append(tonality_distance(next(a_iter), next(b_iter)))
        a_embedding = matrix[:, next(a_iter)]
        b_embedding = matrix[:, next(b_iter)]

        #distances.append(embedding_similarity(next(a_iter), next(b_iter)))
        distances.append(embedding_similarity(a_embedding, b_embedding))
        average_distances.append(distances)
    average_distances = np.array(average_distances)
    return np.mean(average_distances, axis=0)


class EstimatorService(Service):
    def __init__(self, cached_path=CACHED_PATH):
        filepath = os.path.join(cached_path, 'estimator_service.pk')
        self.coherence_estimator = LinearRegression()
        self.transition_estimator = LinearRegression()
        self.attributes = ['length', 'log_length', 'num_edits', 'log_num_edits', 'popularity']
        super().__init__(filepath)

    def train_transitions(self, playlist_service, feature_service: NormalizedFeatureService,
                          matrix, track_to_ids, coherence_service: CoherenceService):

        X = []
        Y = []

        significance_iter = coherence_service.data.iterrows()
        for i, batch in enumerate(batched( significance_iter, 2048)):
            futures = []
            with (concurrent.futures.ProcessPoolExecutor(max_workers=48) as executor):
                print(f"Starting {i}")
                for pid, row in batch:
                    playlist = playlist_service[pid]
                    tracks = playlist.tracks

                    features = []
                    for track in tracks:
                        track_id = track.track_id
                        if track_id not in feature_service:
                            features.append(None)
                        else:
                            vector = list(feature_service[track_id])
                            vector.append(track_to_ids[track_id])
                            features.append(vector)

                    futures.append(executor.submit(calc_distances, features))
                    X.append(row[self.attributes].values)

                print(f"Waiting {i}")
                for f in futures:
                    Y.append(f.result())

        X = np.array(X)
        Y = np.array(Y)
        print("training transition")
        self.transition_estimator.fit(X, Y)
        print("finished training transition")

    def load_from_data(self, playlist_service, feature_service: NormalizedFeatureService,
                     matrix, track_to_ids, coherence_service: CoherenceService):
        self.train_transitions(playlist_service, feature_service, matrix, track_to_ids, coherence_service)
        self.save()

        data = coherence_service.data[self.attributes + coherence_service.features].dropna()
        X = data[self.attributes].values
        Y = data[coherence_service.features].values
        print("training coherence")
        self.coherence_estimator.fit(X, Y)
        print("finished training coherence")
        self.save()

    def save(self):
        self._save((self.coherence_estimator, self.transition_estimator))

    def load_from_cache(self):
        self.coherence_estimator, self.transition_estimator = self._load_from_cache()


def save():
    playlist_service = PlaylistService()
    playlist_service.load_from_cache()

    coherence_service = CoherenceService()
    coherence_service.load_from_cache()

    feature_service = NormalizedFeatureService()
    feature_service.load_from_cache()

    artist_matrix_service = ArtistMatrixService()
    artist_matrix_service.load_from_cache()

    global matrix
    matrix = artist_matrix_service.matrix.tocsc()
    track_to_ids = artist_matrix_service.track_to_ids

    estimator_service = EstimatorService()
    estimator_service.load_from_data(playlist_service, feature_service, matrix, track_to_ids, coherence_service)
    print('Finished')


def load():
    estimator_service = EstimatorService()
    estimator_service.load_from_cache()
    print('Finished')


if __name__ == '__main__':
    save()
    # load()
