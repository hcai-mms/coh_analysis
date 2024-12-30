import os
import pandas as pd
import numpy as np

from ppo.audio_feature import AudioFeature
from services import CACHED_PATH
from algorithm.bayesian_service import BayesianService
from algorithm.estimator_service import EstimatorService
from algorithm.node import Node
from services.artist_matrix_service import ArtistMatrixService
from services.normalized_feature_service import NormalizedFeatureService
from services.playlist_service import PlaylistService
from services.service import Service
from services.coherence_service import CoherenceService
from utils.variance_util import pair_distance, tonality_distance
from utils.variance_util import sequential_variance as f_sequential_variance
from utils.variance_util import playlist_variance as f_playlist_variance
from services.artist_variance_service import sequential_variance as artist_sequential_variance
from services.artist_variance_service import playlist_variance as artist_playlist_variance


def get_pl_variance(nodes: list[Node]):
    pl_variances = []
    for label in AudioFeature.feature_labels:
        features = []
        for node in nodes:
            features += [f[label] for f in node.features]
        features = [(f, None) for f in features]
        if label == 'tonality':
            f = tonality_distance
        else:
            f = pair_distance
        p, p_c = f_playlist_variance(features, 0.0, f=f)
        pl_variances.append(p)

    artists = []
    for node in nodes:
        artists += [f for f in node.artists]
    p, p_c = artist_playlist_variance(artists)
    pl_variances.append(p)
    return np.array(pl_variances)


def get_sq_variance(nodes: list[Node]):
    sq_variances = []
    for label in AudioFeature.feature_labels:
        features = []
        for node in nodes:
            features += [f[label] for f in node.features]
        features = [(f, None) for f in features]
        if label == 'tonality':
            f = tonality_distance
        else:
            f = pair_distance
        s, s_c = f_sequential_variance(features, 1, 0.0, f=f)
        sq_variances.append(s)

    artists = []
    for node in nodes:
        artists += [f for f in node.artists]
    s, s_c = artist_sequential_variance(artists)
    sq_variances.append(s)
    return np.array(sq_variances)


def get_coherence(nodes: list[Node]):
    return 1.0 - get_sq_variance(nodes) / get_pl_variance(nodes)


class AlgorithmService(Service):
    def __init__(self, pid, bayesian_threshold=0.70, cached_path=CACHED_PATH):
        self.pid = pid
        self.bayesian_threshold = bayesian_threshold
        filepath = os.path.join(cached_path, f'algorithm_{pid}.pk')

        self.playlist = None
        self.nodes = []

        self.target_coherence = np.zeros(11, dtype=float)
        self.target_transition = np.zeros(11, dtype=float)
        self.pl_variance = np.zeros(11, dtype=float)

        super().__init__(filepath)

    def _set_targets(self, estimator_service: EstimatorService, coherence_service: CoherenceService):
        X = coherence_service.data.loc[[self.pid]][estimator_service.attributes].values
        self.target_coherence = estimator_service.coherence_estimator.predict(X)[0]
        self.target_transition = estimator_service.transition_estimator.predict(X)[0]

    def _create_node_system(self, bayesian_service: BayesianService,
                            feature_service, matrix, track_to_ids):
        i = 0
        self.nodes = []
        tracks = self.playlist.tracks

        while i < len(tracks):
            a_track = tracks[i]
            node_tracks = [a_track]
            node_name = [i+1]
            for j in range(i + 1, len(tracks)):
                b_track = tracks[j]
                try:
                    score, counts = bayesian_service.mapping[a_track][b_track]
                    if score > self.bayesian_threshold:
                        node_tracks.append(b_track)
                        node_name.append(j+1)
                        i = j
                        continue
                except KeyError:
                    break
            node = Node.create_node(str(node_name), node_tracks, feature_service, matrix, track_to_ids)
            self.nodes.append(node)
            i+=1

    def get_coherence(self, nodes: list[Node]):
        return 1.0 - get_sq_variance(nodes) / self.pl_variance

    def load_from_data(self,
                       playlist_service: PlaylistService,
                       estimator_service: EstimatorService,
                       coherence_service: CoherenceService,
                       bayesian_service: BayesianService,
                       feature_service, matrix, track_to_ids):
        self.playlist = playlist_service[self.pid]
        self._set_targets(estimator_service, coherence_service)
        self._create_node_system(bayesian_service, feature_service, matrix, track_to_ids)
        self.pl_variance = get_pl_variance(self.nodes)

    def coherence_error(self, nodes: list[Node]):
        return np.sum(np.square(self.target_coherence - self.get_coherence(nodes)))

    def transition_error(self, i, current: list[Node], previous: list[Node]):
        current_dist = current[i] - current[i+1]
        previous_dist = previous[i] - previous[i + 1]

        error = np.square(current_dist - self.target_transition) - np.square(previous_dist - self.target_transition)
        return np.sum(error)

    @staticmethod
    def get_new_transitions(i, dist, current: list[Node]):
        if dist == 1:
            if len(current[i].tracks) == 1 and len(current[i+1].tracks) == 1:
                return [i-1, i+1]
            else:
                return [i - 1, i, i + 1]
        else:
            return [i-1, i, i+dist-1, i+dist]


    def train(self):
        playlist = self.nodes.copy()
        playlist_coh_error = self.coherence_error(playlist)
        print(f"{0:3d}: dist={0:3d}, coh_error={playlist_coh_error:.4f}")

        iteration = 0
        total_trans_error = 0
        dist = 1
        N = len(playlist)
        while dist < N-2:
            min_trans_error = float('inf')
            playlist_best = None
            for i in range(1, N-dist-1):
                playlist_prime = playlist.copy()
                playlist_prime[i], playlist_prime[i+dist] = playlist_prime[i+dist], playlist_prime[i]
                playlist_prime_coh_error = self.coherence_error(playlist_prime)
                if playlist_prime_coh_error >= playlist_coh_error:
                    continue

                trans_error = sum(self.transition_error(j, playlist_prime, playlist) for j in
                                  self.get_new_transitions(i, dist, playlist_prime))
                if trans_error < min_trans_error:
                    min_trans_error = trans_error
                    playlist_best = playlist_prime
            if playlist_best is not None:
                iteration += 1
                total_trans_error += min_trans_error
                playlist = playlist_best
                playlist_coh_error = self.coherence_error(playlist_best)
                print(f"{iteration:3d}: dist={dist:3d}, coh_error={playlist_coh_error:.4f}, trans_error={min_trans_error: .4f}")
                dist = 1
            else:
                dist += 1
        print("Total trans error:", total_trans_error / iteration)
        return playlist

    @staticmethod
    def print_nodes(nodes):
        for node in nodes:
            track_str = []
            for track in node.tracks:
                track_str.append(track.track_id)
            print(node.name, ", ".join(track_str))

    def print_stats(self, reordered):
        print("Original Coherence Error: ", self.coherence_error(self.nodes))
        print("Reordered Coherence Error:", self.coherence_error(reordered))

        original_coherence = self.get_coherence(self.nodes)
        reordered_coherence = self.get_coherence(reordered)

        labels = list(AudioFeature.feature_labels) + ['artists']
        rows = list(zip(labels, self.target_coherence, original_coherence, reordered_coherence))
        columns = ['Feature', 'Target', 'Original', 'Reordered']
        df = pd.DataFrame(rows, columns=columns)
        print(df)


    def save(self):
        self._save((self.playlist, self.nodes, self.target_coherence, self.target_transition, self.pl_variance))

    def load_from_cache(self):
        self.playlist, self.nodes, self.target_coherence, self.target_transition, self.pl_variance = self._load_from_cache()


def save(pid=195677):
    playlist_service = PlaylistService()
    playlist_service.load_from_cache()

    coherence_service = CoherenceService()
    coherence_service.load_from_cache()

    estimator_service = EstimatorService()
    estimator_service.load_from_cache()

    bayesian_service = BayesianService()
    bayesian_service.load_from_cache()

    feature_service = NormalizedFeatureService()
    feature_service.load_from_cache()

    artist_matrix_service = ArtistMatrixService()
    artist_matrix_service.load_from_cache()
    matrix = artist_matrix_service.matrix.tocsc()
    track_to_ids = artist_matrix_service.track_to_ids

    algorithm_service = AlgorithmService(pid)
    algorithm_service.load_from_data(playlist_service,
                       estimator_service,
                       coherence_service,
                       bayesian_service,
                       feature_service, matrix, track_to_ids)
    algorithm_service.save()


def train(pid=195677):
    algorithm_service = AlgorithmService(pid)
    algorithm_service.load_from_cache()
    algorithm_service.print_nodes(algorithm_service.nodes)
    reordered = algorithm_service.train()
    algorithm_service.print_nodes(reordered)
    algorithm_service.print_stats(reordered)


if __name__ == '__main__':
    # save()
    train()

