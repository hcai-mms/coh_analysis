import concurrent.futures
import os
from random import Random

from more_itertools import batched

from ppo.audio_feature import AudioFeature
from services import CACHED_PATH
from services.normalized_feature_service import NormalizedFeatureService
from services.playlist_service import PlaylistService
from services.service import Service
from services.track_info_service import TrackInfoService
from utils.variance_util import calc_variances, tonality_distance, pair_distance


def features_to_variance(pid, features, artists, distance, threshold):
    variances = {}
    zipped = pid, list(zip(features['tonality'], artists))
    _, t_sq_var, t_pl_var, t_s_c, t_p_c, t_track_len = calc_variances(zipped, distance=distance, threshold=threshold,
                                                                      f=tonality_distance)

    for label, feature in features.items():
        if label == 'tonality':
            continue
        zipped = pid, list(zip(feature, artists))
        _, sq_var, pl_var, s_c, p_c, track_len = calc_variances(zipped, distance=distance, threshold=threshold,
                                                                f=pair_distance)
        assert s_c == t_s_c
        assert p_c == t_p_c
        assert track_len == t_track_len
        variances[label] = sq_var, pl_var
    variances['tonality'] = t_sq_var, t_pl_var
    return pid, variances, t_s_c, t_p_c, t_track_len


class FeatureVarianceService(Service):

    def __init__(self, distance=1, threshold=0.0, shuffled=False, seed=42, cached_path=CACHED_PATH):
        self.distance = distance
        self.threshold = threshold
        self.variances = {}
        self.shuffled = shuffled
        self.seed = seed
        if shuffled:
            filepath = os.path.join(cached_path, f'feature_variances_{distance:02d}_{threshold:3.1f}_shuffled.pk')
        else:
            filepath = os.path.join(cached_path, f'feature_variances_{distance:02d}_{threshold:3.1f}.pk')
        super().__init__(filepath)

    def _transform(self, playlist, feature_service: NormalizedFeatureService, track_info: TrackInfoService, random):
        features = {label: [] for label in AudioFeature.feature_labels}
        artists = []
        tracks = playlist.tracks
        if self.shuffled:
            random.shuffle(tracks)

        for track in tracks:
            if track not in feature_service:
                for lst in features.values():
                    lst.append(None)
            else:
                feature = feature_service[track]
                for label, lst in features.items():
                    lst.append(feature[label])

            if self.threshold > 0.0:
                track_artists = {track.artist_id}
                if track in track_info:
                    track_artists.update(track_info[track.track_id].artist_ids)
                artists.append(track_artists)
            else:
                artists.append(None)
        return features, artists

    def load_from_data(self, playlists, feature_service: NormalizedFeatureService, track_info: TrackInfoService = None):
        random = Random(self.seed)
        assert self.threshold == 0.0 and track_info is None
        for i, batch in enumerate(batched(playlists, 10000)):
            futures = []
            with concurrent.futures.ProcessPoolExecutor(max_workers=48) as executor:
                print(f"Starting {i}")
                for playlist in batch:
                    playlist_features, playlist_artists = self._transform(playlist, feature_service, track_info, random)
                    future = executor.submit(features_to_variance, pid=playlist.playlist_id,
                                             features=playlist_features, artists=playlist_artists,
                                             distance=self.distance, threshold=self.threshold)
                    futures.append(future)
                print(f"Waiting {i}")
                for f in futures:
                    pid, variances, s_c, p_c, track_len = f.result()
                    self.variances[pid] = variances, s_c, p_c, track_len

    def save(self):
        self._save(self.variances)

    def load_from_cache(self):
        self.variances = self._load_from_cache()


def save(shuffled=False):
    playlist_service = PlaylistService()
    playlist_service.load_from_cache()

    feature_service = NormalizedFeatureService()
    feature_service.load_from_cache()

    feature_analyser = FeatureVarianceService(shuffled=shuffled)
    feature_analyser.load_from_data(playlist_service.playlists.values(), feature_service)
    feature_analyser.save()
    print('Finished')


def load(shuffled=False):
    feature_analyser = FeatureVarianceService(shuffled=shuffled)
    feature_analyser.load_from_cache()
    print('Finished')


if __name__ == '__main__':
    # save(shuffled=False)
    # save(shuffled=True)
    load(False)
