import os

from services import CACHED_PATH
from services.playlist_service import PlaylistService

from services.service import Service


class BayesianService(Service):
    def __init__(self, alpha=1e-5, threshold=0.1, cached_path=CACHED_PATH):
        self.alpha = alpha
        self.threshold = threshold
        self.mapping = {}

        filepath = os.path.join(cached_path, f'bayesian_service.pk')
        super().__init__(filepath)

    @staticmethod
    def _create_neighbors_dict(playlist_service: PlaylistService):
        neighbors_dict = {}
        for i, (pid, playlist) in enumerate(playlist_service.playlists.items()):
            if i % 10000 == 0:
                print(f'creating neighboring dict {i}')
            tracks = playlist.tracks
            for i in range(len(tracks)-1):
                track = tracks[i]
                next_track = tracks[i+1]
                neighbors = neighbors_dict.setdefault(track, {})
                neighbors.setdefault(next_track, []).append(pid)
        return neighbors_dict

    def load_from_data(self, playlist_service: PlaylistService):
        neighbors_dict = self._create_neighbors_dict(playlist_service)
        M = len(neighbors_dict)
        for i, (a_track, neighbors) in enumerate(neighbors_dict.items()):
            if i % 10000 == 0:
                print(f'searching for track sequences {i}')
            denominator = sum(len(pids) for pids in neighbors.values()) + self.alpha*M
            for b_track, pids in neighbors.items():
                likelihood = (self.alpha + len(pids)) / denominator
                if likelihood > self.threshold:
                    self.mapping.setdefault(a_track, {})[b_track] = likelihood, len(pids)

    def save(self):
        self._save(self.mapping)

    def load_from_cache(self):
        self.mapping = self._load_from_cache()


def save():
    playlist_service = PlaylistService()
    playlist_service.load_from_cache()

    bayesian_service = BayesianService()
    bayesian_service.load_from_data(playlist_service)
    bayesian_service.save()
    print(len(bayesian_service.mapping))


def load():
    bayesian_service = BayesianService()
    bayesian_service.load_from_cache()
    print("Loaded")


if __name__ == '__main__':
    save()
    # load()
