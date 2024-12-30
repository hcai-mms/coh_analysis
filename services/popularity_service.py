import math
import os

from services import CACHED_PATH
from services.playlist_service import PlaylistService
from services.service import Service


class PopularityService(Service):

    def __init__(self, cached_path=CACHED_PATH):
        filepath = os.path.join(cached_path, 'popularity_service.pk')
        self.popularity = {}
        super().__init__(filepath)

    def load_from_data(self, playlist_service: PlaylistService):
        counts = playlist_service.track_service.count
        for pid, playlist in playlist_service.playlists.items():
            popularity = 0
            for track in playlist.tracks:
                popularity += math.log(counts[track.track_id])
            popularity /= len(playlist.tracks)
            self.popularity[pid] = popularity

    def save(self):
        self._save(self.popularity)

    def load_from_cache(self):
        self.popularity = self._load_from_cache()


def save():
    playlist_service = PlaylistService(filtered=False)
    playlist_service.load_from_cache()

    popularity_service = PopularityService()
    popularity_service.load_from_data(playlist_service)
    popularity_service.save()
    print('Finished')


def load():
    popularity_service = PopularityService()
    popularity_service.load_from_cache()
    print('Finished')


if __name__ == '__main__':
    # save()
    load()
