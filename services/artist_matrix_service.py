import os

import numpy as np
from bidict import bidict
from scipy.sparse import lil_matrix, coo_matrix

from services import CACHED_PATH
from services.playlist_service import PlaylistService
from services.service import Service
from services.track_info_service import TrackInfoService


class ArtistMatrixService(Service):

    def __init__(self, cached_path=CACHED_PATH):
        filepath = os.path.join(cached_path, 'artist_matrix_service.pk')
        self.artist_to_id = bidict()
        self.track_to_ids = {}
        self.artist_in_playlist_count = {}
        self.matrix = coo_matrix((0, 0), dtype=np.float32)
        super().__init__(filepath)

    def init_mapping(self, playlist_service: PlaylistService, track_service: TrackInfoService):
        data = {}
        for pid, playlist in playlist_service.playlists.items():
            playlist_artist_ids = {}
            for track in playlist.tracks:
                track_id = track.track_id
                if track_id in self.track_to_ids:
                    track_artist_ids = self.track_to_ids[track_id]
                else:
                    track_artists = {track.artist_id}
                    if track.track_id in track_service:
                        track_artists.update(track_service[track_id].artist_ids)

                    track_artist_ids = []
                    for artist in track_artists:
                        if artist in self.artist_to_id:
                            artist_id = self.artist_to_id[artist]
                        else:
                            artist_id = len(self.artist_to_id)
                            self.artist_to_id[artist] = artist_id
                        track_artist_ids.append(artist_id)
                    self.track_to_ids[track_id] = track_artist_ids
                for artist_id in track_artist_ids:
                    playlist_artist_ids[artist_id] = playlist_artist_ids.get(artist_id, 0) + 1

            for artist_id in playlist_artist_ids:
                self.artist_in_playlist_count[artist_id] = self.artist_in_playlist_count.get(artist_id, 0) + 1
            data[pid] = playlist_artist_ids
        return data

    def permutate_data(self, data):
        n = len(self.artist_to_id)
        permutation = np.random.permutation(n)

        perm_track_to_ids = {}
        for track_id, artist_ids in self.track_to_ids.items():
            perm_track_to_ids[track_id] = [permutation[artist_id] for artist_id in artist_ids]
        self.track_to_ids = perm_track_to_ids

        perm_artist_to_id = bidict()
        for artist, artist_id in self.artist_to_id.items():
            perm_artist_to_id[artist] = permutation[artist_id]
        self.artist_to_id = perm_artist_to_id

        perm_data = {}
        for pid, playlist_artist_ids in data.items():
            perm_playlist_artist_ids = {}
            for artist_id, count in playlist_artist_ids.items():
                perm_playlist_artist_ids[permutation[artist_id]] = count
            perm_data[pid] = perm_playlist_artist_ids

        return perm_data

    def normalize_data(self, data):
        for pid, playlist_artist_ids in data.items():
            max_term = max(playlist_artist_ids.values())
            for artist_id, count in playlist_artist_ids.items():
                tf = count / max_term
                idf = np.log(len(data) / self.artist_in_playlist_count[artist_id])
                playlist_artist_ids[artist_id] = tf * idf

    def init_matrix(self, data):
        n = len(self.artist_to_id)
        m = len(data)
        matrix = lil_matrix((m, n), dtype=np.float32)
        for i, playlist_artist_ids in enumerate(data.values()):
            for artist_id, count in playlist_artist_ids.items():
                matrix[i, artist_id] = count
        return coo_matrix(matrix)

    def load_from_data(self, playlist_service: PlaylistService, track_service: TrackInfoService):
        data = self.init_mapping(playlist_service, track_service)
        data = self.permutate_data(data)
        self.normalize_data(data)
        self.matrix = self.init_matrix(data)

    def save(self):
        self._save((self.artist_to_id, self.track_to_ids, self.artist_in_playlist_count, self.matrix))

    def load_from_cache(self):
        self.artist_to_id, self.track_to_ids, self.artist_in_playlist_count, self.matrix = self._load_from_cache()


def save():
    playlist_service = PlaylistService()
    playlist_service.load_from_cache()

    track_service = TrackInfoService()
    track_service.load_from_cache()

    artist_matrix_service = ArtistMatrixService()
    artist_matrix_service.load_from_data(playlist_service, track_service)
    artist_matrix_service.save()
    print('Finished')


def load():
    artist_matrix_service = ArtistMatrixService()
    artist_matrix_service.load_from_cache()
    print('Finished')


if __name__ == '__main__':
    # save()
    load()
