import json as json_reader
import os
from typing import Dict

from ppo.playlist import Playlist
from ppo.track import Track
from services import CACHED_PATH
from services.service import Service
from services.track_service import TrackService
from utils.bool_util import str_to_bool


class PlaylistService(Service):
    def __init__(self, filtered=True, cached_path=CACHED_PATH):
        if filtered:
            filepath = os.path.join(cached_path, f'playlist_service_filtered.pk')
        else:
            filepath = os.path.join(cached_path, f'playlist_service_all.pk')
        self.size = 1_000_000
        self.filtered = filtered
        self.playlists: Dict[str, Playlist] = {}
        self.track_service: TrackService | None = None
        super().__init__(filepath)

    def load_from_data(self, directory='data/mpd.v1'):
        self.track_service = TrackService()
        for filename in next(os.walk(directory))[2]:
            filepath = os.path.join(directory, filename)
            data = json_reader.load(open(filepath))
            playlist_list = data['playlists']

            for json in playlist_list:
                if len(self.playlists) >= self.size:
                    return
                playlist_id = json['pid']
                title = json['name'].lower()
                num_followers = json['num_followers']
                nb_tracks = json['num_tracks']
                is_collaborative = str_to_bool(json['collaborative'])
                modified_at = json['modified_at']

                tracks = []
                for track_json in json['tracks']:
                    track_id = track_json['track_uri'].split(':')[2]
                    artist_id = track_json['artist_uri'].split(':')[2]
                    album_id = track_json['album_uri'].split(':')[2]
                    duration = track_json['duration_ms'] // 1000
                    track = self.track_service.add_track(Track(track_id, artist_id, album_id, duration))
                    tracks.append(track)

                playlist = Playlist(playlist_id, title, nb_tracks, num_followers, is_collaborative, modified_at, tracks)
                assert len(playlist.tracks) == nb_tracks
                if not self.filtered or len(playlist.tracks) == len(playlist._tracks) <= 250:
                    self.playlists[playlist_id] = playlist

    def save(self):
        self._save((self.playlists, self.track_service))

    def load_from_cache(self):
        self.playlists, self.track_service = self._load_from_cache()

    def __contains__(self, item):
        if type(item) == Playlist:
            idx = item.playlist_id
        else:
            idx = item
        return idx in self.playlists

    def __getitem__(self, item):
        if type(item) == Playlist:
            idx = item.playlist_id
        else:
            idx = item
        return self.playlists[idx]

def save(filtered=True):
    playlist_service = PlaylistService(filtered=filtered)
    playlist_service.load_from_data()
    playlist_service.save()
    print(f"saved {len(playlist_service.playlists)} playlists")


def load():
    playlist_service = PlaylistService()
    playlist_service.load_from_cache()
    print('Finished')


if __name__ == '__main__':
    # save()
    load()
