from typing import Dict

from typeguard import typechecked

from ppo.track import Track


class TrackService:

    def __init__(self):
        self.tracks: Dict[str, Track] = {}
        self.count: Dict[str, int] = {}

    @typechecked
    def add_track(self, track: Track):
        track_id = track.track_id
        if track_id in self.tracks:
            self.count[track_id] += 1
            track = self.tracks[track_id]
        else:
            self.tracks[track_id] = track
            self.count[track_id] = 1
        return track

    def get_popularity(self, track: Track):
        return self.count[track.track_id]
