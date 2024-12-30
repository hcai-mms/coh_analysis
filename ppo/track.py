from typeguard import typechecked


class Track:
    @typechecked
    def __init__(self, track_id: str, artist_id: str, album_id, duration: int):
        self.track_id = track_id
        self.artist_id = artist_id
        self.album_id = album_id
        self.duration = duration

    def __eq__(self, other):
        if hasattr(other, "track_id"):
            return self.track_id == other.track_id
        if type(other) == str:
            return self.track_id == other
        raise ValueError(f'{other} is neither str nor Track')

    def __hash__(self):
        return hash(self.track_id)
