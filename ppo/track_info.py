from typeguard import typechecked


class TrackInfo:
    @typechecked
    def __init__(self, track_id: str, name: str, album: str, duration: int, explicit: bool | None, is_local: bool | None, popularity: int | None, artist_ids: set[str]):
        self.track_id = track_id
        self.name = name
        self.album = album
        self.duration = duration
        self.explicit = explicit
        self.is_local = is_local
        self.popularity = popularity
        self.artist_ids = artist_ids

    def __eq__(self, other):
        if hasattr(other, "track_id"):
            return self.track_id == other.track_id
        if type(other) == str:
            return self.track_id == other
        raise ValueError('{} is neither str nor Track')

    def __hash__(self):
        return hash(self.track_id)
