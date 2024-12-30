from typeguard import typechecked
from ppo.track import Track


class Playlist:
    @typechecked
    def __init__(self,
                 playlist_id: int,
                 title: str,
                 nb_tracks: int,
                 nb_favorites: int,
                 is_collaborative: bool,
                 modified_at: int,
                 tracks: list[Track]):
        self.playlist_id = playlist_id
        self.title = title
        self.nb_tracks = nb_tracks
        self.nb_favorites = nb_favorites
        self.is_collaborative = is_collaborative
        self.modified_at = modified_at
        self.tracks = tracks
        self._tracks = set(tracks)

    @typechecked
    def __contains__(self, track: Track):
        return track in self._tracks

    def __eq__(self, other):
        if hasattr(other, "playlist_id"):
            return self.playlist_id == other.playlist_id
        if type(other) == int:
            return self.playlist_id == other
        raise ValueError(f'{other} is neither int nor Playlist')

    def __hash__(self):
        return self.playlist_id

    def __str__(self):
        return self.title


class PlaylistTest(Playlist):
    def __init__(self, playlist: Playlist,
                 tracks: list[Track],
                 hidden: list[Track]):
        super(PlaylistTest, self).__init__(
            playlist.playlist_id,
            playlist.title,
            playlist.nb_tracks,
            playlist.nb_favorites,
            playlist.is_collaborative,
            playlist.modified_at,
            tracks)
        self.hidden = hidden
