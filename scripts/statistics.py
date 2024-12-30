from services.feature_service import FeatureService
from services.num_edits_service import NumEditsService
from services.playlist_service import PlaylistService
from services.track_info_service import TrackInfoService


def missing_audio_features(playlist_service: PlaylistService, feature_service: FeatureService):
    total_tracks = len(playlist_service.track_service.tracks)
    track_counter = sum(1 for track_id in playlist_service.track_service.tracks if track_id in feature_service)

    print(f'Tracks: f{track_counter} from {total_tracks} -> {track_counter / total_tracks}')

    total_entries = sum(counts for _, counts in playlist_service.track_service.count.items())
    entry_counter = sum(counts for track_id, counts in playlist_service.track_service.count.items() if track_id in feature_service)

    print(f'Entries: f{entry_counter} from {total_entries} -> {entry_counter / total_entries}')


def num_edits_stats(playlist_service: PlaylistService, num_edits_service: NumEditsService):
    num_edits = [edits for pid, edits in num_edits_service.num_edits.items() if pid in playlist_service]
    one_edits = sum([1 for e in num_edits if e == 1])
    two_edits = sum([1 for e in num_edits if e == 2])

    print(f'Average #Edits: {sum(num_edits) / len(num_edits)}')
    print(f'One Edit: {one_edits} from {len(num_edits)} -> {one_edits / len(num_edits)}')
    print(f'Two #Edits: {two_edits} from {len(num_edits)} -> {two_edits / len(num_edits)}')

def artist_albums(playlist_service: PlaylistService):
    num_albums = [len(set(t.album_id for t in playlist.tracks)) for playlist in playlist_service.playlists.values()]
    num_artists = [len(set(t.artist_id for t in playlist.tracks)) for playlist in playlist_service.playlists.values()]

    print(f'Min Albums: {min(num_albums)}')
    print(f'Min Artists: {min(num_artists)}')


def print_collaborative(playlist_service: PlaylistService):
    total = len(playlist_service.playlists)
    collabs = sum(1 for pl in playlist_service.playlists.values() if pl.is_collaborative)

    print(f'Collaborative: f{collabs} from {total} -> {collabs / total}')


def track_info(playlist_service: PlaylistService):
    track_counts = playlist_service.track_service.count
    total = len(track_counts)

    single_usages = sum(1 for c in track_counts.values() if c == 1)

    print(f'Single Tracks: f{single_usages} from {total} -> {single_usages / total}')


def artist_counts(playlist_service: PlaylistService, track_info_service: TrackInfoService):
    artists = set()
    for track in playlist_service.track_service.tracks.values():
        artists.add(track.artist_id)

    print(f'Total Artists: {len(artists)}')

    for track in track_info_service.track_info.values():
        artists.update(track.artist_ids)

    print(f'Total Artists after Update: {len(artists)}')


def main():
    track_info_service = TrackInfoService()
    track_info_service.load_from_cache()

    playlist_service = PlaylistService(filtered=False)
    playlist_service.load_from_cache()
    print(f'#Playlists: {len(playlist_service.playlists)}')

    feature_service = FeatureService()
    feature_service.load_from_cache()

    num_edits_service = NumEditsService()
    num_edits_service.load_from_cache()

    artist_albums(playlist_service)
    missing_audio_features(playlist_service, feature_service)
    num_edits_stats(playlist_service, num_edits_service)
    print_collaborative(playlist_service)
    track_info(playlist_service)

    artist_counts(playlist_service, track_info_service)


if __name__ == '__main__':
    main()



