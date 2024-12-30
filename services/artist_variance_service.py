import concurrent.futures
import os
import time
from random import Random

import numpy as np
from more_itertools import batched
from sklearn.metrics.pairwise import cosine_distances

from services import CACHED_PATH
from services.artist_matrix_service import ArtistMatrixService
from services.playlist_service import PlaylistService
from services.service import Service


def embedding_similarity(a_embeddings, b_embeddings):
    return np.mean(cosine_distances(a_embeddings.T, b_embeddings.T))


def playlist_variance(embeddings):
    sum_ = 0
    n = len(embeddings)
    for i, a_track in enumerate(embeddings[:-1]):
        for j, b_track in enumerate(embeddings[i + 1:]):
            sum_ += embedding_similarity(a_track, b_track) ** 2
    div = n * (n - 1)
    return sum_ / div, div // 2


def sequential_variance(embeddings):
    sum_ = 0
    n = len(embeddings)
    for i, a_artists in enumerate(embeddings[:-1]):
        b_artists = embeddings[i + 1]
        sum_ += embedding_similarity(a_artists, b_artists) ** 2

    return sum_ / (n - 1) / 2, n - 1


def embedding_to_variance(pid, embeddings):
    pl_var, p_c = playlist_variance(embeddings)
    sq_var, s_c = sequential_variance(embeddings)
    return pid, sq_var, pl_var, s_c, p_c, len(embeddings)


class ArtistVarianceService(Service):
    def __init__(self, shuffled=False, seed=42, cached_path=CACHED_PATH):
        self.shuffled = shuffled
        self.seed = seed
        self.cached_path = cached_path

        if shuffled:
            filepath = os.path.join(cached_path, f'artist_variance_service_shuffled.pk')
        else:
            filepath = os.path.join(cached_path, f'artist_variance_service.pk')

        self.variances = {}
        super().__init__(filepath)

    def load_from_data(self, playlists, artist_matrix_service: ArtistMatrixService):
        random = Random(self.seed)
        start_time = time.time()
        track_to_ids = artist_matrix_service.track_to_ids
        matrix = artist_matrix_service.matrix.tocsc()

        if os.path.exists(self.filepath):
            self.load_from_cache()

        for i, batch in enumerate(batched(playlists, 2350)):
            print(f"--- {(time.time() - start_time)} seconds for {i} ---")
            start_time = time.time()
            current_size = len(self.variances)

            futures = []
            with concurrent.futures.ProcessPoolExecutor(max_workers=47) as executor:
                print(f"Starting {i}")
                for playlist in batch:
                    if playlist.playlist_id in self.variances:
                        continue
                    track_ids = [t.track_id for t in playlist.tracks]
                    if self.shuffled:
                        random.shuffle(track_ids)
                    embeddings = [matrix[:, track_to_ids[tid]] for tid in track_ids]

                    future = executor.submit(embedding_to_variance, pid=playlist.playlist_id, embeddings=embeddings)
                    futures.append(future)
                print(f"Waiting {i}")
                for f in futures:
                    pid, sq_var, pl_var, s_c, p_c, track_len = f.result()
                    self.variances[pid] = sq_var, pl_var, s_c, p_c, track_len
            if current_size < len(self.variances):
                print(f"Saving {i}")
                self.save()
                print(f"Done Saving {i}")

    def save(self):
        self._save(self.variances)

    def load_from_cache(self):
        self.variances = self._load_from_cache()


def save(shuffled=False):
    playlist_service = PlaylistService()
    playlist_service.load_from_cache()

    artist_matrix_service = ArtistMatrixService()
    artist_matrix_service.load_from_cache()

    artist_variance_service = ArtistVarianceService(shuffled=shuffled)
    artist_variance_service.load_from_data(playlist_service.playlists.values(), artist_matrix_service)
    artist_variance_service.save()
    print('Finished')


def load(shuffled=False):
    artist_variance_service = ArtistVarianceService(shuffled=shuffled)
    artist_variance_service.load_from_cache()


if __name__ == '__main__':
    # save(shuffled=False)
    # save(shuffled=True)
    load(False)
