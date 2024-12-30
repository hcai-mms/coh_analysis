import numpy as np

from ppo.track import Track
from services.artist_variance_service import embedding_similarity
from services.normalized_feature_service import NormalizedFeatureService
from utils.variance_util import tonality_distance


class Node:
    def __init__(self, name, tracks: list[Track], features: list, artists: list):
        self.name = name
        self.tracks = tracks
        self.features = features
        self.artists = artists

    @staticmethod
    def create_node(name, tracks: list[Track],
                 feature_service: NormalizedFeatureService,
                 matrix, track_to_ids):
        features = [feature_service[t.track_id] for t in tracks]
        artists = [matrix[:, track_to_ids[t.track_id]] for t in tracks]
        return Node(name, tracks, features, artists)


    def __sub__(self, other):
        assert isinstance(other, Node)

        sq_distances = []

        a_feature = self.features[-1]
        b_feature = other.features[0]
        for i, a_f, b_f in zip(range(9), a_feature, b_feature):
            sq_distances.append(abs(a_f-b_f))
        sq_distances.append(tonality_distance(a_feature.tonality, b_feature.tonality))

        artist_a = self.artists[-1]
        artist_b = self.artists[0]
        sq_distances.append(embedding_similarity(artist_a, artist_b))
        return np.array(sq_distances)

    def __str__(self):
        return self.name

