import csv
import os
import re

from ppo.audio_feature import AudioFeature
from services import CACHED_PATH
from services.service import Service


class FeatureService(Service):
    def __init__(self, cached_path=CACHED_PATH):
        filepath = os.path.join(cached_path, f'feature_service.pk')
        self.features = dict()
        super().__init__(filepath)

    @staticmethod
    def line_to_feature(line):
        uri = line[0]

        danceability = float(line[1])
        energy = float(line[2])
        loudness = float(line[3])
        speechiness = float(line[4])
        acousticness = float(line[5])
        instrumentalness = float(line[6])
        liveness = float(line[7])
        valence = float(line[8])
        tempo = float(line[9])
        mode = int(line[10])
        key = int(line[11])

        simple_features = [danceability, energy, speechiness, acousticness, instrumentalness, liveness, valence]
        for f in simple_features:
            assert 0.0 <= f <= 1.0
        if sum(simple_features) <= 0.0:
            return uri, None

        assert 30 <= tempo <= 250 or tempo == 0.0
        assert 0 == mode or 1 == mode
        assert 0 <= key <= 12

        return uri, AudioFeature(danceability, energy, loudness, speechiness, acousticness, instrumentalness, liveness,
                                 valence, tempo, mode, key)

    def load_from_data(self, directory='data/features'):
        for file in next(os.walk(directory))[2]:
            if not re.match(r'features_\d{3}.csv', file):
                continue
            reader = csv.reader(open(os.path.join(directory, file)), delimiter='\t')
            for line in reader:
                uri, feature = self.line_to_feature(line)
                if feature is not None:
                    self.features[uri] = feature

    def save(self):
        self._save(self.features)

    def load_from_cache(self):
        self.features = self._load_from_cache()

    def __contains__(self, item):
        return item in self.features

    def __getitem__(self, item):
        return self.features[item]


def save():
    feature_service = FeatureService()
    feature_service.load_from_data()
    feature_service.save()
    print('Finished')


def load():
    feature_service = FeatureService()
    feature_service.load_from_cache()
    print('Finished')


if __name__ == '__main__':
    # save()
    load()
