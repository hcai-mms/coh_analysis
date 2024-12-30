import os
import numpy as np

from services import CACHED_PATH
from services.feature_service import FeatureService
from services.service import Service


class NormalizedFeatureService(Service):
    def __init__(self, cached_path=CACHED_PATH):
        filepath = os.path.join(cached_path, f'normalized_feature_service.pk')
        self.features = dict()
        super().__init__(filepath)

    def load_from_data(self, feature_service: FeatureService):
        data = feature_service.features
        tempo = np.array([f['tempo'] for f in data.values()])

        min_loudness = -60
        diff_loudness = 60

        min_tempo = tempo.min()
        diff_tempo = tempo.max() - min_tempo

        for audio_feature in data.values():
            audio_feature['loudness'] = ((max(min(audio_feature['loudness'], 0),
                                              -60) - min_loudness) / diff_loudness) ** 3
            audio_feature['tempo'] = (audio_feature['tempo'] - min_tempo) / diff_tempo

        self.features = data

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
    feature_service.load_from_cache()

    normalized_feature_service = NormalizedFeatureService()
    normalized_feature_service.load_from_data(feature_service)
    normalized_feature_service.save()
    print('Finished')


def load():
    normalized_feature_service = NormalizedFeatureService()
    normalized_feature_service.load_from_cache()
    print('Finished')


if __name__ == '__main__':
    # save()
    load()
