import csv
import os
import re

from ppo.track_info import TrackInfo
from services import CACHED_PATH
from services.service import Service
from utils.bool_util import str_to_bool


class TrackInfoService(Service):
    def __init__(self, cached_path=CACHED_PATH):
        filepath = os.path.join(cached_path, f'track_info_service.pk')
        self.track_info = dict()
        super().__init__(filepath)

    def load_from_data(self, directory='data/tracks'):
        for file in next(os.walk(directory))[2]:
            if not re.match(r'tracks_\d{3}.csv', file):
                continue
            reader = csv.reader(open(os.path.join(directory, file)), delimiter='\t')
            for line in reader:
                track_id = line[0]
                name = line[1]
                album = line[2]
                duration = int(line[3])

                explicit = str_to_bool(line[4])
                is_local = str_to_bool(line[5])
                popularity = None if line[6] == 'None' else int(line[6])
                self.track_info[track_id] = TrackInfo(track_id, name, album, duration, explicit, is_local, popularity,
                                                      set(line[7:]))

    def save(self):
        self._save(self.track_info)

    def load_from_cache(self):
        self.track_info = self._load_from_cache()

    def __contains__(self, item):
        return item in self.track_info

    def __getitem__(self, item):
        return self.track_info[item]


def save():
    track_info_service = TrackInfoService()
    track_info_service.load_from_data()
    track_info_service.save()
    print('Finished')


def load():
    track_info_service = TrackInfoService()
    track_info_service.load_from_cache()
    print('Finished')


if __name__ == '__main__':
    # save()
    load()
