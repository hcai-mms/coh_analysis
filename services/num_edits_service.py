import json as json_reader
import os

from services import CACHED_PATH
from services.service import Service


class NumEditsService(Service):
    def __init__(self, cached_path=CACHED_PATH):
        filepath = os.path.join(cached_path, f'num_edits_service.pk')
        self.num_edits = {}
        super().__init__(filepath)

    def load_from_data(self, directory='data/mpd.v1'):
        for filename in next(os.walk(directory))[2]:
            filepath = os.path.join(directory, filename)
            data = json_reader.load(open(filepath))
            playlist_list = data['playlists']

            for json in playlist_list:
                playlist_id = json['pid']
                num_edits = json['num_edits']
                self.num_edits[playlist_id] = int(num_edits)

    def save(self):
        self._save(self.num_edits)

    def load_from_cache(self):
        self.num_edits = self._load_from_cache()


def save():
    num_edits_service = NumEditsService()
    num_edits_service.load_from_data()
    num_edits_service.save()
    print('Finished')


def load():
    num_edits_service = NumEditsService()
    num_edits_service.load_from_cache()
    print('Finished')


if __name__ == '__main__':
    # save()
    load()
