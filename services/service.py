import os
import pickle as pk


class Service:
    def __init__(self, filepath):
        self.filepath = filepath

    def is_cached(self):
        return os.path.exists(self.filepath)

    def _save(self, data):
        pk.dump(data, open(self.filepath, 'wb'), pk.HIGHEST_PROTOCOL)

    def _load_from_cache(self):
        return pk.load(open(self.filepath, 'rb'))

    def save(self):
        raise NotImplementedError()

    def load_from_cache(self):
        raise NotImplementedError()

    def load_from_data(self, **kwargs):
        raise NotImplementedError()
