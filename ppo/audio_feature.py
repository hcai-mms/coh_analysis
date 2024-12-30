from bidict import bidict


class AudioFeature:
    feature_labels = bidict({l: i for i, l in enumerate(
        ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence',
         'tempo', 'tonality'])})

    def __init__(self, danceability: float, energy: float, loudness: float, speechiness: float,
                 acousticness: float, instrumentalness: float, liveness: float, valence: float,
                 tempo: float, mode: int, key: int):
        self.danceability = danceability
        self.energy = energy
        self.loudness = loudness
        self.speechiness = speechiness
        self.acousticness = acousticness
        self.instrumentalness = instrumentalness
        self.liveness = liveness
        self.valence = valence
        self.tempo = tempo
        self.tonality = mode, key

    def __iter__(self):
        yield self.danceability
        yield self.energy
        yield self.loudness
        yield self.speechiness
        yield self.acousticness
        yield self.instrumentalness
        yield self.liveness
        yield self.valence
        yield self.tempo
        yield self.tonality

    def __getitem__(self, item):
        if type(item) == int:
            item_str = self.feature_labels.inv[item]
        elif type(item) == str:
            item_str = item
        else:
            raise KeyError(item)
        return getattr(self, item_str)

    def __setitem__(self, item, value):
        if type(item) == int:
            item_str = self.feature_labels.inv[item]
        elif type(item) == str:
            item_str = item
        else:
            raise KeyError(item)
        setattr(self, item_str, value)
