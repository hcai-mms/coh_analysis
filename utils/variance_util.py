from utils.tonality_distances import tonality_distances


def get_coherence(sq_var, pl_var, threshold):
    if pl_var < threshold:
        return float('nan')
    return 1.0 - sq_var / pl_var


def jaccard_distance(A, B):
    return 1 - jaccard_similarity(A, B)


def jaccard_similarity(A, B):
    return len(A & B) / len(A | B)


def pair_distance(a, b):
    return abs(b - a)


def tonality_distance(a, b):
    mode_a, key_a = a
    mode_b, key_b = b
    return tonality_distances[12*mode_a+key_a, 12*mode_b+key_b]


def playlist_variance(playlist, threshold, f):
    sum_ = 0
    counter = 0
    for i, (a_feature, a_artists) in enumerate(playlist[:-1]):
        if a_feature is None or (threshold > 0 and len(a_artists) == 0):
            continue
        for j, (b_feature, b_artists) in enumerate(playlist[i + 1:]):
            if b_feature is None or (threshold > 0 and len(b_artists) == 0):
                continue
            if threshold <= 0 or jaccard_distance(a_artists, b_artists) >= threshold:
                sum_ += f(a_feature, b_feature) ** 2
                counter += 1
    if counter > 0:
        return sum_ / counter / 2, counter
    else:
        return float('NaN'), 0


def sequential_variance(playlist, d, threshold, f):
    sum_ = 0
    counter = 0
    for i, (a_feature, a_artists) in enumerate(playlist[:-d]):
        if a_feature is None or (threshold > 0 and len(a_artists) == 0):
            continue
        b_feature, b_artists = playlist[i + d]
        if b_feature is None or (threshold > 0 and len(b_artists) == 0):
            continue
        if threshold <= 0 or jaccard_distance(a_artists, b_artists) >= threshold:
            sum_ += f(a_feature, b_feature) ** 2
            counter += 1
    if counter > 0:
        return sum_ / counter / 2, counter
    else:
        return float('NaN'), 0


def calc_variances(data, distance, threshold, f=pair_distance):
    pid, tracks = data
    s, s_c = sequential_variance(tracks, distance, threshold, f=f)
    p, p_c = playlist_variance(tracks, threshold, f=f)
    return pid, s, p, s_c, p_c, len(tracks)


def main():
    import random
    n = 10000
    #features = [i/n for i in range(n)]
    #features = [0 for i in range(n)] +  [1 for i in range(n)]
    features = [i%2 for i in range(n)]
    #features = [n-i//2-1 if i % 2 else i//2 for i in range(n)]
    #print(features)

    playlist = [(feature, None) for feature in features]
    sq_var, _ = sequential_variance(playlist, 1, 0.0, f=pair_distance)
    pl_var, _ = playlist_variance(playlist, 0.0, f=pair_distance)
    coherence = get_coherence(sq_var, pl_var, threshold=1-5)
    print(coherence)

if __name__ == '__main__':
    main()
