# The Impact of Playlist Characteristics on Coherence in User-curated Music Playlists

Harald Schweiger: [harald.schweiger@jku.at](mailto:harald.schweiger@jku.at)

Emilia Parada-Cabaleiro: [emiliaparada.cabaleiro@hfm-nuernberg.de](mailto:emiliaparada.cabaleiro@hfm-nuernberg.de)

Markus Schedl: [markus.schedl@jku.at](markus.schedl@jku.at)

## Abstract

Music playlist creation is a crucial, yet not fully explored task in music data mining and music information retrieval.
Previous studies have largely focused on investigating diversity, popularity, and serendipity of tracks in human- or
machine-generated playlists.
However, the concept of playlist coherence -- vaguely defined as smooth transitions between tracks -- remains poorly
understood and even lacks a standardized definition.
In this paper, we provide a formal definition for measuring playlist coherence based on the sequential ordering of
tracks, offering a more interpretable measurement compared to existing literature, and allowing for comparisons between
playlists with different musical styles.
The presented formal framework to measure coherence is applied to analyze a substantial dataset of user-generated
playlists, examining how various playlist characteristics influence coherence. We identified four key attributes:
playlist length, number of edits, track popularity, and collaborative playlist curation as potential influencing
factors. Using correlation and causal inference models, the impact of these attributes on coherence across ten auditory
and one metadata feature are assessed.
The findings reveal that these attributes influence coherence to varying degrees, offering valuable insights for
improving the quality of automatic playlist generation and playlist continuation tasks beyond traditional accuracy
metrics. Additionally, these insights can be applied to develop tools for automatic playlist reordering. By
incorporating playlist coherence, music streaming platforms can enhance user satisfaction and increase retention.

## Requirements

The results have been calculated using Python version **3.11**.  
The required packages can be installed from the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

## Project Structure

This repository provides the code and data required to perform the experiments described in the paper.  
The project follows a single responsibility pattern, with each class fulfilling a specific role.  
Most classes also inherit from the service class, which contains the following persistence methods:

* `load_from_data(...)`: Calculates the data/results for the respective class.  
  This method always requires some input data (e.g., `csv` files or other services) to process the data.

* `load_from_cache()`: Does not take any arguments and loads information from cached data.  
  Cached data is uploaded to the repository for most classes and can be found in the `cached` directory as a single
  pickle (`.pk`) file.  
  Notably, data retrieved through the Spotify API is not included in this directory.

* `save`: Saves the calculated data for the respective class to the cache directory  
  (only useful if data is recalculated using `load_from_data`).

## Important Classes

This project includes a subset of primary classes that are essential for reproducing the results.  
These classes are summarized in the following table:

| Class                               | Description                                                                                                                                                                                                                                                                          |
|-------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `services/coherence_service`        | Contains the coherence values and attributes of all playlists used for the experiments. This class is also responsible for categorizing playlists into control and treatment groups.                                                                                                 |
| `services/causal_inference_service` | Performs causal inference experiments based on the control and treatment groups defined in the `coherence_service`.                                                                                                                                                                  |
| `services/latex_table`              | Uses the `coherence_service` and `causal_inference_service` to generate Table 4: *"Results of Correlation and Causal Inference Analysis."* If `shuffled` is set to `true`, the results will mirror the table in the supplementary information section.                               |
| `algorithm/algorithm_service`       | Contains the code for rearranging playlists. The coherence values targeted during rearrangement are retrieved using the multivariate linear regression model from the `estimator_service`. The `bayesian_service` identifies track combinations that belong together in the dataset. |
| `utils/variance_util`               | Offers the implementation for the population variance, sequential variance and coherence formula.                                                                                                                                                                                    |
| `jupyter/*`                         | A directory containing Jupyter notebooks for reproducing the plots from the paper.                                                                                                                                                                                                   |
| `ppo/*`                             | Contains plain Python objects (PPOs) that serve as container classes for storing data such as playlists and tracks.                                                                                                                                                                  |


## Spotify Data

While most data is cached, allowing the main results to be reproduced, it is also possible to recalculate everything from scratch.  
To do so, the following datasets and resources must be retrieved via Spotify's [Web API](https://developer.spotify.com/documentation/web-api):

- [Million Playlist Dataset](https://www.aicrowd.com/challenges/spotify-million-playlist-dataset-challenge)  
- [Audio Features](https://developer.spotify.com/documentation/web-api/reference/get-audio-features)  
- Additional [Track Information](https://developer.spotify.com/documentation/web-api/reference/get-several-tracks)  

**Note:**  
The results may deviate slightly as Spotify's audio features are continuously updated.  
If you need further assistance, please feel free to contact us.