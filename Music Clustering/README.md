## Multimodal Music Clustering

This repository provides a reproducible pipeline for clustering music using
Spotify-style tabular audio features, Bengali lyrics (TF-IDF + SVD), and audio
MFCC features. It includes a modular Python implementation and an exploratory
notebook. The original notebook is not included.

### Project Structure

```
project/
  data/
    audio/               # audio dataset (wav)
    lyrics/              # lyrics datasets (Bangla CSV)
  notebooks/
    exploratory.ipynb    # main notebook entrypoint
  src/
    dataset.py
    clustering.py
    evaluation.py
    vae.py
  results/
```

### Requirements

Install dependencies:

```
pip install -r requirements.txt
```

### Data Setup

By default, the code expects:

- Spotify-style tabular data at `data/lyrics/English Lyrics/data.csv`
- Bengali lyrics CSV under `data/lyrics/` (auto-detects the first CSV)
- Audio wav files under `data/audio/Data/genre_original/`

If your data lives elsewhere, update paths in:

- `notebooks/exploratory.ipynb` (cells that set `SPOTIFY_CSV`, `LYRICS_BN_ROOT`,
  and `AUDIO_ROOT`)
- or edit defaults in `src/dataset.py`

### Run

Open the notebook:

```
jupyter notebook notebooks/exploratory.ipynb
```

Run cells in order to:

- load and explore datasets
- build PCA+KMeans baselines
- train a (Beta-)VAE and cluster its latent space
- cluster lyrics and audio embeddings
- fuse audio+lyrics embeddings and cluster
- save results to `results/`

### Outputs

The notebook saves clustering metrics and cluster assignments to `results/`.

### Notes

- `plotly` is used for interactive t-SNE plots. If you prefer static plots,
  modify `plot_tsne` in `src/clustering.py`.
- Large datasets can be downsampled in the notebook (`n_sample`, `max_audio`,
  `max_lyrics`).
