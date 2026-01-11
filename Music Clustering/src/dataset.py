import os
import glob
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, PCA


def _repo_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


DEFAULT_SPOTIFY_CSV = os.path.join(
    _repo_root(), "data", "lyrics", "English Lyrics", "data.csv"
)
DEFAULT_LYRICS_BN_ROOT = os.path.join(_repo_root(), "data", "lyrics")
DEFAULT_AUDIO_ROOT = os.path.join(_repo_root(), "data", "audio", "Data", "genre_original")


def load_spotify_features(csv_path, features, n_sample=None, seed=42):
    df = pd.read_csv(csv_path)
    df_clean = df.dropna(subset=features).copy()
    if n_sample is not None and len(df_clean) > n_sample:
        df_clean = df_clean.sample(n_sample, random_state=seed)
    X = df_clean[features].astype(float).values
    scaler = StandardScaler()
    Xz = scaler.fit_transform(X)
    return df_clean, Xz, scaler


def find_first_csv(root):
    csvs = glob.glob(os.path.join(root, "*.csv")) + glob.glob(
        os.path.join(root, "**", "*.csv"), recursive=True
    )
    return csvs[0] if csvs else None


def load_bengali_lyrics(root, candidates=None):
    if candidates is None:
        candidates = ["lyrics", "lyric", "text", "content", "song_lyrics", "lyric_text"]

    csv_path = find_first_csv(root)
    if csv_path is None:
        raise FileNotFoundError(
            f"No CSV found under {root}. Upload dataset and set root correctly."
        )

    bn = pd.read_csv(csv_path)
    text_col = None
    for c in candidates:
        if c in bn.columns:
            text_col = c
            break
    if text_col is None:
        raise ValueError(
            f"Couldn't find lyrics column. Available columns: {list(bn.columns)[:50]}"
        )

    bn = bn.dropna(subset=[text_col]).copy()
    bn["lyrics_text"] = bn[text_col].astype(str)
    return bn, text_col, csv_path


def lyrics_tfidf_svd(
    bn,
    text_col="lyrics_text",
    max_lyrics=20000,
    svd_dim=128,
    seed=42,
):
    bn_small = bn.sample(min(max_lyrics, len(bn)), random_state=seed).reset_index(drop=True)
    tfidf = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        min_df=3,
        max_features=200000,
    )
    X_text = tfidf.fit_transform(bn_small[text_col])
    svd = TruncatedSVD(n_components=svd_dim, random_state=seed)
    Z_text = svd.fit_transform(X_text)
    scaler = StandardScaler(with_mean=True, with_std=True)
    Z_text_z = scaler.fit_transform(Z_text)
    return bn_small, Z_text_z, tfidf, svd, scaler


def list_wav_files(audio_root):
    return glob.glob(os.path.join(audio_root, "**", "*.wav"), recursive=True)


def mfcc_stats_safe(path, sr=22050, n_mfcc=20, duration=30):
    import librosa

    try:
        y, _sr = librosa.load(path, sr=sr, mono=True, duration=duration)
        if y is None or len(y) < sr * 0.5:
            return None
        mfcc = librosa.feature.mfcc(y=y, sr=_sr, n_mfcc=n_mfcc)
        feat = np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)], axis=0)
        return feat
    except Exception:
        return None


def extract_mfcc_features(
    wav_files,
    max_audio=3000,
    sr=22050,
    n_mfcc=20,
    duration=30,
):
    if len(wav_files) == 0:
        raise FileNotFoundError("No .wav files found under the provided audio root.")

    wav_use = wav_files[:max_audio] if max_audio is not None else wav_files
    X_list = []
    bad_files = []

    for p in wav_use:
        feat = mfcc_stats_safe(p, sr=sr, n_mfcc=n_mfcc, duration=duration)
        if feat is None:
            bad_files.append(p)
        else:
            X_list.append(feat)

    if len(X_list) < 10:
        raise RuntimeError("Too few valid audio files after filtering.")

    Xa = np.vstack(X_list)
    Xa_z = StandardScaler().fit_transform(Xa)
    return Xa_z, bad_files


def fuse_audio_text_embeddings(
    Xa_z,
    Z_text_z,
    fuse_dim_audio=64,
    fuse_dim_text=64,
    seed=42,
):
    pca_audio = PCA(n_components=min(fuse_dim_audio, Xa_z.shape[1]), random_state=seed)
    A_f = pca_audio.fit_transform(Xa_z)

    pca_text = PCA(n_components=min(fuse_dim_text, Z_text_z.shape[1]), random_state=seed)
    T_f = pca_text.fit_transform(Z_text_z)

    n = min(len(A_f), len(T_f))
    rng = np.random.RandomState(seed)
    idxA = rng.permutation(len(A_f))[:n]
    idxT = rng.permutation(len(T_f))[:n]

    F = np.concatenate([A_f[idxA], T_f[idxT]], axis=1)
    Fz = StandardScaler().fit_transform(F)
    return Fz


def load_audio_paths_and_labels(gtzan_dir):
    patterns = [
        os.path.join(gtzan_dir, "genres_original", "*", "*.wav"),
        os.path.join(gtzan_dir, "genres", "*", "*.wav"),
        os.path.join(gtzan_dir, "*", "*.wav"),
    ]
    files = []
    for pat in patterns:
        files.extend(glob.glob(pat))
    files = sorted(list(set(files)))
    if len(files) == 0:
        raise FileNotFoundError("No .wav files found under the provided audio root.")

    labels = [os.path.basename(os.path.dirname(fp)) for fp in files]
    label_to_id = {g: i for i, g in enumerate(sorted(set(labels)))}
    y = np.array([label_to_id[g] for g in labels], dtype=np.int64)
    return files, y, label_to_id


def safe_load_audio(fp, sr=22050, mono=True, duration=30.0):
    import librosa

    try:
        y, sr_out = librosa.load(fp, sr=sr, mono=mono, duration=duration)
        if y is None or len(y) < 1000:
            return None, None
        return y, sr_out
    except Exception:
        return None, None


def audio_to_logmel(y, sr, n_mels=128, n_fft=2048, hop_length=512, fmin=20, fmax=None):
    import librosa

    S = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        fmin=fmin,
        fmax=fmax,
    )
    logS = librosa.power_to_db(S, ref=np.max)
    logS = (logS - logS.mean()) / (logS.std() + 1e-8)
    return logS.astype(np.float32)


def pad_or_crop(spec, target_frames=128):
    n_mels, frames = spec.shape
    if frames == target_frames:
        return spec
    if frames > target_frames:
        return spec[:, :target_frames]
    pad_width = target_frames - frames
    return np.pad(spec, ((0, 0), (0, pad_width)), mode="constant", constant_values=0.0)


def prefilter_readable_files(files, labels, sr=22050, duration=30.0, max_check=None):
    good_files, good_labels = [], []
    bad = []

    n = len(files) if max_check is None else min(len(files), max_check)
    for i in range(n):
        fp = files[i]
        y, _ = safe_load_audio(fp, sr=sr, duration=duration)
        if y is None:
            bad.append(fp)
        else:
            good_files.append(fp)
            good_labels.append(int(labels[i]))

    report = pd.DataFrame(
        {"total_checked": [n], "good": [len(good_files)], "bad": [len(bad)]}
    )
    bad_df = pd.DataFrame({"bad_file": bad})
    return good_files, np.array(good_labels, dtype=np.int64), report, bad_df


class RobustMelSpecDataset:
    def __init__(self, file_paths, labels, sr=22050, seconds=30, n_mels=128, target_frames=128):
        self.file_paths = file_paths
        self.labels = labels
        self.sr = sr
        self.seconds = seconds
        self.n_mels = n_mels
        self.target_frames = target_frames

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        fp = self.file_paths[idx]
        y, sr = safe_load_audio(fp, sr=self.sr, duration=self.seconds)
        if y is None:
            return None
        spec = audio_to_logmel(y, sr, n_mels=self.n_mels)
        spec = pad_or_crop(spec, target_frames=self.target_frames)
        x = np.expand_dims(spec.astype(np.float32), axis=0)
        label = int(self.labels[idx])
        return x, label


def drop_none_collate(batch):
    import torch

    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    xs, ys = zip(*batch)
    return torch.tensor(np.stack(xs, axis=0)), torch.tensor(ys, dtype=torch.long)


def load_bangla_lyrics_texts(lyrics_dir, max_docs=2000):
    texts = []
    txt_files = glob.glob(os.path.join(lyrics_dir, "**", "*.txt"), recursive=True)
    for fp in txt_files[:max_docs]:
        try:
            with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                t = f.read().strip()
            if len(t) > 30:
                texts.append(t)
        except Exception:
            pass

    if len(texts) == 0:
        csv_files = glob.glob(os.path.join(lyrics_dir, "**", "*.csv"), recursive=True)
        for fp in csv_files[:10]:
            try:
                df = pd.read_csv(fp)
                cand_cols = [
                    c for c in df.columns if str(c).lower() in ["lyrics", "lyric", "text", "content"]
                ]
                if len(cand_cols) == 0:
                    obj_cols = [c for c in df.columns if df[c].dtype == "object"]
                    cand_cols = obj_cols[:1]
                if len(cand_cols) > 0:
                    col = cand_cols[0]
                    for t in df[col].dropna().astype(str).tolist():
                        t = t.strip()
                        if len(t) > 30:
                            texts.append(t)
                        if len(texts) >= max_docs:
                            break
            except Exception:
                pass

    if len(texts) == 0:
        raise FileNotFoundError("Could not load lyrics texts. Adjust loader to dataset structure.")
    return texts[:max_docs]


def load_spotify_texts(spotify_path, max_docs=2000):
    df = pd.read_csv(spotify_path)
    if "name" in df.columns and "artists" in df.columns:
        texts = (df["name"].astype(str) + " " + df["artists"].astype(str)).tolist()
    else:
        cols = [c for c in df.columns if c.lower() in ["name", "track_name", "artist", "artists", "artist_name"]]
        if len(cols) >= 2:
            texts = (df[cols[0]].astype(str) + " " + df[cols[1]].astype(str)).tolist()
        else:
            obj_cols = [c for c in df.columns if df[c].dtype == "object"]
            if not obj_cols:
                raise ValueError("Spotify CSV has no usable text columns.")
            texts = df[obj_cols[0]].astype(str).tolist()
    texts = [t.strip() for t in texts if isinstance(t, str) and len(t.strip()) > 10]
    return texts[:max_docs]


def make_text_embeddings(texts, max_features=20000, ngram_range=(1, 2), svd_dim=128, seed=42):
    tfidf = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
    X_tfidf = tfidf.fit_transform(texts)
    svd = TruncatedSVD(n_components=svd_dim, random_state=seed)
    X_txt = svd.fit_transform(X_tfidf)
    X_txt = StandardScaler().fit_transform(X_txt)
    return X_txt, tfidf, svd


def fuse_hybrid_features(audio_latents, lyrics_emb, seed=42):
    rng = np.random.default_rng(seed)
    n = audio_latents.shape[0]
    m = lyrics_emb.shape[0]
    lyrics_idx = rng.integers(0, m, size=n)
    lyrics_for_audio = lyrics_emb[lyrics_idx]
    hybrid = np.concatenate([audio_latents, lyrics_for_audio], axis=1)
    return StandardScaler().fit_transform(hybrid)
