"""AQI-36 loading, splitting and geographic adjacency.

The dataset is the 36-station Beijing PM2.5 archive released with ST-MVL
(Yi et al., IJCAI 2016) and used by both CSDI and PriSTI, which is what makes
our numbers directly comparable to their published tables.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# CSDI and PriSTI both hold out these months on AQI-36. Keeping the same split
# is what lets us compare against their reported results.
TEST_MONTHS = (3, 6, 9, 12)
VAL_MONTHS = (2, 5, 8, 11)
WINDOW_LEN = 36


def load_ground_truth(path: str) -> pd.DataFrame:
    """Hourly PM2.5 for 36 stations, indexed by timestamp. NaN = genuinely missing."""
    df = pd.read_csv(path, index_col="datetime", parse_dates=True)
    df.columns = [str(c) for c in df.columns]
    return df


def load_stations(path: str) -> pd.DataFrame:
    """Station coordinates, indexed by the same ids used as dataframe columns."""
    df = pd.read_csv(path)
    df["sensor_id"] = df["sensor_id"].astype(str).str.zfill(6)
    return df.set_index("sensor_id")


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = np.radians(lat2 - lat1)
    dl = np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def geographic_adjacency(stations: pd.DataFrame, threshold: float = 0.1) -> np.ndarray:
    """Thresholded Gaussian kernel over inter-station distance, as in PriSTI.

    Returns an (N, N) weight matrix with zeros below `threshold`.
    """
    lat = stations["latitude"].to_numpy()
    lon = stations["longitude"].to_numpy()
    n = len(stations)
    dist = np.zeros((n, n))
    for i in range(n):
        dist[i] = haversine_km(lat[i], lon[i], lat, lon)
    sigma = dist[dist > 0].std()
    adj = np.exp(-((dist / sigma) ** 2))
    adj[adj < threshold] = 0.0
    np.fill_diagonal(adj, 0.0)
    return adj


def split_by_month(df: pd.DataFrame):
    """Split into train / val / test frames using the CSDI-PriSTI month convention."""
    month = df.index.month
    test = df[np.isin(month, TEST_MONTHS)]
    val = df[np.isin(month, VAL_MONTHS)]
    train = df[~np.isin(month, TEST_MONTHS + VAL_MONTHS)]
    return train, val, test


def to_windows(df: pd.DataFrame, window_len: int = WINDOW_LEN,
               stride: int | None = None) -> np.ndarray:
    """Chop a frame into (n_windows, window_len, n_stations) blocks.

    `stride` defaults to `window_len` (non-overlapping). Training uses stride=1
    so that 12 months of hourly data yields a few thousand windows rather than
    a few dozen -- CSDI trains on the sliding view, and with only ~80 windows
    the diffusion model does not converge at all.
    """
    arr = df.to_numpy(dtype=np.float32)
    stride = window_len if stride is None else stride
    starts = range(0, len(arr) - window_len + 1, stride)
    if not starts:
        return np.empty((0, window_len, arr.shape[1]), dtype=np.float32)
    return np.stack([arr[i:i + window_len] for i in starts]).astype(np.float32)


class Standardiser:
    """Per-station mean/std computed on observed training values only."""

    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, arr: np.ndarray) -> "Standardiser":
        flat = arr.reshape(-1, arr.shape[-1])
        self.mean = np.nanmean(flat, axis=0)
        self.std = np.nanstd(flat, axis=0)
        self.std[self.std == 0] = 1.0
        return self

    def transform(self, arr: np.ndarray) -> np.ndarray:
        return (arr - self.mean) / self.std

    def inverse(self, arr: np.ndarray) -> np.ndarray:
        return arr * self.std + self.mean
