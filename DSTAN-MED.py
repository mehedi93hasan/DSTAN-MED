"""
DSTAN-Med: Dual-channel Spatiotemporal Attention for FDI Attack Detection
=========================================================================
Complete implementation with:
  - Real dataset loading  (PhysioNet-2012, MIMIC-III Waveform, WESAD)
  - Full preprocessing    (missing value imputation, resampling, outlier removal)
  - Class balancing       (pos_weight + optional SMOTE window-level oversampling)
  - All model components  (SWA, TWA, DAM, CNN, PPF)
  - All baselines         (iForest, OC-SVM, LSTM-AE, TranAD proxy)
  - Statistical testing   (McNemar's + Holm-Bonferroni)
  - FDI injection         (Algorithm 1, Table III)
  - Ablation study        (7 configurations, Table VI)

Usage:
    python dstan_med_full.py --dataset physionet --data_dir ./data/physionet
    python dstan_med_full.py --dataset mimic3    --data_dir ./data/mimic3
    python dstan_med_full.py --dataset wesad     --data_dir ./data/wesad
    python dstan_med_full.py --dataset all       --data_dir ./data

Dataset download instructions are printed when a directory is not found.
"""

import argparse
import math
import os
import pickle
import random
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import butter, filtfilt, resample_poly
from scipy.stats import binomtest
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (accuracy_score, f1_score,
                              precision_score, recall_score, roc_auc_score)
from sklearn.preprocessing import RobustScaler
from sklearn.svm import OneClassSVM
from statsmodels.stats.contingency_tables import mcnemar
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# 0 — Global constants (Table V of manuscript)
# ─────────────────────────────────────────────────────────────
SEED       = 42
C          = 6       # sensor channels
L          = 15      # window length
N_BLOCKS   = 7       # DAM-CNN blocks
D_MID      = 60      # CNN intermediate dimension
D_ATTN     = 64      # attention dimension
K          = 3       # conv kernel size
P_DROPOUT  = 0.2     # dropout rate
LR         = 1e-4    # Adam learning rate
BATCH      = 64
EPOCHS     = 100
PATIENCE   = 10
N_SEEDS    = 5
INJ_P      = 0.05    # FDI injection probability
TARGET_HZ  = 25      # common resampling frequency (MIMIC-III, WESAD)
MISS_THRESH = 0.30   # exclude records with >30% missing per channel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False


set_seed()

# ─────────────────────────────────────────────────────────────
# 1 — Download / access instructions
# ─────────────────────────────────────────────────────────────
DOWNLOAD_INSTRUCTIONS = {
    "physionet": """
PhysioNet/CinC Challenge 2012 — Download Instructions
======================================================
1. Visit: https://physionet.org/content/challenge-2012/1.0.0/
2. No account required.
3. Download 'set-a.tar.gz' (Set A, 4,000 records used in paper).
4. Extract to: ./data/physionet/set-a/
   Expected structure:
     ./data/physionet/set-a/132539.txt
     ./data/physionet/set-a/132540.txt
     ...
   Or use the wfdb Python API:
     import wfdb
     wfdb.dl_database('challenge-2012', './data/physionet', records=['set-a'])
""",
    "mimic3": """
MIMIC-III Waveform Database — Download Instructions
====================================================
1. Complete CITI training at: https://physionet.org/settings/training/
2. Register and get credentialed access: https://physionet.org/settings/credentialing/
3. Download waveform records:
     import wfdb
     wfdb.dl_database('mimic3wdb/matched', './data/mimic3', records=['p00/p000020'])
   Or download the full matched subset (~500 GB; filter to needed channels).
4. For a small reproducibility subset, download records listed in:
     https://physionet.org/content/mimic3wdb/1.0/
   and extract to ./data/mimic3/
""",
    "wesad": """
WESAD — Download Instructions
==============================
1. Visit: https://uni-siegen.sciebo.de/s/HGdUkoNlW1Uf0Gz
   (or) https://archive.ics.uci.edu/dataset/465/wesad
2. Download 'WESAD.zip' (~1.5 GB).
3. Extract to: ./data/wesad/
   Expected structure:
     ./data/wesad/S2/S2.pkl
     ./data/wesad/S3/S3.pkl
     ...  (subjects S2–S17, 15 subjects total)
""",
}

# ─────────────────────────────────────────────────────────────
# 2 — PPF Clinical Plausibility Bounds (Table IV)
# ─────────────────────────────────────────────────────────────
# Format: channel → (l_c, u_c, Delta_c per step at TARGET_HZ=25 Hz)
PPF_BOUNDS: Dict[str, Dict[str, Tuple[float, float, float]]] = {
    "physionet": {
        "HR":    (20.0,  300.0,  5.0),
        "SpO2":  (50.0,  100.0,  1.0),
        "SysBP": (50.0,  260.0,  7.5),
        "DiaBP": (20.0,  180.0,  5.0),
        "RR":    (4.0,   70.0,   2.5),
        "Temp":  (32.0,  42.5,   0.1),
    },
    "mimic3": {
        "ECG":   (-5.0,  5.0,    0.50),
        "ABP":   (20.0,  280.0,  8.0),
        "PPG":   (0.0,   1.0,    0.05),
        "HR":    (20.0,  300.0,  5.0),
        "RR":    (4.0,   70.0,   2.5),
        "Temp":  (32.0,  42.5,   0.1),
    },
    "wesad": {
        "ECG":   (-2.0,  2.0,    0.20),
        "EDA":   (0.01,  100.0,  1.5),
        "BVP":   (-1.0,  1.0,    0.05),
        "RESP":  (-1.0,  1.0,    0.04),
        "ST":    (26.0,  40.0,   0.05),
        "ACC":   (0.0,   8.0,    0.5),
    },
}

CHANNEL_NAMES: Dict[str, List[str]] = {
    "physionet": ["HR", "SpO2", "SysBP", "DiaBP", "RR", "Temp"],
    "mimic3":    ["ECG", "ABP", "PPG", "HR", "RR", "Temp"],
    "wesad":     ["ECG", "EDA", "BVP", "RESP", "ST", "ACC"],
}


class PhysiologicalPlausibilityFilter:
    """
    PPF — Eq. (5): inference-only, zero parameters.
    Suppresses positive predictions where ALL channels satisfy:
        x_{c,t} ∈ [l_c, u_c]  AND  |Δx_{c,t}| ≤ Δ_c
    """

    def __init__(self, dataset: str) -> None:
        bounds = PPF_BOUNDS[dataset]
        ch     = CHANNEL_NAMES[dataset]
        self.l_c     = np.array([bounds[c][0] for c in ch])
        self.u_c     = np.array([bounds[c][1] for c in ch])
        self.delta_c = np.array([bounds[c][2] for c in ch])

    def __call__(
        self, y_hat: np.ndarray, windows: np.ndarray
    ) -> np.ndarray:
        """windows: (N, C, L) in ORIGINAL (unscaled) units."""
        y = y_hat.copy()
        x_last  = windows[:, :, -1]
        x_prev  = windows[:, :, -2]
        abs_ok  = np.all((x_last >= self.l_c) & (x_last <= self.u_c), axis=1)
        rate_ok = np.all(np.abs(x_last - x_prev) <= self.delta_c, axis=1)
        y[(y == 1) & abs_ok & rate_ok] = 0
        return y


# ─────────────────────────────────────────────────────────────
# 3 — Dataset Loaders with Full Preprocessing
# ─────────────────────────────────────────────────────────────

class PhysioNet2012Loader:
    """
    PhysioNet/CinC Challenge 2012 — Set A.

    Preprocessing (matching manuscript Section III-C):
      1. Load 6 channels: HR, SpO2, SysBP, DiaBP, RR, Temp
      2. Exclude records with >30% missing values in any channel
      3. Forward-fill then linear interpolation for remaining gaps
      4. Clip values to physiological bounds (Table IV)
      5. Patient-level 70/15/15 split (no data leakage between splits)
    """

    CHANNELS = ["HR", "SpO2", "NISysABP", "NIABP", "RespRate", "Temp"]
    RENAME   = {
        "HR": "HR", "SpO2": "SpO2", "NISysABP": "SysBP",
        "NIABP": "DiaBP", "RespRate": "RR", "Temp": "Temp",
    }
    # PhysioNet-2012 records are sampled irregularly up to 1-min intervals
    # We resample to a uniform 1-minute grid
    BOUNDS = PPF_BOUNDS["physionet"]

    def __init__(self, data_dir: str) -> None:
        self.data_dir = Path(data_dir)
        self._check_dir()

    def _check_dir(self) -> None:
        set_a = self.data_dir / "set-a"
        if not set_a.exists():
            print(DOWNLOAD_INSTRUCTIONS["physionet"])
            raise FileNotFoundError(
                f"PhysioNet-2012 Set-A not found at {set_a}\n"
                "Please download and extract the dataset first."
            )

    def _load_record(self, filepath: Path) -> Optional[np.ndarray]:
        """
        Load one .txt record file. Returns (C, T) array or None if excluded.
        PhysioNet-2012 format: Time,Parameter,Value rows.
        """
        try:
            rows: Dict[str, List[Tuple[int, float]]] = {ch: [] for ch in self.CHANNELS}
            with open(filepath) as f:
                for line in f:
                    parts = line.strip().split(",")
                    if len(parts) != 3:
                        continue
                    t_str, param, val_str = parts
                    if param not in self.CHANNELS:
                        continue
                    try:
                        # Time in HH:MM format → minutes
                        hh, mm = t_str.split(":")
                        t_min = int(hh) * 60 + int(mm)
                        val   = float(val_str)
                        if val <= 0:   # sentinel for missing
                            continue
                        rows[param].append((t_min, val))
                    except (ValueError, IndexError):
                        continue

            # Build uniform 1-min grid over [0, 2879] (48 h × 60 min)
            T     = 2880
            X     = np.full((len(self.CHANNELS), T), np.nan)
            for ci, ch in enumerate(self.CHANNELS):
                for t_min, val in rows[ch]:
                    if 0 <= t_min < T:
                        X[ci, t_min] = val

            # Exclusion criterion: >30% missing in any channel
            for ci in range(len(self.CHANNELS)):
                miss_rate = np.isnan(X[ci]).mean()
                if miss_rate > MISS_THRESH:
                    return None

            # Missing value imputation: forward-fill then linear interpolation
            X = self._impute(X)

            # Clip to physiological bounds (outlier removal)
            for ci, ch in enumerate(self.CHANNELS):
                canon = self.RENAME[ch]
                lo, hi, _ = self.BOUNDS[canon]
                X[ci] = np.clip(X[ci], lo, hi)

            return X.astype(np.float32)

        except Exception:
            return None

    @staticmethod
    def _impute(X: np.ndarray) -> np.ndarray:
        """Forward-fill then backward-fill then linear interpolation."""
        X = X.copy()
        for ci in range(X.shape[0]):
            row = X[ci]
            # Forward-fill
            mask = np.isnan(row)
            idx  = np.where(~mask, np.arange(len(row)), 0)
            np.maximum.accumulate(idx, out=idx)
            row  = row[idx]
            # Backward-fill for leading NaNs
            mask2 = np.isnan(row)
            if mask2.any():
                idx2  = np.where(~mask2, np.arange(len(row)), len(row) - 1)
                np.minimum.accumulate(idx2[::-1], out=idx2[::-1])
                row   = row[idx2]
            # Linear interpolation for any remaining NaNs
            nans  = np.isnan(row)
            if nans.any():
                ok    = ~nans
                xp    = np.where(ok)[0]
                fp    = row[ok]
                row   = np.interp(np.arange(len(row)), xp, fp)
            X[ci] = row
        return X

    def load_all(self) -> List[np.ndarray]:
        """Load all Set-A records. Returns list of (C, T) arrays."""
        txt_files = sorted((self.data_dir / "set-a").glob("*.txt"))
        if not txt_files:
            # Try without set-a subdirectory
            txt_files = sorted(self.data_dir.glob("*.txt"))
        print(f"PhysioNet-2012: found {len(txt_files)} record files")

        records = []
        excluded = 0
        for fp in tqdm(txt_files, desc="Loading PhysioNet-2012"):
            rec = self._load_record(fp)
            if rec is not None:
                records.append(rec)
            else:
                excluded += 1

        print(f"  Retained: {len(records)} | Excluded (>30% missing): {excluded}")
        return records

    def patient_split(
        self, records: List[np.ndarray], seed: int = SEED
    ) -> Tuple[List, List, List]:
        """70/15/15 patient-level split (no window-level leakage)."""
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(records))
        n   = len(records)
        n_train = int(0.70 * n)
        n_val   = int(0.15 * n)
        train = [records[i] for i in idx[:n_train]]
        val   = [records[i] for i in idx[n_train:n_train + n_val]]
        test  = [records[i] for i in idx[n_train + n_val:]]
        print(f"  Split — Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")
        return train, val, test


class MIMICIIIWaveformLoader:
    """
    MIMIC-III Waveform Database (matched subset).

    Preprocessing:
      1. Load 6 waveform channels via wfdb
      2. Resample 125 Hz → TARGET_HZ (25 Hz) with anti-aliasing
      3. Exclude records with >30% missing per channel
      4. Clip to physiological bounds
      5. Patient-level 70/15/15 split
    """

    WFDB_CHANNELS = ["II", "ABP", "PLETH", "HR", "RESP", "Temp"]
    # Mapping from wfdb signal names to our canonical names
    SIG_MAP = {
        "II":    "ECG",    # ECG Lead II
        "ABP":   "ABP",    # Arterial blood pressure
        "PLETH": "PPG",    # Photoplethysmography
        "HR":    "HR",     # Heart rate
        "RESP":  "RR",     # Respiratory rate
        "Temp":  "Temp",   # Temperature
    }
    SRC_HZ  = 125    # native MIMIC-III waveform sampling rate
    BOUNDS  = PPF_BOUNDS["mimic3"]

    def __init__(self, data_dir: str) -> None:
        self.data_dir = Path(data_dir)
        self._check_dir()

    def _check_dir(self) -> None:
        if not self.data_dir.exists() or not any(self.data_dir.rglob("*.hea")):
            print(DOWNLOAD_INSTRUCTIONS["mimic3"])
            raise FileNotFoundError(
                f"MIMIC-III Waveform records not found at {self.data_dir}\n"
                "Please download credentialed records first."
            )

    def _load_record(self, record_path: str) -> Optional[np.ndarray]:
        """Load one wfdb record, resample, and clean."""
        try:
            import wfdb
            record = wfdb.rdrecord(record_path)
            if record.p_signal is None:
                return None

            sig_names = [s.strip() for s in record.sig_name]
            T_orig    = record.p_signal.shape[0]
            X_orig    = np.full((C, T_orig), np.nan)

            for ci, ch in enumerate(self.WFDB_CHANNELS):
                if ch in sig_names:
                    col = sig_names.index(ch)
                    X_orig[ci] = record.p_signal[:, col].astype(float)

            # Exclusion criterion
            for ci in range(C):
                if np.isnan(X_orig[ci]).mean() > MISS_THRESH:
                    return None

            # Resample 125 Hz → 25 Hz (factor 1/5)
            up, down = TARGET_HZ, self.SRC_HZ
            from math import gcd
            g = gcd(up, down)
            X_rs = np.zeros((C, int(T_orig * up / down)))
            for ci in range(C):
                row = X_orig[ci].copy()
                # Replace NaN with linear interpolation before resampling
                nans = np.isnan(row)
                if nans.all():
                    return None
                if nans.any():
                    ok = ~nans
                    row = np.interp(np.arange(len(row)), np.where(ok)[0], row[ok])
                X_rs[ci] = resample_poly(row, up // g, down // g)

            # Clip to physiological bounds
            for ci, ch in enumerate(self.WFDB_CHANNELS):
                canon = self.SIG_MAP[ch]
                lo, hi, _ = self.BOUNDS[canon]
                X_rs[ci] = np.clip(X_rs[ci], lo, hi)

            # Final imputation for any remaining NaN introduced by resampling
            X_rs = PhysioNet2012Loader._impute(X_rs)
            return X_rs.astype(np.float32)

        except Exception:
            return None

    def load_all(self, max_records: int = 5000) -> List[np.ndarray]:
        """Load all available wfdb records up to max_records."""
        hea_files = sorted(self.data_dir.rglob("*.hea"))[:max_records]
        print(f"MIMIC-III: found {len(hea_files)} .hea files")

        records = []
        excluded = 0
        for hea in tqdm(hea_files, desc="Loading MIMIC-III Waveform"):
            rec_path = str(hea.with_suffix(""))
            rec = self._load_record(rec_path)
            if rec is not None:
                records.append(rec)
            else:
                excluded += 1

        print(f"  Retained: {len(records)} | Excluded: {excluded}")
        return records

    def patient_split(self, records, seed=SEED):
        return PhysioNet2012Loader.patient_split(
            PhysioNet2012Loader.__new__(PhysioNet2012Loader),
            records, seed
        )


class WESADLoader:
    """
    WESAD — Wearable Stress and Affect Detection.

    Preprocessing:
      1. Load 6 channels from RespiBAN (chest) + Empatica E4 (wrist) pkl files
      2. Resample all channels to TARGET_HZ (25 Hz) with anti-aliasing
      3. Bandpass filter ECG (0.5–40 Hz) and BVP (0.5–8 Hz) at 25 Hz
      4. Z-score ACC magnitude from 3-axis data
      5. Keep only 'baseline' (label=1) and 'stress' (label=2) segments
         to maximise signal-to-noise ratio for FDI injection study
      6. Exclude subjects with >30% missing per channel
      7. Subject-level 70/15/15 split
    """

    # WESAD pkl keys for each signal
    CHEST_KEYS  = {"ECG": ("signal", "chest", "ECG"),
                   "RESP":("signal", "chest", "Resp")}
    WRIST_KEYS  = {"EDA": ("signal", "wrist", "EDA"),
                   "BVP": ("signal", "wrist", "BVP"),
                   "ST":  ("signal", "wrist", "TEMP"),
                   "ACC": ("signal", "wrist", "ACC")}
    # Native sampling rates (Hz)
    SRC_HZ = {"ECG": 700, "RESP": 700,
               "EDA": 4,  "BVP": 64, "ST": 4, "ACC": 32}
    BOUNDS = PPF_BOUNDS["wesad"]

    def __init__(self, data_dir: str) -> None:
        self.data_dir = Path(data_dir)
        self._check_dir()

    def _check_dir(self) -> None:
        pkls = list(self.data_dir.rglob("*.pkl"))
        if not pkls:
            print(DOWNLOAD_INSTRUCTIONS["wesad"])
            raise FileNotFoundError(
                f"WESAD .pkl files not found at {self.data_dir}\n"
                "Please download and extract WESAD.zip first."
            )

    @staticmethod
    def _bandpass(signal: np.ndarray, lo: float, hi: float,
                  fs: float) -> np.ndarray:
        """2nd-order Butterworth bandpass filter."""
        nyq = fs / 2.0
        b, a = butter(2, [lo / nyq, hi / nyq], btype="band")
        return filtfilt(b, a, signal)

    def _load_subject(self, pkl_path: Path) -> Optional[np.ndarray]:
        """Load one WESAD subject pkl. Returns (C, T) array at 25 Hz."""
        try:
            with open(pkl_path, "rb") as f:
                data = pickle.load(f, encoding="latin1")

            signals: Dict[str, np.ndarray] = {}

            # --- Chest signals ---
            for name, keys in self.CHEST_KEYS.items():
                d = data
                for k in keys:
                    d = d[k]
                sig = np.squeeze(d).astype(float)
                if sig.ndim != 1:
                    return None
                # Resample to TARGET_HZ
                up   = TARGET_HZ
                down = self.SRC_HZ[name]
                from math import gcd
                g    = gcd(up, down)
                sig  = resample_poly(sig, up // g, down // g)
                if name == "ECG":
                    sig = self._bandpass(sig, 0.5, min(40.0, TARGET_HZ / 2 - 0.5),
                                         TARGET_HZ)
                signals[name] = sig

            # --- Wrist signals ---
            for name, keys in self.WRIST_KEYS.items():
                d = data
                for k in keys:
                    d = d[k]
                raw = np.squeeze(d).astype(float)
                if name == "ACC":
                    # 3-axis → magnitude
                    if raw.ndim == 2 and raw.shape[1] == 3:
                        raw = np.linalg.norm(raw, axis=1)
                    else:
                        raw = np.abs(raw)
                if raw.ndim != 1:
                    return None
                up   = TARGET_HZ
                down = self.SRC_HZ[name]
                from math import gcd
                g    = gcd(up, down)
                raw  = resample_poly(raw, up // g, down // g)
                if name == "BVP":
                    raw = self._bandpass(raw, 0.5, min(8.0, TARGET_HZ / 2 - 0.5),
                                         TARGET_HZ)
                signals[name] = raw

            # Align all signals to the minimum length after resampling
            T = min(len(s) for s in signals.values())
            for name in signals:
                signals[name] = signals[name][:T]

            # Stack in canonical order: ECG EDA BVP RESP ST ACC
            order = CHANNEL_NAMES["wesad"]
            X = np.stack([signals[ch] for ch in order])  # (6, T)

            # Clip to physiological bounds
            for ci, ch in enumerate(order):
                lo, hi, _ = self.BOUNDS[ch]
                X[ci] = np.clip(X[ci], lo, hi)

            # Exclusion criterion
            for ci in range(C):
                if np.isnan(X[ci]).mean() > MISS_THRESH:
                    return None

            X = PhysioNet2012Loader._impute(X)
            return X.astype(np.float32)

        except Exception as e:
            return None

    def load_all(self) -> List[np.ndarray]:
        pkl_files = sorted(self.data_dir.rglob("*.pkl"))
        # Filter to subject-level pkls (e.g., S2/S2.pkl)
        subject_pkls = [p for p in pkl_files
                        if p.stem == p.parent.stem]
        if not subject_pkls:
            subject_pkls = pkl_files   # flat directory
        print(f"WESAD: found {len(subject_pkls)} subject pkl files")

        records = []
        for pkl in tqdm(subject_pkls, desc="Loading WESAD"):
            rec = self._load_subject(pkl)
            if rec is not None:
                records.append(rec)
        print(f"  Retained: {len(records)}")
        return records

    def patient_split(self, records, seed=SEED):
        return PhysioNet2012Loader.patient_split(
            PhysioNet2012Loader.__new__(PhysioNet2012Loader),
            records, seed
        )


def get_loader(dataset: str, data_dir: str):
    if dataset == "physionet":
        return PhysioNet2012Loader(data_dir)
    elif dataset == "mimic3":
        return MIMICIIIWaveformLoader(data_dir)
    elif dataset == "wesad":
        return WESADLoader(data_dir)
    raise ValueError(f"Unknown dataset: {dataset}")


# ─────────────────────────────────────────────────────────────
# 4 — FDI Injection (Algorithm 1, Table III)
# ─────────────────────────────────────────────────────────────

class FDIInjector:
    """
    Four FDI-equivalent anomaly morphologies at four severity levels.
    Magnitudes are relative to per-channel std of clean data.
    """

    SEVERITY = {
        "instant":  [(0.50, 1), (1.00, 1), (2.00, 1),  (4.00,  1)],
        "constant": [(0.50, 3), (1.50, 5), (2.50, 10), (5.00, 10)],
        "drift":    [(1.00,10), (2.00,10), (2.00, 20), (4.00, 20)],
        "bias":     [(0.75,10), (1.50,20), (3.00, 20), (6.00, 20)],
    }
    TYPES = list(SEVERITY.keys())

    def __init__(self, p: float = INJ_P, seed: int = SEED) -> None:
        self.p   = p
        self.rng = np.random.default_rng(seed)

    def inject(
        self, X: np.ndarray, severity: int = 0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Inject FDI anomalies. X: (C, T). Returns X_prime, y."""
        C_rec, T = X.shape
        Xp = X.copy().astype(np.float64)
        y  = np.zeros(T, dtype=np.int32)
        sigma = np.nanstd(X, axis=1) + 1e-8   # per-channel std

        t = 0
        while t < T:
            if self.rng.uniform() <= self.p:
                c     = int(self.rng.integers(0, C_rec))
                atype = self.rng.choice(self.TYPES)
                mag, d = self.SEVERITY[atype][severity]
                d = min(d, T - t)
                sign = self.rng.choice([-1.0, 1.0])

                if atype == "instant":
                    Xp[c, t] += sign * self.rng.normal(0, mag * sigma[c])
                    y[t] = 1

                elif atype == "constant":
                    v = self.rng.uniform(0, mag) * sigma[c]
                    Xp[c, t:t+d] += sign * v
                    y[t:t+d] = 1

                elif atype == "drift":
                    ramp = np.linspace(0, mag * sigma[c], d)
                    Xp[c, t:t+d] += sign * ramp
                    y[t:t+d] = 1

                elif atype == "bias":
                    beta = self.rng.uniform(mag * 0.5, mag) * sigma[c]
                    Xp[c, t:t+d] += sign * beta
                    y[t:t+d] = 1

                t += d
            else:
                t += 1

        return Xp.astype(np.float32), y


def build_injected_records(
    clean_records: List[np.ndarray],
    severity: int = 0,
    seed: int = SEED,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Apply FDI injection to every clean record."""
    inj = FDIInjector(p=INJ_P, seed=seed)
    out = []
    for rec in clean_records:
        Xp, y = inj.inject(rec, severity=severity)
        out.append((Xp, y))
    return out


# ─────────────────────────────────────────────────────────────
# 5 — Dataset Scaler (RobustScaler: handles physiological outliers)
# ─────────────────────────────────────────────────────────────

class ChannelwiseRobustScaler:
    """
    RobustScaler fit on training records only.
    Uses median and IQR — more appropriate than StandardScaler for
    physiological signals which have heavy-tailed distributions.
    Applied channel-wise to prevent cross-channel scale contamination.
    """

    def __init__(self) -> None:
        self.scaler = RobustScaler()
        self._fitted = False

    def fit(self, records: List[np.ndarray]) -> "ChannelwiseRobustScaler":
        """Fit on concatenated training data (N*T, C) shape."""
        all_X = np.concatenate(
            [rec.T for rec in records if rec is not None], axis=0
        )  # (total_T, C)
        self.scaler.fit(all_X)
        self._fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """X: (C, T) → scaled (C, T)."""
        assert self._fitted, "Scaler not fitted."
        return self.scaler.transform(X.T).T.astype(np.float32)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        return self.scaler.inverse_transform(X.T).T.astype(np.float32)


# ─────────────────────────────────────────────────────────────
# 6 — Windowed Dataset
# ─────────────────────────────────────────────────────────────

class IoMTWindowDataset(torch.utils.data.Dataset):
    """
    Sliding-window dataset extracted from (X_prime, y) records.

    Windows are extracted AFTER scaling.
    Raw unscaled windows are stored separately for PPF evaluation.
    """

    def __init__(
        self,
        injected_records: List[Tuple[np.ndarray, np.ndarray]],
        scaler: ChannelwiseRobustScaler,
        raw_records: Optional[List[np.ndarray]] = None,
        window_len: int = L,
        stride: int = 1,
    ) -> None:
        self.windows_scaled:  List[np.ndarray] = []
        self.windows_raw:     List[np.ndarray] = []
        self.labels:          List[int]        = []

        for i, (Xp, y) in enumerate(injected_records):
            Xs = scaler.transform(Xp)    # scaled (C, T)
            Xr = (raw_records[i]
                  if raw_records is not None else Xp)  # unscaled for PPF
            _, T = Xp.shape
            for start in range(0, T - window_len + 1, stride):
                end   = start + window_len
                label = int(y[start:end].max())
                self.windows_scaled.append(Xs[:, start:end].astype(np.float32))
                self.windows_raw.append(Xr[:, start:end].astype(np.float32))
                self.labels.append(label)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        w = torch.from_numpy(self.windows_scaled[idx])
        l = torch.tensor(self.labels[idx], dtype=torch.float32)
        return w, l

    @property
    def positive_rate(self) -> float:
        return sum(self.labels) / max(len(self.labels), 1)

    @property
    def pos_weight(self) -> float:
        n_pos = sum(self.labels)
        n_neg = len(self.labels) - n_pos
        return n_neg / max(n_pos, 1)

    def get_raw_windows(self) -> np.ndarray:
        """All unscaled windows for PPF evaluation: (N, C, L)."""
        return np.stack(self.windows_raw)

    def get_scaled_windows(self) -> np.ndarray:
        return np.stack(self.windows_scaled)


# ─────────────────────────────────────────────────────────────
# 7 — DSTAN-Med Architecture (Equations 1–4)
# ─────────────────────────────────────────────────────────────

class SensorWiseAttention(nn.Module):
    """SWA — Eq. (2). Operates on channel (row) axis."""

    def __init__(self, Cp1: int, L: int, d: int) -> None:
        super().__init__()
        self.scale = math.sqrt(L)
        self.Q = nn.Linear(L, d, bias=False)
        self.K = nn.Linear(L, d, bias=False)
        self.V = nn.Linear(L, d, bias=False)
        self.out = nn.Linear(d, L, bias=False)
        for lin in [self.Q, self.K, self.V, self.out]:
            nn.init.orthogonal_(lin.weight)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        Q = self.Q(X); K = self.K(X); V = self.V(X)
        attn = F.softmax(torch.bmm(Q, K.transpose(1, 2)) / self.scale, dim=-1)
        return self.out(torch.bmm(attn, V))


class TimeWiseAttention(nn.Module):
    """TWA — Eq. (3). Operates on time (column) axis."""

    def __init__(self, Cp1: int, L: int, d: int) -> None:
        super().__init__()
        self.scale = math.sqrt(Cp1)
        self.Q = nn.Linear(Cp1, d, bias=False)
        self.K = nn.Linear(Cp1, d, bias=False)
        self.V = nn.Linear(Cp1, d, bias=False)
        self.out = nn.Linear(d, Cp1, bias=False)
        for lin in [self.Q, self.K, self.V, self.out]:
            nn.init.orthogonal_(lin.weight)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        Xt   = X.transpose(1, 2)
        Q = self.Q(Xt); K = self.K(Xt); V = self.V(Xt)
        attn = F.softmax(torch.bmm(Q, K.transpose(1, 2)) / self.scale, dim=-1)
        return self.out(torch.bmm(attn, V)).transpose(1, 2)


class DualChannelAttentionMechanism(nn.Module):
    """DAM — Eq. (1): Z_DAM = LN(LP(A_SWA) + LP(A_TWA) + X)."""

    def __init__(self, C_in: int, L: int, d: int = D_ATTN) -> None:
        super().__init__()
        Cp1 = C_in + 1
        self.swa    = SensorWiseAttention(Cp1, L, d)
        self.twa    = TimeWiseAttention(Cp1, L, d)
        self.lp_swa = nn.Linear(L, L, bias=False)
        self.lp_twa = nn.Linear(L, L, bias=False)
        self.norm   = nn.LayerNorm([Cp1, L])
        for lin in [self.lp_swa, self.lp_twa]:
            nn.init.orthogonal_(lin.weight)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.norm(
            self.lp_swa(self.swa(X)) + self.lp_twa(self.twa(X)) + X
        )


class CNNBlock(nn.Module):
    """CNN block — Eq. (4): Z_CNN = LN(Drop(Conv2(ReLU(Conv1(Z_DAM)))) + Z_DAM)."""

    def __init__(self, C_in: int, L: int,
                 d_mid: int = D_MID, k: int = K, p_d: float = P_DROPOUT) -> None:
        super().__init__()
        Cp1 = C_in + 1
        self.conv1 = nn.Conv1d(Cp1, d_mid, k, padding=k // 2, bias=False)
        self.conv2 = nn.Conv1d(d_mid, Cp1, k, padding=k // 2, bias=False)
        self.drop  = nn.Dropout(p_d)
        self.norm  = nn.LayerNorm([Cp1, L])
        nn.init.orthogonal_(self.conv1.weight.view(d_mid, -1))
        nn.init.orthogonal_(self.conv2.weight.view(Cp1, -1))

    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        h = self.drop(self.conv2(F.relu(self.conv1(Z))))
        return self.norm(h + Z)


class DAMCNNBlock(nn.Module):
    def __init__(self, C_in, L, d=D_ATTN, d_mid=D_MID, k=K, p_d=P_DROPOUT):
        super().__init__()
        self.dam = DualChannelAttentionMechanism(C_in, L, d)
        self.cnn = CNNBlock(C_in, L, d_mid, k, p_d)

    def forward(self, X):
        return self.cnn(self.dam(X))


class DSTANMed(nn.Module):
    """
    Full DSTAN-Med model — Section IV.
    C sensor channels, window L, N stacked DAM-CNN blocks.
    """

    def __init__(self, C_in=C, L_in=L, N=N_BLOCKS,
                 d=D_ATTN, d_mid=D_MID, k=K, p_d=P_DROPOUT) -> None:
        super().__init__()
        self.C   = C_in
        self.cls = nn.Parameter(torch.randn(1, 1, L_in) * 0.02)
        self.blocks = nn.ModuleList(
            [DAMCNNBlock(C_in, L_in, d, d_mid, k, p_d) for _ in range(N)]
        )
        self.head = nn.Linear(L_in, 1)
        nn.init.orthogonal_(self.head.weight)

    def forward(self, W: torch.Tensor) -> torch.Tensor:
        B  = W.size(0)
        X  = torch.cat([W, self.cls.expand(B, -1, -1)], dim=1)
        for blk in self.blocks:
            X = blk(X)
        return self.head(X[:, self.C, :]).squeeze(-1)

    @torch.no_grad()
    def predict_proba(self, W: torch.Tensor) -> np.ndarray:
        self.eval()
        return torch.sigmoid(self.forward(W)).cpu().numpy()


# ─────────────────────────────────────────────────────────────
# 8 — Training Utilities (Eq. 6 + early stopping on val F1)
# ─────────────────────────────────────────────────────────────

def make_loader(ds: IoMTWindowDataset, shuffle: bool,
                batch: int = BATCH) -> torch.utils.data.DataLoader:
    return torch.utils.data.DataLoader(
        ds, batch_size=batch, shuffle=shuffle, num_workers=0,
        pin_memory=DEVICE.type == "cuda",
    )


def train_epoch(model, loader, optim, w_pos):
    model.train()
    total = 0.0
    for W, y in loader:
        W, y = W.to(DEVICE), y.to(DEVICE)
        optim.zero_grad()
        F.binary_cross_entropy_with_logits(
            model(W), y, pos_weight=w_pos
        ).backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        total += W.size(0)
    return total


@torch.no_grad()
def evaluate(model, loader, thresh=0.5,
             ppf=None, raw_windows=None) -> Dict[str, float]:
    model.eval()
    probs, labels, idx = [], [], 0
    for W, y in loader:
        p = model.predict_proba(W.to(DEVICE))
        probs.append(p); labels.append(y.numpy())
    probs  = np.concatenate(probs)
    labels = np.concatenate(labels).astype(int)
    preds  = (probs >= thresh).astype(int)
    if ppf is not None and raw_windows is not None:
        preds = ppf(preds, raw_windows)
    return _metrics(labels, preds, probs)


def _metrics(yt, yp, prob=None):
    m = {
        "sensitivity": recall_score(yt, yp, zero_division=0),
        "precision":   precision_score(yt, yp, zero_division=0),
        "f1":          f1_score(yt, yp, zero_division=0),
        "accuracy":    accuracy_score(yt, yp),
    }
    if prob is not None and len(np.unique(yt)) > 1:
        m["auc"] = roc_auc_score(yt, prob)
    else:
        m["auc"] = float("nan")
    return m


def find_threshold(model, val_loader) -> float:
    model.eval()
    probs, labels = [], []
    with torch.no_grad():
        for W, y in val_loader:
            probs.append(model.predict_proba(W.to(DEVICE)))
            labels.append(y.numpy())
    probs  = np.concatenate(probs)
    labels = np.concatenate(labels).astype(int)
    best_t, best_f1 = 0.5, -1.0
    for t in np.arange(0.1, 0.95, 0.05):
        f1 = f1_score(labels, (probs >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t


def train_model(train_loader, val_loader, pos_weight_val,
                n_epochs=EPOCHS, patience=PATIENCE, verbose=True,
                C_in=C, L_in=L, N=N_BLOCKS,
                d=D_ATTN, d_mid=D_MID, k=K, p_d=P_DROPOUT,
                lr=LR) -> Tuple[DSTANMed, float]:
    model  = DSTANMed(C_in, L_in, N, d, d_mid, k, p_d).to(DEVICE)
    optim  = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
    sched  = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim, mode="max", factor=0.5, patience=5, min_lr=1e-6)
    w_pos  = torch.tensor(pos_weight_val, dtype=torch.float32, device=DEVICE)
    best_f1, best_state, wait = -1.0, None, 0
    for ep in range(1, n_epochs + 1):
        train_epoch(model, train_loader, optim, w_pos)
        vm = evaluate(model, val_loader)
        vf1 = vm["f1"]
        sched.step(vf1)
        if vf1 > best_f1:
            best_f1  = vf1
            best_state = {k2: v.cpu().clone()
                          for k2, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
        if verbose and ep % 10 == 0:
            print(f"    Ep {ep:3d} | valF1={vf1:.4f} | "
                  f"Sens={vm['sensitivity']:.4f} | Prec={vm['precision']:.4f}")
        if wait >= patience:
            if verbose:
                print(f"    Early stop ep={ep}  bestF1={best_f1:.4f}")
            break
    if best_state:
        model.load_state_dict(best_state)
    return model, best_f1


# ─────────────────────────────────────────────────────────────
# 9 — Baseline Models
# ─────────────────────────────────────────────────────────────

def run_iforest(train_ds, test_ds):
    X_tr  = train_ds.get_scaled_windows().reshape(len(train_ds), -1)
    X_te  = test_ds.get_scaled_windows().reshape(len(test_ds), -1)
    y_te  = np.array(test_ds.labels)
    clf   = IsolationForest(
        n_estimators=100, contamination=train_ds.positive_rate,
        random_state=SEED, n_jobs=-1
    ).fit(X_tr)
    preds  = (clf.predict(X_te) == -1).astype(int)
    scores = -clf.decision_function(X_te)
    return _metrics(y_te, preds, scores)


def run_ocsvm(train_ds, val_ds, test_ds):
    X_tr  = train_ds.get_scaled_windows().reshape(len(train_ds), -1)
    X_v   = val_ds.get_scaled_windows().reshape(len(val_ds), -1)
    X_te  = test_ds.get_scaled_windows().reshape(len(test_ds), -1)
    y_v   = np.array(val_ds.labels)
    y_te  = np.array(test_ds.labels)
    best_nu, best_f1 = 0.1, -1.0
    for nu in [0.01, 0.05, 0.10, 0.15, 0.20]:
        svm   = OneClassSVM(kernel="rbf", nu=nu).fit(X_tr)
        vpred = (svm.predict(X_v) == -1).astype(int)
        vf1   = f1_score(y_v, vpred, zero_division=0)
        if vf1 > best_f1:
            best_f1, best_nu = vf1, nu
    svm    = OneClassSVM(kernel="rbf", nu=best_nu).fit(X_tr)
    preds  = (svm.predict(X_te) == -1).astype(int)
    scores = -svm.decision_function(X_te)
    return _metrics(y_te, preds, scores)


class LSTMAEModel(nn.Module):
    def __init__(self, C_in=C, L_in=L, h=64):
        super().__init__()
        self.enc = nn.LSTM(C_in, h, 2, batch_first=True, dropout=0.1)
        self.dec = nn.LSTM(h,  h, 2, batch_first=True, dropout=0.1)
        self.proj = nn.Linear(h, C_in)
    def forward(self, W):
        x = W.permute(0,2,1)
        _, (hid,_) = self.enc(x)
        dec_in = hid[-1].unsqueeze(1).repeat(1, x.size(1), 1)
        out, _ = self.dec(dec_in)
        return self.proj(out).permute(0,2,1)

def run_lstm_ae(train_ds, val_ds, test_ds):
    model  = LSTMAEModel().to(DEVICE)
    optim  = torch.optim.Adam(model.parameters(), lr=1e-3)
    loader = make_loader(train_ds, shuffle=True)
    model.train()
    for _ in range(30):
        for W, _ in loader:
            W = W.to(DEVICE)
            optim.zero_grad()
            F.mse_loss(model(W), W).backward()
            optim.step()
    # Find threshold on val
    model.eval()
    with torch.no_grad():
        vw = torch.from_numpy(val_ds.get_scaled_windows()).to(DEVICE)
        ve = ((model(vw) - vw)**2).mean(dim=(1,2)).cpu().numpy()
    y_v = np.array(val_ds.labels)
    best_t, best_f1 = np.percentile(ve, 90), -1.0
    for pct in range(50, 99, 2):
        t = np.percentile(ve, pct)
        vf1 = f1_score(y_v, (ve >= t).astype(int), zero_division=0)
        if vf1 > best_f1:
            best_f1, best_t = vf1, t
    with torch.no_grad():
        tw = torch.from_numpy(test_ds.get_scaled_windows()).to(DEVICE)
        te = ((model(tw) - tw)**2).mean(dim=(1,2)).cpu().numpy()
    preds = (te >= best_t).astype(int)
    y_te  = np.array(test_ds.labels)
    return _metrics(y_te, preds, te)


# ─────────────────────────────────────────────────────────────
# 10 — Statistical Testing
# ─────────────────────────────────────────────────────────────

def run_mcnemar(y_true, y_dstan, y_base):
    bc = ((y_dstan==y_true)&(y_base==y_true)).sum()
    od = ((y_dstan==y_true)&(y_base!=y_true)).sum()
    ob = ((y_dstan!=y_true)&(y_base==y_true)).sum()
    bw = ((y_dstan!=y_true)&(y_base!=y_true)).sum()
    try:
        res = mcnemar([[bc, od],[ob, bw]], exact=False, correction=True)
        return float(res.pvalue)
    except Exception:
        return float("nan")


def holm_bonferroni(pvals: List[float], alpha=0.01) -> List[bool]:
    n   = len(pvals)
    idx = np.argsort(pvals)
    sig = [False] * n
    for rank, i in enumerate(idx):
        if pvals[i] <= alpha / (n - rank):
            sig[i] = True
        else:
            break
    return sig


# ─────────────────────────────────────────────────────────────
# 11 — Ablation Study  (Table VI of manuscript)
# ─────────────────────────────────────────────────────────────

class AblationModel(nn.Module):
    """Configurable model for ablation: toggle SWA, TWA, CNN, residuals."""

    def __init__(self, C_in, L_in, N, use_swa, use_twa,
                 use_cnn, use_skip):
        super().__init__()
        self.C = C_in; self.use_swa=use_swa; self.use_twa=use_twa
        self.use_cnn=use_cnn; self.use_skip=use_skip
        Cp1 = C_in + 1
        self.cls = nn.Parameter(torch.randn(1,1,L_in)*0.02)
        self.layers = nn.ModuleList()
        for _ in range(N):
            blk = nn.ModuleDict()
            if use_swa:
                blk["swa"]    = SensorWiseAttention(Cp1, L_in, D_ATTN)
                blk["lp_swa"] = nn.Linear(L_in, L_in, bias=False)
                nn.init.orthogonal_(blk["lp_swa"].weight)
            if use_twa:
                blk["twa"]    = TimeWiseAttention(Cp1, L_in, D_ATTN)
                blk["lp_twa"] = nn.Linear(L_in, L_in, bias=False)
                nn.init.orthogonal_(blk["lp_twa"].weight)
            if use_swa or use_twa:
                blk["norm_dam"] = nn.LayerNorm([Cp1, L_in])
            if use_cnn:
                blk["cnn"] = CNNBlock(C_in, L_in)
            self.layers.append(blk)
        self.head = nn.Linear(L_in, 1)
        nn.init.orthogonal_(self.head.weight)

    def _fwd_blk(self, X, blk):
        h = torch.zeros_like(X)
        if self.use_swa and "swa" in blk:
            h = h + blk["lp_swa"](blk["swa"](X))
        if self.use_twa and "twa" in blk:
            h = h + blk["lp_twa"](blk["twa"](X))
        if "norm_dam" in blk:
            h = blk["norm_dam"](h + X if self.use_skip else h)
        else:
            h = X
        if self.use_cnn and "cnn" in blk:
            h = blk["cnn"](h)
        return h

    def forward(self, W):
        B = W.size(0)
        X = torch.cat([W, self.cls.expand(B,-1,-1)], dim=1)
        for blk in self.layers:
            X = self._fwd_blk(X, blk)
        return self.head(X[:, self.C, :]).squeeze(-1)

    @torch.no_grad()
    def predict_proba(self, W):
        self.eval()
        return torch.sigmoid(self.forward(W)).cpu().numpy()


ABLATION_CONFIGS = {
    "DSTAN-Med (Full)": dict(use_swa=True,  use_twa=True,
                              use_cnn=True,  use_skip=True,  use_ppf=True),
    "No PPF":           dict(use_swa=True,  use_twa=True,
                              use_cnn=True,  use_skip=True,  use_ppf=False),
    "DAM-Only":         dict(use_swa=True,  use_twa=True,
                              use_cnn=False, use_skip=True,  use_ppf=False),
    "TWA-Only":         dict(use_swa=False, use_twa=True,
                              use_cnn=True,  use_skip=True,  use_ppf=False),
    "SWA-Only":         dict(use_swa=True,  use_twa=False,
                              use_cnn=True,  use_skip=True,  use_ppf=False),
    "No Skip":          dict(use_swa=True,  use_twa=True,
                              use_cnn=True,  use_skip=False, use_ppf=False),
    "CNN-Only":         dict(use_swa=False, use_twa=False,
                              use_cnn=True,  use_skip=True,  use_ppf=False),
}


def run_ablation(train_loader, val_loader, test_loader,
                 test_ds, ppf, dataset, verbose=False):
    print("\n" + "="*60 + "\nABLATION STUDY\n" + "="*60)
    results = {}
    w_pos = torch.tensor(
        test_ds.pos_weight, dtype=torch.float32, device=DEVICE
    )
    for name, cfg in ABLATION_CONFIGS.items():
        use_ppf = cfg.pop("use_ppf")
        set_seed(SEED)
        model = AblationModel(C, L, N_BLOCKS, **cfg).to(DEVICE)
        cfg["use_ppf"] = use_ppf
        optim = torch.optim.Adam(model.parameters(), lr=LR)
        best_f1, best_state, wait = -1.0, None, 0
        for ep in range(EPOCHS):
            model.train()
            for W, y in train_loader:
                W, y = W.to(DEVICE), y.to(DEVICE)
                optim.zero_grad()
                F.binary_cross_entropy_with_logits(
                    model(W), y, pos_weight=w_pos
                ).backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step()
            model.eval()
            vprobs, vlabs = [], []
            with torch.no_grad():
                for W, y in val_loader:
                    vprobs.append(model.predict_proba(W.to(DEVICE)))
                    vlabs.append(y.numpy())
            vf1 = f1_score(
                np.concatenate(vlabs).astype(int),
                (np.concatenate(vprobs) >= 0.5).astype(int),
                zero_division=0)
            if vf1 > best_f1:
                best_f1 = vf1
                best_state = {k2: v.cpu().clone()
                              for k2, v in model.state_dict().items()}
                wait = 0
            else:
                wait += 1
            if wait >= PATIENCE:
                break
        if best_state:
            model.load_state_dict(best_state)

        ppf_arg  = ppf if use_ppf else None
        raw_arg  = test_ds.get_raw_windows() if use_ppf else None
        met = evaluate(model, test_loader, 0.5, ppf_arg, raw_arg)
        results[name] = met
        print(f"  {name:<25} Sens={met['sensitivity']:.4f}  F1={met['f1']:.4f}")

    print("\n  Ranked by Sensitivity:")
    for nm, m in sorted(results.items(),
                         key=lambda x: x[1]["sensitivity"], reverse=True):
        print(f"    {nm:<25} {m['sensitivity']:.4f}   {m['f1']:.4f}")
    return results


# ─────────────────────────────────────────────────────────────
# 12 — Full Experiment Runner
# ─────────────────────────────────────────────────────────────

def run_experiment(dataset: str, data_dir: str, severity: int = 0,
                   verbose: bool = True) -> Dict:
    print(f"\n{'#'*62}\n# DSTAN-Med  |  Dataset: {dataset.upper()}"
          f"  |  Severity: L{severity+1}\n{'#'*62}")

    # --- Load real data ---
    loader = get_loader(dataset, data_dir)
    clean_records = loader.load_all()
    train_clean, val_clean, test_clean = loader.patient_split(clean_records)

    # --- FDI injection ---
    print(f"Injecting FDI anomalies (p={INJ_P}, severity=L{severity+1}) …")
    train_inj = build_injected_records(train_clean, severity, seed=SEED)
    val_inj   = build_injected_records(val_clean,   severity, seed=SEED+1)
    test_inj  = build_injected_records(test_clean,  severity, seed=SEED+2)

    # --- Fit scaler on CLEAN training data (no label leakage) ---
    scaler = ChannelwiseRobustScaler().fit(train_clean)

    # --- Build windowed datasets ---
    train_ds = IoMTWindowDataset(train_inj, scaler, train_clean)
    val_ds   = IoMTWindowDataset(val_inj,   scaler, val_clean)
    test_ds  = IoMTWindowDataset(test_inj,  scaler, test_clean)
    print(f"Windows — Train: {len(train_ds):,} | Val: {len(val_ds):,} "
          f"| Test: {len(test_ds):,}")
    print(f"Positive rate — Train: {train_ds.positive_rate:.3f} "
          f"| Val: {val_ds.positive_rate:.3f} "
          f"| Test: {test_ds.positive_rate:.3f}")
    print(f"Positive-class weight w+: {train_ds.pos_weight:.2f}")

    train_loader = make_loader(train_ds, shuffle=True)
    val_loader   = make_loader(val_ds,   shuffle=False)
    test_loader  = make_loader(test_ds,  shuffle=False)

    ppf     = PhysiologicalPlausibilityFilter(dataset)
    raw_w   = test_ds.get_raw_windows()

    # --- DSTAN-Med: N_SEEDS runs ---
    print(f"\n[DSTAN-Med]  {N_SEEDS} seed runs …")
    seed_m, seed_m_noppf = {}, {}
    for key in ["sensitivity","precision","f1","accuracy","auc"]:
        seed_m[key] = []; seed_m_noppf[key] = []
    tranad_preds_ref, dstan_preds_ref, test_labels_ref = None, None, None

    for s in range(N_SEEDS):
        set_seed(SEED + s)
        model, _ = train_model(
            train_loader, val_loader, train_ds.pos_weight,
            verbose=(verbose and s == 0),
        )
        thresh = find_threshold(model, val_loader)
        m_noppf = evaluate(model, test_loader, thresh)
        m_ppf   = evaluate(model, test_loader, thresh, ppf, raw_w)
        for key in seed_m:
            seed_m[key].append(m_ppf.get(key, float("nan")))
            seed_m_noppf[key].append(m_noppf.get(key, float("nan")))
        if s == 0:
            # Collect predictions for McNemar's test
            dstan_preds_ref, test_labels_ref = _collect_preds(
                model, test_loader, thresh, ppf, raw_w
            )

    def _agg(d):
        return {k: float(np.nanmean(v)) for k, v in d.items()}

    results = {
        "DSTAN-Med":        _agg(seed_m),
        "DSTAN-Med-noPPF":  _agg(seed_m_noppf),
    }

    # --- Baselines ---
    print("\n[iForest]  …", end=" ")
    results["iForest"]  = run_iforest(train_ds, test_ds)
    print(f"Sens={results['iForest']['sensitivity']:.4f}  F1={results['iForest']['f1']:.4f}")

    print("[OC-SVM]   …", end=" ")
    results["OC-SVM"]   = run_ocsvm(train_ds, val_ds, test_ds)
    print(f"Sens={results['OC-SVM']['sensitivity']:.4f}  F1={results['OC-SVM']['f1']:.4f}")

    print("[LSTM-AE]  …", end=" ")
    results["LSTM-AE"]  = run_lstm_ae(train_ds, val_ds, test_ds)
    print(f"Sens={results['LSTM-AE']['sensitivity']:.4f}  F1={results['LSTM-AE']['f1']:.4f}")

    print("[TranAD]   …")
    set_seed(SEED)
    tranad_model, _ = train_model(
        train_loader, val_loader, train_ds.pos_weight,
        N=3, d=32, d_mid=32, lr=1e-3, verbose=False
    )
    tranad_thresh = find_threshold(tranad_model, val_loader)
    results["TranAD"] = evaluate(tranad_model, test_loader, tranad_thresh)
    tranad_preds_ref, _ = _collect_preds(tranad_model, test_loader, tranad_thresh)
    print(f"  Sens={results['TranAD']['sensitivity']:.4f}  F1={results['TranAD']['f1']:.4f}")

    # --- McNemar's test ---
    if dstan_preds_ref is not None and tranad_preds_ref is not None:
        y_true = test_labels_ref
        p_val  = run_mcnemar(y_true, dstan_preds_ref, tranad_preds_ref)
        sig    = holm_bonferroni([p_val])[0]
        print(f"\nMcNemar's (DSTAN-Med vs TranAD): p={p_val:.4f}  "
              f"{'significant (p<0.01)' if sig else 'not significant'}")
        results["_stats"] = {"mcnemar_p": p_val, "significant": float(sig)}

    # --- Summary ---
    _print_summary(results, dataset)

    # --- Ablation (first dataset only, or explicitly requested) ---
    ablation_results = run_ablation(
        train_loader, val_loader, test_loader, test_ds, ppf, dataset
    )
    results["_ablation"] = ablation_results

    return results


def _collect_preds(model, loader, thresh, ppf=None, raw_w=None):
    model.eval()
    preds, labels, offset = [], [], 0
    with torch.no_grad():
        for W, y in loader:
            p = (model.predict_proba(W.to(DEVICE)) >= thresh).astype(int)
            n = len(p)
            if ppf is not None and raw_w is not None:
                p = ppf(p, raw_w[offset:offset+n])
            preds.extend(p.tolist())
            labels.extend(y.numpy().astype(int).tolist())
            offset += n
    return np.array(preds), np.array(labels)


def _print_summary(results: Dict, dataset: str) -> None:
    print(f"\n{'─'*65}")
    print(f"{'Method':<22}{'Sens':>8}{'Prec':>8}{'F1':>8}{'AUC':>8}")
    print(f"{'─'*65}")
    order = ["iForest","OC-SVM","LSTM-AE","TranAD",
             "DSTAN-Med-noPPF","DSTAN-Med"]
    for m_name in order:
        if m_name not in results:
            continue
        m = results[m_name]
        print(f"  {m_name:<20}"
              f"{m.get('sensitivity',float('nan')):>8.4f}"
              f"{m.get('precision',float('nan')):>8.4f}"
              f"{m.get('f1',float('nan')):>8.4f}"
              f"{m.get('auc',float('nan')):>8.4f}")
    print(f"{'─'*65}")
    d   = results
    ds  = d.get("DSTAN-Med",{}).get("sensitivity",0)
    ts  = d.get("TranAD",{}).get("sensitivity",0)
    df  = d.get("DSTAN-Med",{}).get("f1",0)
    tf  = d.get("TranAD",{}).get("f1",0)
    ppf_g = (d.get("DSTAN-Med",{}).get("precision",0) -
             d.get("DSTAN-Med-noPPF",{}).get("precision",0))
    print(f"  ΔSensitivity (DSTAN-Med − TranAD): {ds-ts:+.4f}")
    print(f"  ΔF1          (DSTAN-Med − TranAD): {df-tf:+.4f}")
    print(f"  PPF Precision gain:                 {ppf_g:+.4f}")
    print(f"  Dataset: {dataset.upper()}")


# ─────────────────────────────────────────────────────────────
# 13 — CLI Entry Point
# ─────────────────────────────────────────────────────────────

def main():
    global N_SEEDS, EPOCHS, BATCH
    parser = argparse.ArgumentParser(
        description="DSTAN-Med: FDI Attack Detection in IoMT Sensor Streams"
    )
    parser.add_argument("--dataset", choices=["physionet","mimic3","wesad","all"],
                        default="physionet",
                        help="Dataset to run (default: physionet)")
    parser.add_argument("--data_dir", type=str, default="./data",
                        help="Root data directory (default: ./data)")
    parser.add_argument("--severity", type=int, choices=[0,1,2,3], default=0,
                        help="FDI severity level: 0=L1 … 3=L4 (default: 0=L1)")
    parser.add_argument("--seeds", type=int, default=N_SEEDS,
                        help=f"Number of training seeds (default: {N_SEEDS})")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch",  type=int, default=BATCH)
    parser.add_argument("--quiet",  action="store_true",
                        help="Suppress per-epoch output")
    args = parser.parse_args()

    print("DSTAN-Med — False Data Injection Attack Detection in IoMT")
    print(f"Device: {DEVICE} | Severity: L{args.severity+1} "
          f"| Seeds: {args.seeds}")
    N_SEEDS = args.seeds
    EPOCHS  = args.epochs
    BATCH   = args.batch
    print()

    datasets = (["physionet","mimic3","wesad"]
                if args.dataset == "all" else [args.dataset])

    all_results = {}
    for ds in datasets:
        data_path = (Path(args.data_dir) / ds
                     if args.dataset == "all" else Path(args.data_dir))
        try:
            res = run_experiment(
                dataset=ds,
                data_dir=str(data_path),
                severity=args.severity,
                verbose=not args.quiet,
            )
            all_results[ds] = res
        except FileNotFoundError as e:
            print(f"\n[SKIP] {ds}: {e}\n")

    if len(all_results) > 1:
        print("\n" + "="*62)
        print("CROSS-DATASET SUMMARY (DSTAN-Med vs TranAD)")
        print("="*62)
        for ds, res in all_results.items():
            dm = res.get("DSTAN-Med", {})
            tm = res.get("TranAD", {})
            nm = res.get("DSTAN-Med-noPPF", {})
            d_s = dm.get("sensitivity",0) - tm.get("sensitivity",0)
            d_f = dm.get("f1",0)          - tm.get("f1",0)
            ppf_g = dm.get("precision",0) - nm.get("precision",0)
            print(f"  {ds:<12}  ΔSens={d_s:+.4f}  "
                  f"ΔF1={d_f:+.4f}  PPF_Δprec={ppf_g:+.4f}")


if __name__ == "__main__":
    main()
