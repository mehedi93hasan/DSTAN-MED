# DSTAN-Med: FDI Detection in IoMT

## Overview

Internet of Medical Things (IoMT) devices continuously transmit physiological vital signs across hospital networks. A False Data Injection (FDI) adversary with network-layer access can substitute or corrupt sensor readings with fabricated values that are individually plausible but clinically misleading — suppressing genuine alerts or triggering spurious ones that erode clinical trust.

**DSTAN-Med** addresses two fundamental gaps in existing IoMT FDI detectors:

| Gap | Existing Methods | DSTAN-Med |
| :--- | :--- | :--- |
| **Spatial + temporal FDI signatures** | Conflated in a single attention axis | Separate orthogonal SWA + TWA pathways |
| **Clinical domain knowledge** | Not exploited | PPF enforces per-channel plausibility bounds |

---
## Architecture

## Architecture Components

### 1. Dual-channel Attention Mechanism (DAM)
Routes the input through two structurally independent self-attention pathways operating on provably orthogonal tensor axes:
* **SWA (Sensor-Wise Attention):** Operates across the channel axis, capturing inter-sensor spatial correlations that FDI attacks distort.
* **TWA (Time-Wise Attention):** Operates across the time axis, capturing temporal trajectory distortions characteristic of ramp and drift attacks.

### 2. 1D-CNN Block
Residual depthwise convolution extracts local temporal texture at a granularity that global self-attention cannot efficiently represent.

### 3. Physiological Plausibility Filter (PPF)
A zero-parameter, inference-only module that suppresses positive predictions where all sensor readings simultaneously satisfy absolute clinical bounds and rate-of-change limits — providing partial adversarial robustness at zero computational cost.


## Key Results

All results are averaged over 5 random seeds, at severity L1 (lowest, most clinically consequential), averaged across all four FDI morphologies. Improvements over TranAD are significant at $p < 0.01$ (McNemar's test, Holm–Bonferroni corrected).

### Table 1 — Comprehensive Detection Results

| Method | Params | PhysioNet Sens | PhysioNet F1 | MIMIC-III Sens | MIMIC-III F1 | WESAD Sens | WESAD F1 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **iForest** | — | 39.9 | 47.3 | 37.6 | 44.7 | 35.1 | 42.2 |
| **OC-SVM** | — | 37.2 | 44.8 | 36.4 | 43.7 | 34.8 | 42.1 |
| **LSTM-AE** | 1.2M | 55.0 | 62.4 | 53.6 | 61.0 | 49.7 | 57.2 |
| **TranAD** | 3.8M | 65.6 | 72.1 | 64.2 | 71.0 | 60.8 | 67.9 |
| **DSTAN-Med** | **2.1M** | **73.3** | **80.0** | **72.5** | **79.0** | **68.2** | **75.4** |
| *Δ vs. TranAD* | *—* | *+7.7\** | *+7.9* | *+8.3\** | *+8.0* | *+7.4\** | *+7.5* |

> *All sensitivity and F1 values are in %. \*p < 0.01, McNemar's test, Holm–Bonferroni corrected.*

---

### Table 2 — PPF Contribution (PhysioNet-2012)

| Metric | Without PPF | With PPF | Gain |
| :--- | :--- | :--- | :--- |
| **Sensitivity (%)** | 73.1 | **73.3** | +0.2 pp |
| **Precision (%)** | 83.8 | **87.6** | +3.8 pp |
| **F1 (%)** | 78.0 | **80.0** | +2.0 pp |
| **AUC-ROC** | 0.924 | **0.924** | 0.0 |

---

### Table 3 — Ablation Study (PhysioNet-2012, Mixed-Type Conditions)

| Configuration | Sensitivity (%) | F1 (%) | Δ Sensitivity |
| :--- | :--- | :--- | :--- |
| **DSTAN-Med (Full)** | **88.6** | **90.1** | **—** |
| No PPF | 88.4 | 88.1 | −0.2 |
| DAM-Only (no CNN) | 82.1 | 85.6 | −6.5 |
| TWA-Only (no SWA) | 79.4 | 83.5 | −9.2 |
| SWA-Only (no TWA) | 76.7 | 81.7 | −11.9 |
| No Skip (no residuals) | 74.6 | 79.4 | −14.0 |
| CNN-Only (no DAM) | 71.3 | 77.8 | −17.3 |

> 💡 **Note:** Removing residual connections is the single most harmful architectural ablation (−14.0 pp sensitivity).
