# Crosstrack-validator
## DLC-TopScan Validation Pipeline

A Python pipeline for technical validation of markerless animal tracking (DeepLabCut) against a gold-standard reference system (TopScan) in rodent behavioral experiments.

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Platform](https://img.shields.io/badge/platform-Google%20Colab%20%7C%20Jupyter-orange)

---

## Overview

This pipeline addresses a common challenge in behavioral neuroscience: quantitatively validating a cost-effective, markerless pose estimation system (DeepLabCut, ResNet-50) against an established commercial tracking solution (TopScan) for open-field and novel object recognition (NOR) experiments.

The pipeline is organized in two sequential stages:

| Script | Description |
|---|---|
| `1_preprocessing_topscan_dlc.py` | Raw data cleaning, outlier removal, spatial calibration |
| `2_analysis_topscan_dlc.py` | Trajectory comparison, behavioral metrics, and statistical validation |

Portuguese-language versions (`[1]pre_processamento_arquivos_topscan_dlc.py` and `[2]análise_topscan_dlc.py`) are included for use in the original laboratory environment.

---

## Methods Summary

### Preprocessing (`Script 1`)
- **Likelihood filtering** — DLC predictions below 0.9 confidence are discarded as `NaN` before any processing
- **IQR outlier removal** — frame-to-frame displacement jumps above `Q3 + 2.5 × IQR` are removed
- **DBSCAN clustering** — isolated prediction glitches are identified and removed (`eps=15 px`, `min_samples=3`)
- **Savitzky-Golay smoothing** — high-frequency noise removed without distorting movement kinematics (`window=11 frames`, `polyorder=3`)
- **Huber regression calibration** — robust pixel-to-mm spatial mapping with intercept correction (R² threshold: 0.90)
- **Temporal alignment diagnostics** — FFT cross-correlation to detect frame offset between systems

### Analysis (`Script 2`)
- **Trajectory metrics**: RMSE, MAE, Fréchet distance, DTW, Pearson correlation
- **Behavioral classification**: frame-by-frame ROI assignment (familiar object / novel object / floor)
- **Discrimination Index (DI)**: Shapiro-Wilk normality test + paired t-test (TopScan vs. DLC)
- **Bland-Altman analysis**: agreement of total exploration time between systems
- **Temporal cross-correlation**: group-level synchrony analysis between displacement signals
- **Spatial error analysis**: center vs. periphery zone comparison
- **Velocity profiles**: kernel density estimation + Kolmogorov-Smirnov test
- **Video annotation**: trajectory overlay exported via OpenCV

---

## Requirements

### Environment
- Python 3.9+
- Google Colab (recommended) or local Jupyter with the packages below

### Installation

```bash
pip install numpy>=1.24 pandas>=2.0 scipy>=1.11 scikit-learn>=1.3 \
            matplotlib>=3.7 seaborn>=0.12 pingouin>=0.5 \
            similaritymeasures>=0.7 opencv-python>=4.8 tqdm>=4.66
```

Or install from the requirements file:

```bash
pip install -r requirements.txt
```

For video conversion (Colab only):
```bash
apt-get install -q ffmpeg
pip install -q ffmpeg-python moviepy
```

---

## Repository Structure

```
.
├── README.md
├── LICENSE
├── requirements.txt
├── CITATION.cff
│
├── 1_preprocessing_topscan_dlc.py      # English — preprocessing pipeline
├── 2_analysis_topscan_dlc.py           # English — analysis & statistics pipeline
│
├── [1]pre_processamento_arquivos_topscan_dlc.py   # Portuguese (original)
├── [2]análise_topscan_dlc.py                      # Portuguese (original)
│
└── data/
    └── example/
        ├── RATO_00_topscan.txt     # Example TopScan output format
        └── RATO_00_dlc.txt         # Example DLC output format
```

---

## Input Data Format

### TopScan TXT (`RATO_XX_topscan.txt`)
Space-separated, produced by Script 1:
```
FrameNum CenterX(mm) CenterY(mm) Areas
0        142.35       98.12       Floor
1        143.01       98.44       Floor
...
```

### DLC TXT (`RATO_XX_dlc.txt`)
Space-separated, produced by Script 1:
```
Frame X       Y
0     153.22  102.45
1     153.89  102.71
...
```
Frames with likelihood < 0.9 are exported as `NaN` and dropped during analysis.

---

## Usage

### Step 1 — Preprocessing

1. Open `1_preprocessing_topscan_dlc.py` in Google Colab
2. Mount your Google Drive
3. Configure the paths and parameters at the top of each section:

```python
# DLC export section
experiment           = "experiment"    # prefix of .h5 files
bodypart_selected    = 'centerbody'    # bodypart of interest
LIKELIHOOD_THRESHOLD = 0.9             # frames below this → NaN

# TopScan calibration section
base_path      = ".../TOPSCAN/TXT original/"   # raw TXT and XLSX files
output_path_dir = ".../TOPSCAN/TXT new/"        # calibrated output folder
```

4. Run all cells top-to-bottom

### Step 2 — Analysis

1. Open `2_analysis_topscan_dlc.py` in Google Colab
2. Adjust the `CONFIGURATION` section at the top:

```python
TOPSCAN_TXT_DIR   = '/content/drive/MyDrive/.../TOPSCAN/TXT new/'
DLC_TXT_DIR       = '/content/drive/MyDrive/.../DEEPLABCUT/trajetoria_txt_dlc/'
VIDEOS_DIR        = '/content/drive/MyDrive/.../videos/'
OUTPUT_VIDEOS_DIR = '/content/drive/MyDrive/.../resultados/videos/'

FPS_VIDEO        = 30
ARENA_WIDTH_CM   = 60.0
ARENA_HEIGHT_CM  = 60.0
```

3. Call `run_pipeline()` at the bottom of the notebook

---

## Expected Google Drive Structure

```
MyDrive/
└── content/
    └── validacao_topscan/
        ├── TOPSCAN/
        │   ├── TXT original e XLSX original/  ← raw TopScan files (input to Script 1)
        │   └── TXT novo/                      ← calibrated files (output of Script 1 → input to Script 2)
        ├── DEEPLABCUT/
        │   ├── h5_validacao/                  ← DLC .h5 exports (input to Script 1)
        │   └── trajetoria_txt_dlc/            ← per-bodypart TXTs (output of Script 1 → input to Script 2)
        ├── videos_ratinho/                    ← original .mp4 videos
        └── resultados/
            └── videos/                        ← annotated videos (output of Script 2)
```

---

## DLC Model

- **Architecture**: ResNet-50
- **Training iterations**: 100,000
- **Training data**: rodents in laboratory arena settings
- **Tracked bodyparts**: `nose`, `neck`, `centerbody`, `tail_base`, `tail_middle`, `tail_tip`
- **Reference bodypart for statistics**: `centerbody`

---

## Citation

If you use this pipeline in your research, please cite:

> [Authors]. [Title of paper]. [Journal], [Year]. DOI: [to be added upon publication]

You can also cite the software repository directly using the `CITATION.cff` file included in this repository.

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

