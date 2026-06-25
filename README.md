# CrossTrack Validator — DLC vs. TopScan Pipeline

A Python-based validation pipeline for comparing markerless animal tracking outputs from **DeepLabCut** against a reference commercial tracking system, **TopScan**, in rodent behavioral experiments.

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Platform](https://img.shields.io/badge/platform-Google%20Colab%20%7C%20Jupyter-orange)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19323217.svg)](https://doi.org/10.5281/zenodo.19323217)

---

## Overview

**CrossTrack Validator** addresses a common challenge in behavioral neuroscience: validating whether a cost-effective, markerless pose estimation system can reproduce the spatial, locomotor, and behavioral outputs obtained with an established commercial tracking platform.

The pipeline was designed to compare **DeepLabCut** and **TopScan** tracking data in open-field and **Novel Object Recognition (NOR)** experiments. It performs coordinate normalization, frame-level synchronization, ROI-based behavioral classification, trajectory similarity analysis, positional error quantification, velocity profiling, discrimination index calculation, statistical testing, figure generation, and annotated video export.

The workflow is organized into two main stages:

| File | Purpose |
|---|---|
| `1_preprocessing_topscan_dlc.py` | Raw data cleaning, DLC TXT export, TopScan calibration, outlier removal, and coordinate preparation |
| `2_analysis_topscan_dlc.py` | Trajectory comparison, behavioral classification, statistical validation, figure export, and video annotation |

English and Portuguese versions are provided as both `.py` scripts and `.ipynb` notebooks.

---

## Main Features

- Synchronization of DeepLabCut and TopScan tracking files.
- Coordinate conversion and normalization to centimeters.
- ROI-based classification of familiar object, novel object, and floor states.
- Positional error metrics: MAE, RMSE, and maximum error.
- Trajectory similarity metrics: Pearson correlation, Fréchet distance, and DTW.
- Temporal synchrony analysis using cross-correlation and optimal lag.
- Velocity distribution and locomotor profile analysis.
- Discrimination Index calculation for Novel Object Recognition.
- Statistical comparison between tracking systems.
- Export of publication-ready figures in SVG, PDF, and PNG.
- Export of annotated videos for visual validation.

---

## Methods Summary

### Preprocessing

- **Likelihood filtering** — DeepLabCut predictions below the defined confidence threshold are discarded as `NaN` before downstream processing.
- **IQR outlier removal** — frame-to-frame displacement jumps above the IQR-based threshold are removed.
- **DBSCAN clustering** — isolated prediction glitches are identified and removed.
- **Savitzky-Golay smoothing** — high-frequency tracking noise is reduced while preserving movement kinematics.
- **Spatial calibration** — coordinates are converted and normalized to the experimental arena scale.
- **Temporal alignment diagnostics** — frame-level synchronization between systems is evaluated.

### Analysis

- **Trajectory metrics**: RMSE, MAE, maximum error, Fréchet distance, DTW, and Pearson correlation.
- **Behavioral classification**: frame-by-frame assignment to familiar object, novel object, or floor.
- **Discrimination Index (DI)**: comparison of cognitive outcome between TopScan and DeepLabCut.
- **Normality and statistical testing**: Shapiro-Wilk, paired t-test, Wilcoxon test, and related comparisons according to data distribution.
- **Bland-Altman analysis**: agreement between systems for behavioral outcomes.
- **Velocity profiles**: distributional comparison of movement speed.
- **Video annotation**: visual overlay of tracking and behavioral labels using OpenCV.

---

## Requirements

### Environment

- Python 3.9+
- Google Colab, or local Jupyter environment
- Google Drive is recommended for large video and output storage

### Installation

Install the dependencies manually:

```bash
pip install numpy>=1.24 pandas>=2.0 scipy>=1.11 scikit-learn>=1.3 \
            matplotlib>=3.7 seaborn>=0.12 pingouin>=0.5 \
            similaritymeasures>=0.7 opencv-python>=4.8 tqdm>=4.66
```

Or install from the requirements file:

```bash
pip install -r requirements.txt
```

For video conversion and video export in Google Colab:

```bash
apt-get install -q ffmpeg
pip install -q ffmpeg-python moviepy
```

---

## Repository Structure

```text
.
├── README.md
├── LICENSE
├── requirements.txt
├── CITATION.cff
│
├── notebooks/
│   ├── english/
│   │   └── DLC_vs_TopScan_pipeline_reorganized_EN.ipynb
│   └── portuguese/
│       └── Pipeline_DLC_vs_TopScan_fluxo_reorganizado.ipynb
│
├── scripts/
│   ├── english/
│   │   ├── 1_preprocessing_topscan_dlc.py
│   │   └── 2_analysis_topscan_dlc.py
│   └── portuguese/
│       ├── [1]pre_processamento_arquivos_topscan_dlc.py
│       └── [2]analise_topscan_dlc.py
│
└── Data/
    └── Examples/
        ├── README_example.md
        ├── DLC/
        │   ├── ID_AA_trajetoria_body.txt
        │   ├── ID_AA_trajetoria_nose.txt
        │   ├── ID_DF_trajetoria_body.txt
        │   ├── ID_DF_trajetoria_nose.txt
        │   ├── ID_FJ_trajetoria_body.txt
        │   └── ID_FJ_trajetoria_nose.txt
        └── TopScan/
            ├── AA-BB-CC_1_TCR_NOVO.TXT
            ├── AG-BG-CG_3_TCR_NOVO.TXT
            └── RR-PP-II_1_TCR_NOVO.TXT
```

---

## Example Data

Example tracking files are available in `Data/Examples/`.

The `TopScan/` folder contains three example TXT files exported from TopScan-like behavioral tracking data. The `DLC/` folder contains six DeepLabCut-derived TXT files, with paired `nose` and `body` trajectories for three example animals.

These files are included only to demonstrate the expected input structure and to allow users to test the pipeline format. They do not represent the full experimental dataset.

---

## Input Data Format

### TopScan TXT

TopScan-derived files are expected to contain frame number, center coordinates, and behavioral area labels:

```text
FrameNum CenterX(mm) CenterY(mm) Areas
0        142.35       98.12       Floor
1        143.01       98.44       Floor
...
```

The `Areas` column may include labels such as `Floor`, familiar-object labels, and novel-object labels. Portuguese labels such as `Novo` may be kept in the raw input files and mapped internally during analysis.

### DeepLabCut TXT

DeepLabCut-derived TXT files are expected separately for each tracked reference point, such as `nose` and `body`/`centerbody`:

```text
Frame X       Y
0     153.22  102.45
1     153.89  102.71
...
```

Frames with likelihood below the configured threshold may be exported as `NaN` and removed during analysis.

---

## Usage

### Option 1 — Google Colab / Jupyter notebooks

1. Open the English or Portuguese notebook version.
2. Mount Google Drive if running in Colab.
3. Configure the input and output paths in the configuration section.
4. Run the notebook sequentially from top to bottom.
5. Inspect the generated tables, figures, and annotated videos.

### Option 2 — Python scripts

#### Step 1 — Preprocessing

Open or run:

```text
scripts/english/1_preprocessing_topscan_dlc.py
```

Configure the paths and parameters at the top of the script, for example:

```python
experiment = "experiment"
bodypart_selected = "centerbody"
LIKELIHOOD_THRESHOLD = 0.9

base_path = ".../topscan/raw_txt_and_xlsx/"
output_path_dir = ".../topscan/calibrated_txt/"
```

Then execute the script in Google Colab, Jupyter, or a local Python environment.

#### Step 2 — Analysis

Open or run:

```text
scripts/english/2_analysis_topscan_dlc.py
```

Adjust the configuration section, for example:

```python
TOPSCAN_TXT_DIR = "/content/drive/MyDrive/.../topscan/calibrated_txt/"
DLC_TXT_DIR = "/content/drive/MyDrive/.../deeplabcut/tracking_txt/"
VIDEOS_DIR = "/content/drive/MyDrive/.../videos/"
OUTPUT_VIDEOS_DIR = "/content/drive/MyDrive/.../results/videos/"

FPS_VIDEO = 30
ARENA_WIDTH_CM = 60.0
ARENA_HEIGHT_CM = 60.0
```

Run the analysis script or call the main pipeline function if using the notebook version.

---

## Recommended Google Drive Structure

The paths can be adapted, but the following organization is recommended for Google Colab users:

```text
MyDrive/
└── crosstrack_validator/
    ├── topscan/
    │   ├── raw_txt_and_xlsx/          ← raw TopScan files
    │   └── calibrated_txt/            ← calibrated TopScan TXT files
    ├── deeplabcut/
    │   ├── h5_exports/                ← DLC .h5 exports
    │   └── tracking_txt/              ← per-bodypart TXT files
    ├── videos/                        ← original .mp4 videos
    └── results/
        ├── figures/                   ← exported figures
        └── videos/                    ← annotated videos
```

---

## DLC Model Information

- **Architecture**: ResNet-50
- **Training context**: rodents in laboratory arena settings
- **Tracked bodyparts**: `nose`, `neck`, `centerbody`, `tail_base`, `tail_middle`, `tail_tip`
- **Common reference bodyparts for analysis**: `nose`, `body`, or `centerbody`, depending on the analysis goal

---

## Outputs

The pipeline can generate:

- Cleaned and synchronized tracking tables.
- ROI-classified behavioral state tables.
- Positional error summaries.
- Trajectory similarity metrics.
- Discrimination Index tables.
- Statistical test results.
- Publication-ready figures in `.svg`, `.pdf`, and `.png` formats.
- Annotated videos with trajectory overlays and behavioral labels.

---

## Data Availability

This repository contains source code, notebooks, scripts, and small example TXT files. The example files are provided only for format demonstration and testing.

The archived software release is available through Zenodo:

> https://doi.org/10.5281/zenodo.19323217

Full experimental datasets should be cited separately if deposited in an independent data repository.

---

## Citation

If you use this pipeline in your research, please cite:

**Article**  
Update after publication:

> [Authors]. [Title of paper]. [Journal], [Year]. DOI: [to be added upon publication]

**Pipeline software**

> Orvate, R. (2026). *CrossTrack Validator: A Python Pipeline for DeepLabCut vs. TopScan Validation* (v1.0.0). Zenodo. https://doi.org/10.5281/zenodo.19323217

You can also use the **Cite this repository** button on GitHub, powered by the `CITATION.cff` file.

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
