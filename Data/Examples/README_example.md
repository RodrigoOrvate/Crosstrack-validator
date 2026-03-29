# Example Data

This folder contains demonstration data for two animals (`RATO_02` and `RATO_05`),
representing the **output of Script 1 (preprocessing)** — the files that serve as
direct input to **Script 2 (analysis)**.

> These files use real trajectory structure and coordinate ranges but are **not** the
> original research data. They are provided solely to allow the pipeline to be tested
> without access to the full dataset.

---

## Folder Contents

```
example/
├── topscan/
│   ├── RATO_02_topscan.txt    ← calibrated TopScan trajectory (mm → cm)
│   └── RATO_05_topscan.txt
└── dlc/
    ├── RATO_02_centerbody_dlc.txt    ← DLC trajectory, bodypart: centerbody
    └── RATO_05_centerbody_dlc.txt
```

---

## File Format

### TopScan TXT
Space-separated. Produced by Script 1 after outlier removal, Savitzky-Golay
smoothing, and Huber regression calibration.

```
FrameNum CenterX(mm) CenterY(mm) Areas
0        142.35      98.12       Floor
1        143.01      98.44       Floor
2        143.67      99.01       Objeto Familiar
...
```

| Column | Description |
|---|---|
| `FrameNum` | Frame index (integer, starting at 0) |
| `CenterX(mm)` | X coordinate in millimetres |
| `CenterY(mm)` | Y coordinate in millimetres |
| `Areas` | Behavioural zone label assigned by TopScan |

### DLC TXT
Space-separated. Produced by Script 1 after likelihood filtering (> 0.9),
outlier removal, and Savitzky-Golay smoothing. Coordinates remain in pixels
and are converted to cm during Script 2.

```
Frame X       Y
0     153.22  102.45
1     153.89  102.71
2     154.31  103.10
...
```

| Column | Description |
|---|---|
| `Frame` | Frame index (integer, starting at 0) |
| `X` | X coordinate in pixels |
| `Y` | Y coordinate in pixels |

Frames where DLC likelihood was below 0.9 were removed during preprocessing
and are absent from this file.

---

## How to Use

1. Point the `CONFIGURATION` section of `2_analysis_topscan_dlc.py` to these folders:

```python
TOPSCAN_TXT_DIR = '/content/drive/MyDrive/.../data/example/topscan/'
DLC_TXT_DIR     = '/content/drive/MyDrive/.../data/example/dlc/'
```

2. Set `RATOS_PARA_GERAR_VIDEO = []` to skip video generation (no video files are
   included in this example).

3. Run `run_pipeline()`. All statistical analyses (Sections 4.1–4.9) will execute
   on the two example animals.

---

## Raw Data Availability

The original raw files used in the study — `.h5` pose estimation exports
(DeepLabCut), multi-animal `.mpg` videos, and TopScan `.xlsx` event logs — are
available in the companion data repository on Zenodo:

> [Dataset DOI — to be added upon publication]

These files are required to run Script 1 (preprocessing) from scratch.

---

## Recording Conditions

| Parameter | Value |
|---|---|
| Paradigm | Novel Object Recognition (NOR) |
| Arena | 60 × 60 cm open field |
| Camera | Overhead, fixed mount |
| Frame rate | 30 fps |
| Video resolution | 310–330 × 224–240 px (cropped per animal) |
| Animals per raw video | 3 (cropped into individual files by Script 1) |
| DLC model | ResNet-50, 100k iterations, rodent-specific |
| Reference bodypart | `centerbody` |