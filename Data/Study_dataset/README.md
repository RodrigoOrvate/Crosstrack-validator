# Study dataset

Frame-resolved coordinate files analyzed in the article *Behavioral Validation of a
DeepLabCut-Based Pipeline for Novel Object Recognition Analysis in Rats* (18 test-session
recordings, one per rat; 30 frames/s). These are the inputs from which the reported results
were computed with the analysis pipeline in `Scripts/` and `Notebooks/`.

## Contents

```
Study_dataset/
├── TopScan/                     18 files, one per arena-specific recording
│   └── AA-BB-CC_1_TCR_NOVO.TXT  ...
└── DLC/
    ├── full/                    36 files (nose and mid-body x 18 recordings)
    │   └── ID_M1-AA_trajetoria_body_full.txt  ...
    └── holdouts/                36 files from the leave-one-mosaic-out networks
        ├── M1/ ... M6/
        │   └── ID_M1-AA_trajetoria_body_heldout_M1.txt  ...
```

Recordings come from six three-arena mosaics (M1-M6). Animal identifiers follow the pattern
`M<mosaic>-<code>` (for example, `M1-AA`); in the TopScan file names, the number after the
mosaic codes is the arena position (`AA-BB-CC_1` = `M1-AA`, `AA-BB-CC_2` = `M1-BB`, ...).

## Formats

**TopScan** (`*_TCR_NOVO.TXT`, space-separated): `FrameNum CenterX(mm) CenterY(mm) Areas`.
Raw TopScan body-center coordinates in the mosaic coordinate space (the analysis removes the
mosaic offset), organized per animal and merged with the TopScan exploration events.
`Areas` is `Floor` or the object region (`OBJ` + the object letter used in Fig. S1). When both
regions share a letter, the one with suffix `1` is the novel object; otherwise the pipeline
assigns the familiar and novel objects by position (see `_montar_mapa_areas_topscan`).
No filtering, smoothing, or calibration has been applied.

**DeepLabCut** (`ID_<animal>_trajetoria_<bodypart>_<network>.txt`, space-separated):
`Frame X Y`, in pixels of the arena-specific video. `body` is the mid-body landmark and `nose`
the nose landmark. Predictions with likelihood below 0.80 are `NaN`. `full` files come from the
network trained on all 18 recordings; `heldout_M<k>` files come from the network trained
without mosaic `M<k>`.

## Reproducing the analyses

Point the analysis pipeline to this folder:

```python
DIRETORIO_TOPSCAN_TXT = os.path.join(RAIZ, 'Data', 'Study_dataset', 'TopScan') + os.sep
DIRETORIO_DLC_TXT     = os.path.join(RAIZ, 'Data', 'Study_dataset', 'DLC') + os.sep
```

The object regions of interest calibrated for each recording and the manual exploration scores
of the two observers are defined in the analysis code. Filtering, alignment, smoothing, and the
exclusion of the first 150 frames are applied by the pipeline. Original videos are not included;
they are available from the corresponding author on reasonable request.
