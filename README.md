# AGRI → GPM Precipitation Classification

FY-4A/B AGRI 多光谱数据 → GPM IMERG 降水率四分类深度学习反演。

---

## Project layout

```
.
├── config.py          ← ALL paths and hyper-parameters (edit here only)
├── fusion_config.py   ← Fusion thresholds: time window, spatial params
├── fusion_core.py     ← AGRI→GPM spatial matching engine (cKDTree, patch extraction)
├── fusion_io.py       ← AGRI FDI/GEO + GPM HDF5 file IO, H5 write
├── data_fusion.py     ← Stage 1: GPM+AGRI multi-process fusion scheduler
├── sample_filters.py  ← Patch / sample supervision quality filters
├── dataset.py         ← Stage 2: PyTorch Dataset + normalisation statistics
├── model.py           ← U-Net architecture (7 AGRI + 4 geo channels → 4 classes)
├── losses.py          ← WeightedCrossEntropyLoss + FocalLoss
├── train.py           ← Stage 3: training loop with AMP, multi-checkpoint saving
├── test.py            ← Stage 4: evaluation (OA, F1, HSS, ETS, confusion matrix)
├── inference.py       ← Stage 5: full-disk sliding-window inference
├── main.py            ← Orchestrator (single entry point for all stages)
├── requirements.txt
├── tools/             ← Diagnostic and utility scripts
│   ├── balance_split.py
│   ├── plot_validation_figures.py
│   └── visualize_fusion_geo.py
├── logs/              ← Training logs
└── summary/           ← Session summaries
```

---

## Quick start

### 1 · Install dependencies

```bash
pip install -r requirements.txt
```

### 2 · Environment

Always use the `cloudunet` conda environment:

```bash
conda activate cloudunet
```

GPU: 2× NVIDIA GeForce RTX 4090. Set `CUDA_VISIBLE_DEVICES=1` if GPU 0 is occupied.

Matplotlib rendering may require:

```bash
LD_LIBRARY_PATH=/home/yuq/anaconda3/envs/cloudunet/lib:$LD_LIBRARY_PATH MPLCONFIGDIR=/tmp/matplotlib
```

### 3 · Edit config.py

Set data paths (defaults point to `/data/Data_yuq/`):

```python
AGRI_ROOT = Path("/your/AGRI/data")      # parent of YYYYMMDD/ day-folders
GPM_ROOT  = Path("/your/GPM/data")       # IMERG V07B HDF5 day-folders
ROOT      = Path("/your/workdir")        # all outputs written here
```

Date splits can be overridden via env vars `UNET_TRAIN_DATES`, `UNET_VAL_DATES`, `UNET_TEST_DATES`.

### 4 · Run the full pipeline

```bash
conda run -n cloudunet python main.py --stages fuse --split train --workers 8
conda run -n cloudunet python main.py --stages stats
conda run -n cloudunet python main.py --stages train
conda run -n cloudunet python main.py --stages test
```

Or single command (not recommended for first run):

```bash
conda run -n cloudunet python main.py --stages fuse stats train test
```

### 5 · Quick diagnostic run (single day, limited samples)

```bash
FUSION_GPM_SAMPLE_STEP=10 FUSION_MAX_SAMPLES_PER_SCENE=100 \
  conda run -n cloudunet python data_fusion.py --split train --day 20190101 --workers 1
```

---

## Data directory structure expected

```
AGRI_ROOT/
  20190101/
    FY4A-_AGRI--_N_DISK_xxx_L1-_FDI-_MULT_NOM_20190101000000_*.HDF
    FY4A-_AGRI--_N_DISK_xxx_L1-_GEO-_MULT_NOM_20190101000000_*.HDF
    ...

GPM_ROOT/
  20190101/
    3B-HHR.MS.MRG.3IMERG.20190101-S000000-E002959.0000.V07B.HDF5
    3B-HHR.MS.MRG.3IMERG.20190101-S003000-E005959.0030.V07B.HDF5
    ...
```

---

## Output structure

```
ROOT/
  paired_gpm/
    train/<YYYYMMDD>/GPM_AGRI_YYYYMMDD_HHMMSS_HHMMSS.h5
    val/  ...
    test/ ...
  stats_gpm/
    norm_stats.npz
  model_gpm/
    AGRI_GPM_Precip_UNet_best.pth          (best by monitored metric)
    AGRI_GPM_Precip_UNet_best_loss.pth     (best by val loss)
    AGRI_GPM_Precip_UNet_best_oa.pth       (best by overall accuracy)
    AGRI_GPM_Precip_UNet_best_f1_c3.pth    (best by heavy-rain F1)
    AGRI_GPM_Precip_UNet_last.pth          (most recent epoch)
  logs_gpm/
    pipeline.log
    train_log.csv
  retrieval_gpm/
    <stem>_precip.npz     (lat, lon, precip_class, precip_prob)
  eval_gpm/
    metrics_summary.csv
    classification_report.csv
    confusion_matrix.{svg,pdf,png}
```

---

## Paired HDF5 format (samples_v3)

```
/Samples/agri      float32 (N, 7, 11, 11)  AGRI BT patches (A01,A02,A03,A09,A10,A12,A13)
/Samples/geo       float32 (N, 4, 11, 11)  lat, lon, VZA, SZA (raw values)
/Samples/label     int32   (N,)             precipitation class 0–3
/Samples/precip    float32 (N,)             GPM precipitation rate (mm/h)
/Samples/gpm_lat   float32 (N,)             GPM grid latitude
/Samples/gpm_lon   float32 (N,)             GPM grid longitude
/Samples/dt_min    float32 (N,)             AGRI–GPM time difference (minutes)
```

File-level attributes: `format`, `task`, `agri_datetime`, `gpm_datetime`, `agri_channels`, `patch_size`, `num_classes`, `class_names`, `precip_thresholds`, `mode`, `num_samples`.

---

## Model input / output

|         | Channels | Description |
|---------|----------|-------------|
| Input   | 7        | AGRI A01,A02,A03(vis) A09,A10(wv) A12,A13(ir) |
| Input   | 4        | Geo: lat, lon, VZA, SZA |
| Output  | 4        | Precipitation class logits (No-rain / Light / Moderate / Heavy) |

Architecture: U-Net with 4 encoder stages (DoubleConv + MaxPool), bottleneck, 4 decoder stages (bilinear upsample + skip connection), single 1×1 conv head. Base channels: 64. Input is reflect-padded to multiples of 16 to support 11×11 patches.

---

## Precipitation classes

| Class | Name          | Threshold (mm/h) | Description       |
|-------|---------------|-------------------|--------------------|
| 0     | No-rain       | precip < 0.1      | No precipitation  |
| 1     | Light rain    | 0.1 ≤ p < 2.5     | Light rain        |
| 2     | Moderate rain | 2.5 ≤ p < 8.0     | Moderate rain     |
| 3     | Heavy rain    | p ≥ 8.0           | Heavy rain        |

---

## Nighttime handling

When SZA median > 85°, visible channels (A01, A02, A03) are set to zero. Input shape remains [7, 11, 11] — channels are never removed and the network architecture is static.

---

## Loss functions

| Loss                | Description                          |
|---------------------|--------------------------------------|
| `weighted_ce`       | CrossEntropy with inverse-frequency class weights (default) |
| `focal`             | Focal Loss with γ=2.0, class weights (set `UNET_LOSS_TYPE=focal`) |

---

## Evaluation metrics

| Metric | Description |
|--------|-------------|
| OA     | Overall Accuracy |
| F1_c0 ~ F1_c3 | Per-class F1-score (F1_c3 = heavy rain, most important) |
| HSS    | Heidke Skill Score |
| ETS    | Equitable Threat Score |
| Confusion matrix | Numerical + image output |
| Classification report | Precision, recall, F1, support per class |

---

## Key hyper-parameters (all in config.py)

| Parameter          | Default  | Description                         |
|--------------------|----------|-------------------------------------|
| AGRI_CHANNELS      | 7        | A01,A02,A03,A09,A10,A12,A13         |
| PATCH_SIZE         | (11, 11) | Training patch size                 |
| BATCH_SIZE         | 64       | Training batch size                 |
| NUM_EPOCHS         | 30       | Training epochs                     |
| LEARNING_RATE      | 1e-4     | AdamW initial LR                    |
| UNET_BASE_CHANNELS | 64       | UNet base width                     |
| NUM_CLASSES        | 4        | Precipitation classes               |
| GRAD_CLIP          | 1.0      | Gradient clipping max norm          |
| LR_PATIENCE        | 6        | ReduceLROnPlateau patience (epochs) |
| EARLY_STOP_PATIENCE| 10       | Early stopping patience (epochs)    |
| RANDOM_SEED        | 42       | Reproducibility seed                |

---

## Fusion parameters (fusion_config.py, env-overridable)

| Parameter               | Default | Description                          |
|-------------------------|---------|--------------------------------------|
| TIME_MAX_MIN            | 15.0    | Max AGRI–GPM time difference (min)   |
| PATCH_HALF              | 5       | Half-size for 11×11 patch            |
| GPM_SAMPLE_STEP         | 1       | GPM grid subsampling step            |
| MAX_SAMPLES_PER_SCENE   | 0       | Max samples per scene (0=unlimited)  |
| MIN_PRECIP_QUALITY      | 0.0     | Min GPM precipitationQualityIndex    |
| AGRI_DISK_MARGIN_DEG    | 5.0     | AGRI disk edge margin (degrees)      |
| N_FUSION_WORKERS        | cpu-1   | Number of fusion worker processes    |

---

## Key env vars

| Variable | Default | Purpose |
|----------|---------|---------|
| `UNET_WORKDIR` | `/data/Data_yuq/unet_workdir` | Root output directory |
| `UNET_CHECKPOINT_MONITOR` | `val_f1_class3` | Metric for best checkpoint |
| `UNET_LOSS_TYPE` | `weighted_ce` | Loss: `weighted_ce` or `focal` |
| `UNET_TRAIN_DATES` | config list | Override train dates |
| `UNET_VAL_DATES` | config list | Override val dates |
| `UNET_TEST_DATES` | config list | Override test dates |
| `FUSION_TIME_MAX_MIN` | `15.0` | Max AGRI–GPM Δt (minutes) |
| `FUSION_GPM_SAMPLE_STEP` | `1` | GPM grid subsampling |
| `FUSION_MAX_SAMPLES_PER_SCENE` | `0` | Per-scene sample cap |
| `CUDA_VISIBLE_DEVICES` | `1` | GPU selection (if GPU 0 busy) |
