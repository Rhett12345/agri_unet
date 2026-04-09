# AGRI + MYD06 Cloud Property Retrieval Pipeline

Retrieves cloud properties (CLP, CER, COT, CTH) from FY-4B AGRI FDI/GEO data,
supervised by MYD06 (Aqua/MODIS) labels, using the HIR-COMP-UNet architecture.

---

## Project layout

```
.
├── config.py         ← ALL paths and hyper-parameters (edit here only)
├── data_fusion.py    ← Stage 1: match AGRI + MYD06, write paired HDF5
├── dataset.py        ← Stage 2: Dataset class + normalisation statistics
├── model.py          ← Network architecture (AGRI-only variant)
├── train.py          ← Stage 3: training loop
├── test.py           ← Stage 4: evaluation + metrics
├── inference.py      ← Stage 5: full-disk retrieval
├── main.py           ← Orchestrator (single entry point)
└── requirements.txt
```

---

## Quick start

### 1 · Install dependencies
```bash
pip install -r requirements.txt
```

### 2 · Edit config.py
Set at minimum:
```python
AGRI_ROOT  = Path("/your/AGRI/data")    # parent of YYYYMMDD/ day-folders
MODIS_ROOT = Path("/your/MYD06/data")   # parent of YYYYMMDD/ day-folders
ROOT       = Path("/your/project")      # all outputs written here
TRAIN_DATES = ["20230601", "20230602"]  # or leave [] to use all folders
VAL_DATES   = ["20230615"]
TEST_DATES  = ["20230701"]
```

### 3 · Run the full pipeline
```bash
python main.py --stages fuse stats train test
```

Or step by step:
```bash
# Fuse training data (one day at a time if needed)
python main.py --stages fuse --split train --day 20230601

# Compute normalisation statistics (uses PAIRED_TRAIN_DIR)
python main.py --stages stats

# Train the model
python main.py --stages train

# Evaluate on test split
python main.py --stages test

# Full-disk inference on a new AGRI file
python main.py --stages infer --agri_file /data/raw/AGRI/20230815/FY4B_AGRI_*.HDF
```

---

## Data directory structure expected

```
AGRI_ROOT/
  20230601/
    FY4B-_AGRI--_N_DISK_xxxxxxxx_20230601060000_L1.HDF
    FY4B-_AGRI--_N_DISK_xxxxxxxx_20230601061500_L1.HDF
    ...
  20230602/
    ...

MODIS_ROOT/
  20230601/
    MYD06_L2.A2023152.0600.061.*.hdf
    MYD06_L2.A2023152.0605.061.*.hdf
    ...
  20230602/
    ...
```

---

## Output structure

```
ROOT/
  paired/
    train/<YYYYMMDD>/AGRI_MYD06_pair_YYYYMMDD_HHMMSS.h5
    val/  ...
    test/ ...
  stats/
    norm_stats.npz
  model/
    HIR_COMP_UNet_AGRIonly_best.pth
    HIR_COMP_UNet_AGRIonly_last.pth
  logs/
    pipeline.log
    train_log.csv
  retrieval/
    <stem>_retrieval.npz   (lat, lon, CLP_pred, CER_pred, COT_pred, CTH_pred, CLP_prob)
  eval/
    metrics_summary.csv
    confusion_matrix.png
    scatter_CER.png
    scatter_COT.png
    scatter_CTH.png
```

---

## Paired HDF5 format (produced by data_fusion.py)

```
/AGRI/Geolocation/{lat, lon, VZA, SZA}   float32 (H, W)
/AGRI/Aux/ELE                             float32 (H, W)   surface elevation (m)
/AGRI/BT/ch{NN}                           float32 (H, W)   brightness temperature (K)
/Labels/{CLP, CER, COT, CTH}             float32 (H, W)
```
`CLP` is an integer class (0=clear, 1=water, 2=supercool, 3=mixed, 4=ice).
`CER` in µm, `COT` dimensionless, `CTH` in metres.

---

## Label channels in model output

| Channel | Variable | Unit      | Notes                          |
|---------|----------|-----------|--------------------------------|
| 0       | CLP      | class 0-4 | CrossEntropy target            |
| 1       | CER      | µm        | SmoothL1, z-score normalised   |
| 2       | COT      | –         | SmoothL1, z-score normalised   |
| 3       | CTH      | m         | SmoothL1, z-score normalised   |

---

## Key hyper-parameters (all in config.py)

| Parameter            | Default | Description                         |
|----------------------|---------|-------------------------------------|
| AGRI_BT_CHANNEL_INDICES | [8..14] | 7 thermal channels (0-based)     |
| PATCH_SIZE           | (32,32) | Training patch size                 |
| BATCH_SIZE           | 64      | Training batch size                 |
| NUM_EPOCHS           | 50      | Training epochs                     |
| LEARNING_RATE        | 1e-4    | Adam initial LR                     |
| MODEL_BASE_CHANNELS  | 32      | UNet base width                     |
| TRANSFORMER_DEPTH    | 4       | Bottleneck transformer layers       |
| MAX_VZA_DEG          | 65      | Satellite zenith angle filter       |
| MAX_SZA_DEG          | 65      | Solar zenith angle filter (day)     |
| MAX_TIME_DIFF_MIN    | 15      | AGRI–MODIS temporal match window    |
| MAX_MATCH_DIST_KM    | 3.0     | AGRI–MODIS spatial match radius     |
