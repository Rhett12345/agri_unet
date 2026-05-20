# AGRI → GPM Precipitation Classification Pipeline

## Environment

Always use the `cloudunet` conda environment:

```bash
conda run -n cloudunet python <script>
# or activate first:
conda activate cloudunet
```

GPU: 2× NVIDIA GeForce RTX 4090. Set `CUDA_VISIBLE_DEVICES=1` if GPU 0 is occupied.

Matplotlib rendering may require:
```bash
LD_LIBRARY_PATH=/home/yuq/anaconda3/envs/cloudunet/lib:$LD_LIBRARY_PATH MPLCONFIGDIR=/tmp/matplotlib
```

## Project Structure

```
unet/
├── config.py          # Global config: paths, channels, hyperparams, date splits
├── fusion_config.py   # Fusion thresholds (time window, spatial params, env-overridable)
├── fusion_core.py     # AGRI→GPM spatial matching engine (cKDTree, patch extraction)
├── fusion_io.py       # File IO: AGRI FDI/GEO reads, GPM HDF5 reads, H5 writes
├── data_fusion.py     # Fusion scheduler: GPM+AGRI pairing, multiprocess orchestration
├── dataset.py         # PyTorch Dataset + stats computation
├── sample_filters.py  # Sample supervision filtering (stub for precip classification)
├── model.py           # SimpleUNet for precipitation classification (7ch → 4 classes)
├── losses.py          # WeightedCrossEntropyLoss + FocalLoss
├── train.py           # Training loop with AMP, multi-checkpoint saving
├── test.py            # Evaluation: OA, per-class F1, HSS, ETS, confusion matrix
├── inference.py       # Full-disk sliding-window precipitation inference
├── main.py            # Pipeline orchestrator (fuse/stats/train/test/infer)
├── tools/             # Diagnostic and utility scripts
│   ├── balance_split.py
│   ├── plot_validation_figures.py
│   └── visualize_fusion_geo.py
├── logs/              # Training logs
└── summary/           # Session summaries
```

## Pipeline Stages

1. **fuse** — GPM IMERG + AGRI data pairing, time matching (≤15 min), spatial resampling, outputs H5 samples
2. **stats** — Compute AGRI BT normalisation statistics from train split
3. **train** — Train SimpleUNet for 4-class precipitation classification
4. **test** — Evaluate on test split (OA, F1 per class, HSS, ETS, confusion matrix)
5. **infer** — Full-disk inference on new AGRI scenes

## Key Commands

```bash
# Full pipeline (step by step recommended)
conda run -n cloudunet python main.py --stages fuse --split train --workers 8
conda run -n cloudunet python main.py --stages stats
conda run -n cloudunet python main.py --stages train
conda run -n cloudunet python main.py --stages test

# Fusion only, one day
conda run -n cloudunet python data_fusion.py --split train --day 20190101 --workers 8

# Fusion with GPM grid subsampling and sample cap (quick test)
FUSION_GPM_SAMPLE_STEP=10 FUSION_MAX_SAMPLES_PER_SCENE=100 \
  conda run -n cloudunet python data_fusion.py --split train --day 20190101 --workers 1

# Training only
conda run -n cloudunet python main.py --stages train

# Test a specific checkpoint
conda run -n cloudunet python test.py --checkpoint model_gpm/AGRI_GPM_Precip_UNet_best.pth

# Full-disk inference
conda run -n cloudunet python main.py --stages infer --agri_file /path/to/FY4A_AGRI_*.HDF
```

## Input / Output

|         | Channels | Description |
|---------|----------|-------------|
| Input   | 7        | AGRI A01,A02,A03(vis) A09,A10(wv) A12,A13(ir) |
| Input   | 4        | Geo: lat, lon, VZA, SZA |
| Output  | 4        | Precipitation class logits |

## Precipitation Classes

| Class | Name          | Threshold (mm/h) |
|-------|---------------|-------------------|
| 0     | No-rain       | < 0.1             |
| 1     | Light rain    | 0.1 – 2.5         |
| 2     | Moderate rain | 2.5 – 8.0         |
| 3     | Heavy rain    | ≥ 8.0             |

## Paired HDF5 Format (samples_v3)

```
/Samples/agri      float32 (N, 7, 11, 11)   AGRI BT patches (7 channels)
/Samples/geo       float32 (N, 4, 11, 11)   lat, lon, VZA, SZA
/Samples/label     int32   (N,)              precipitation class 0-3
/Samples/precip    float32 (N,)              precipitation rate (mm/h)
/Samples/gpm_lat   float32 (N,)              GPM grid point latitude
/Samples/gpm_lon   float32 (N,)              GPM grid point longitude
/Samples/dt_min    float32 (N,)              AGRI-GPM time difference (minutes)
```

## Important Env Vars

| Variable | Default | Purpose |
|----------|---------|---------|
| `UNET_WORKDIR` | `/data/Data_yuq/unet_workdir` | Root output dir |
| `UNET_CHECKPOINT_MONITOR` | `val_f1_class3` | Checkpoint selection metric |
| `UNET_TRAIN_DATES` | config list | Override train dates |
| `UNET_VAL_DATES` | config list | Override val dates |
| `UNET_TEST_DATES` | config list | Override test dates |
| `UNET_LOSS_TYPE` | `weighted_ce` | Loss function: `weighted_ce` or `focal` |
| `FUSION_TIME_MAX_MIN` | `15.0` | Max AGRI-GPM time diff (minutes) |
| `FUSION_GPM_SAMPLE_STEP` | `1` | GPM grid subsampling step |
| `FUSION_MAX_SAMPLES_PER_SCENE` | `0` | Max samples per scene (0=unlimited) |
| `FUSION_N_WORKERS` | `cpu_count-1` | Number of fusion worker processes |

## Nighttime Handling

Visible channels (A01, A02, A03) are zeroed when SZA median > 85°. Input shape remains [7, 11, 11] — no dynamic channel removal.

## Design Conventions

- Do not change model structure or fusion thresholds unless explicitly requested
- Keep changes scoped; prefer editing existing files over creating new ones
- New diagnostics/scripts go under `tools/`
- Prefer CSV/JSON outputs for diagnostics
- Put new fusion outputs under separate experiment dirs, don't overwrite old results

打报告。
/home/yuq/cloudmask/GeoISCLD-Net/路径是原始代码路径可供参考，但是我们用的数据不一样，所以只能够参考
