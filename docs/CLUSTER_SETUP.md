# DrivoR — Cluster Setup (Narval)

## 1. Clone and install

```bash
git clone https://github.com/valeoai/DrivoR
cd DrivoR
uv venv --python 3.10
source .venv/bin/activate
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
pip install -e ./nuplan-devkit
pip install -e .
```

## 2. Download maps

```bash
cd download
bash download_maps.sh
cd ..
```

This downloads the nuPlan maps to `download/maps/`. These must live in a writable directory (not inside the read-only squashfs dataset mount).

## 3. Dataset

The NAVSIM dataset is packaged as a squashfs image at:

```
/project/6061241/rdesc/navsim.sqsh
```

The SLURM scripts automatically mount it at job start via `extract_navsim_nibi.sh`. No manual download needed.

The extraction script mounts the squashfs to `$SLURM_TMPDIR` and creates symlinks under `DrivoR/dataset/`. It sets the following environment variables:

| Variable | Value |
|---|---|
| `OPENSCENE_DATA_ROOT` | `DrivoR/dataset` |
| `NAVSIM_EXP_ROOT` | `DrivoR/exp` |
| `NAVSIM_DEVKIT_ROOT` | `DrivoR/` |

`NUPLAN_MAPS_ROOT` is overridden in each SLURM script to point to the downloaded maps.

## 4. Metric caching (CPU job, run once)

Pre-computes PDM-Closed planner results needed for the training loss. This is a CPU-only job.

```bash
sbatch metric_caching_nibi.sh
```

Monitor progress:

```bash
tail -f navsim-metric-cache-<JOBID>.out
```

The cache is written to `exp/train_metric_cache/`. This only needs to be done once.

## 5. Training (GPU job)

```bash
sbatch train_drivor.sh
```

## Customizing paths

The SLURM scripts (`metric_caching_nibi.sh`, `train_drivor.sh`) hardcode paths under `/scratch/rdesc/DrivoR`. If your repo lives elsewhere, update the following in each script:

- `.venv/bin/activate` path
- `extract_navsim_nibi.sh` argument (repo root)
- `NUPLAN_MAPS_ROOT` (maps directory)
