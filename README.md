# Vision Transformer on CIFAR-100

Locality and hierarchy in Vision Transformers — comparing a Swin/hybrid
primary model against a parameter-matched plain ViT baseline.

**Status:** Part 1 (data pipeline) complete. Model, training, and evaluation
sections below will be filled in as those parts are built.

## Workflow

Code is developed locally in VS Code and version-controlled with git; actual
training runs on Colab's GPUs. The intended loop:

1. Edit code locally in VS Code.
2. Commit and push to GitHub.
3. In a Colab notebook, clone the repo and install dependencies (first cell,
   below).
4. Run training/evaluation from the notebook, which calls into `src/`.
5. Pull results (logs, figures) back down and commit them from VS Code.

### 1. Local environment setup (VS Code)

```bash
git clone <your-repo-url>.git
cd vision_transformer
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Run the test suite to confirm the environment is set up correctly:

```bash
python -m pytest tests/ -v
```

### 2. Colab bootstrap cell

Paste this as the first cell of any Colab notebook working on this project
(or just upload `colab_train.ipynb`, which already does this):

```python
!git clone https://github.com/<your-username>/<your-repo>.git
%cd vision_transformer
!pip install -r requirements.txt -q
```

`requirements.txt` intentionally has no version pins, so on Colab this
leaves the runtime's preinstalled, GPU-matched `torch`/`torchvision`/`numpy`
alone rather than forcing a version that conflicts with other preinstalled
packages (rasterio, jax, opencv, etc. all require numpy>=2) or silently
replaces Colab's CUDA-matched torch build with a mismatched one. It only
installs what's actually missing.

If the repo is private, use a GitHub personal access token in the clone URL
(`https://<token>@github.com/...`) or use Colab's GitHub auth integration
rather than committing credentials anywhere in the repo.

### 3. Dataset download

No manual download step — `torchvision.datasets.CIFAR100(download=True)` is
called automatically the first time `src/data.py` runs, and caches to
`./data/` (excluded from git via `.gitignore`). On Colab, this download
happens once per runtime session (~170 MB).

```python
from src.data import build_datasets, DataConfig

bundle = build_datasets(DataConfig(data_root="./data"))
print(f"train: {len(bundle.train)}  val: {len(bundle.val)}  test: {len(bundle.test)}")
```

This also runs the stratified split and the disjointness verification
(index-based for train/val, content-hash-based across all three splits)
automatically — it will raise a `ValueError` if anything is wrong rather
than silently producing bad splits.

### 4. Weights & Biases login (Colab, once per runtime)

```python
import wandb
wandb.login()  # paste your W&B API key when prompted
```

### 5. Training the primary model (Swin)

```bash
python -m src.train --config configs/primary.yaml
```

### 6. Training the plain ViT baseline

```bash
python -m src.train --config configs/vit_baseline.yaml
```

Both commands log live metrics to the `cifar100-vit-swin` W&B project and
also write a full CSV log to the path set in each config's
`training.log_path` (`logs/primary_training.csv` / `logs/vit_training.csv`),
satisfying the assignment's "exported logs" requirement independent of W&B.
Checkpoints (`last.pt`, `best.pt` by validation accuracy) are written to
`training.checkpoint_dir` in each config.

To sanity-check the training loop itself without downloading CIFAR-100 or
contacting W&B (useful right after cloning, before a real run):

```bash
python -m src.train --config configs/primary.yaml --dry-run --epochs 2
```

### 7. Evaluating a saved checkpoint

*To be added once `src/evaluate.py` exists (Part 5).*

### 8. Reproducing figures and metrics

*To be added once `src/evaluate.py` and the plotting utilities exist (Part 5).*

## Project structure

```
vision_transformer/
├── README.md
├── requirements.txt
├── .gitignore
├── .pylintrc
├── configs/
│   ├── primary.yaml       # Swin hyperparameters + shared training controls
│   └── vit_baseline.yaml  # ViT hyperparameters (parameter-matched to Swin)
├── src/
│   ├── data.py            # CIFAR-100 pipeline (Part 1) — done
│   ├── models.py          # Swin + ViT architectures (Part 2) — done
│   ├── train.py           # Training loop, W&B + CSV logging (Part 3) — done
│   ├── utils.py           # Seeding, device selection, config loading — done
│   ├── evaluate.py        # Evaluation + metrics (Part 5) — not yet built
│   └── metrics.py         # Metric computation — not yet built
├── tests/
│   ├── test_data.py       # 14 tests: splits, disjointness, transforms
│   ├── test_models.py     # 19 tests: shapes, param counts, checkpoints
│   └── test_train.py      # 11 tests: seeding, schedule, full dry-run loop
├── logs/                  # Exported CSV/JSON training logs (tracked in git)
└── figures/               # Training curves, confusion matrices, error examples
```

## Hardware

*To be filled in with actual Colab GPU tier and observed training time once
training runs are complete.*

## Known limitations

*To be filled in as they're identified during training.*
