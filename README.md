# RF Signal Classification

Modulation classification from raw IQ samples — a GPU-accelerated pipeline built with CuPy, PyTorch and GNU Radio.

> **Status:** Stage 1 (data) — see [Roadmap](#roadmap).

---

## Why this project

<!-- 3-5 sentences. The intersection you're claiming: SDR/DSP domain knowledge + deep learning
     + GPU computing. Say what a classifier over IQ data is actually for, and why the naive
     "just FFT it" approach falls over at low SNR. -->

## Results

<!-- Fill in as Stage 3 completes. Do not populate with anything you haven't measured. -->

| Model | Params | Overall acc. | Acc. @ -6 dB | Acc. @ 0 dB | Acc. @ +18 dB |
|---|---|---|---|---|---|
| Baseline CNN (raw IQ) | TODO | TODO | TODO | TODO | TODO |
| ResNet-style 1D CNN | TODO | TODO | TODO | TODO | TODO |
| Small transformer | TODO | TODO | TODO | TODO | TODO |

Per-SNR confusion matrices: `results/figures/`.

## Dataset

- **RadioML 2016.10a / 2018.01a** — <!-- modulation classes, SNR range, sample length, size on disk -->
- **Custom captures** — generated with GNU Radio flowgraphs in `flowgraphs/`. <!-- which modulations, channel impairments applied -->

Download: `scripts/download_radioml.sh`. Raw data is gitignored; the script is the reproducible path.

## Pipeline

```
IQ samples ──▶ CuPy preprocessing ──▶ model ──▶ per-SNR evaluation ──▶ inference server ──▶ agent
 (Stage 1)        (Stage 2)         (Stage 3)                            (Stage 4)      (Stage 5)
```

## Repository structure

```
rf-signal-classification/
├── configs/                    # experiment configs (one YAML per run)
├── data/
│   ├── raw/                    # gitignored — RadioML archives
│   ├── interim/                # gitignored — cached tensors
│   └── external/
├── flowgraphs/                 # GNU Radio .grc files for custom IQ generation
├── notebooks/
│   ├── 01-data-exploration.ipynb
│   └── 02-snr-breakdown.ipynb
├── src/rfml/
│   ├── data/
│   │   ├── radioml.py          # loaders for 2016.10a / 2018.01a
│   │   ├── gnuradio_gen.py     # programmatic IQ generation
│   │   └── splits.py           # stratified train/val/test by class AND SNR
│   ├── preprocessing/
│   │   ├── cupy_ops.py         # normalisation, framing — GPU
│   │   ├── augment.py          # phase rotation, frequency offset, AWGN
│   │   └── spectrogram.py
│   ├── models/
│   │   ├── baseline_cnn.py
│   │   ├── resnet1d.py
│   │   └── transformer.py
│   ├── training/
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   └── metrics.py          # per-SNR accuracy, confusion matrices
│   ├── serving/
│   │   └── api.py              # Stage 4 — inference endpoint
│   └── agent/
│       ├── tools.py            # classifier exposed as a callable tool
│       └── loop.py             # Stage 5 — raw-API agent loop
├── benchmarks/
│   └── cupy_vs_numpy.py
├── scripts/
│   ├── download_radioml.sh
│   └── run_experiment.py
├── tests/
│   ├── test_data.py
│   ├── test_preprocessing.py
│   └── test_models.py
├── results/
│   ├── figures/
│   └── tables/
└── docs/
    └── writeup.md              # the technical write-up
```

## Setup

```bash
git clone <repo-url> && cd rf-signal-classification
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

<!-- Pin the CUDA/CuPy pair explicitly — cupy-cuda12x vs cupy-cuda11x is the #1 setup failure. -->

**Requirements:** Python 3.11+, CUDA <!-- version -->, CuPy <!-- version -->, PyTorch <!-- version -->. GNU Radio <!-- version --> only needed for custom data generation.

## Usage

```bash
# Generate custom IQ data
python -m rfml.data.gnuradio_gen --config configs/capture_qpsk.yaml

# Train
python scripts/run_experiment.py --config configs/baseline_cnn.yaml

# Evaluate with per-SNR breakdown
python -m rfml.training.evaluate --checkpoint results/baseline_cnn/best.pt

# Benchmark the preprocessing pipeline
python benchmarks/cupy_vs_numpy.py
```

## Preprocessing benchmark

<!-- Stage 2 deliverable. Include batch size and GPU model — a speedup number without
     them is meaningless to a reviewer. -->

| Operation | NumPy (CPU) | CuPy (GPU) | Speedup |
|---|---|---|---|
| Normalisation | TODO | TODO | TODO |
| Phase rotation | TODO | TODO | TODO |
| Frequency offset | TODO | TODO | TODO |
| Spectrogram | TODO | TODO | TODO |

Hardware: <!-- GPU, CPU, batch size, dtype -->

## Design decisions

<!-- The section interviewers actually read. One short paragraph each, written as you go:
     - Why classify on raw IQ rather than spectrograms (or why both)
     - How the train/val/test split handles SNR stratification, and why leakage matters here
     - What the augmentations model physically (channel impairments, not generic noise)
     - Where the model fails and your hypothesis for why -->

## Testing

```bash
pytest                 # unit tests
pytest --cov=rfml      # with coverage
ruff check . && mypy src/
```

## Roadmap

- [ ] **Stage 1** — RadioML loaded; GNU Radio generation reproducible
- [ ] **Stage 2** — CuPy preprocessing + documented benchmark
- [ ] **Stage 3** — baseline CNN → ResNet1D → transformer, tracked experiments
- [ ] **Stage 4** — tests, type hints, write-up, served behind an inference endpoint
- [ ] **Stage 5** — agent layer answering natural-language questions about a capture

## References

<!-- O'Shea & West (RadioML), any modulation-classification papers you draw on,
     GNU Radio docs, CuPy docs -->

## License

<!-- MIT / Apache-2.0 -->
