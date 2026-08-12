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
    ├── writeup.md              # the technical write-up
    └── environment.md          # dated record of verified versions + memory ceilings
```

## Setup

Developed on an **NVIDIA DGX Spark (GB10 Grace Blackwell)**. The GPU reports compute capability **sm_121**, which is not what most prebuilt wheels target — see [Platform notes](#platform-notes-dgx-spark--gb10) before deviating from these commands.

### Target environment

| Component | Version | Notes |
|---|---|---|
| Hardware | NVIDIA GB10 (Grace Blackwell), sm_121 | 128 GB unified memory (CPU+GPU shared) |
| Architecture | aarch64 (ARM) | rules out most x86-only wheels |
| OS | Ubuntu 24.04 (DGX OS base) | |
| NVIDIA driver | 580.173.02 | CUDA 13.0 branch |
| CUDA Toolkit | 13.0 | fixed by the DGX LTS stack — don't upgrade in place |
| Python | 3.12.3 | |
| PyTorch | <!-- record what `pip show torch` reports --> | from the cu130 index |
| CuPy | <!-- record what `cupy.__version__` reports --> | `cupy-cuda13x`, ≥13.6 for CUDA 13 |
| GNU Radio | <!-- version --> | only needed for custom IQ generation |

### Install

```bash
git clone <repo-url> && cd rf-signal-classification
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip setuptools wheel

# PyTorch — must come from the cu130 index; PyPI default has no aarch64+CUDA 13 build
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130

# CuPy — CUDA 13.x wheels, aarch64 included
pip install cupy-cuda13x

# This project
pip install -e ".[dev]"
```

`pyproject.toml` deliberately does **not** pin `torch` or `cupy`. Pinning them causes pip to resolve an x86 or CPU-only build on this platform (see [silent CPU fallback](#silent-cpu-fallback)). Both are installed explicitly, before the project.

### Verify the install

Run this before trusting any benchmark number. A CPU-fallback install will produce plausible-looking results at a fraction of the expected speed.

```python
import torch, cupy as cp

p = torch.cuda.get_device_properties(0)
print(f"Device:       {p.name}")                       # NVIDIA GB10
print(f"Capability:   sm_{p.major}{p.minor}")          # sm_121
print(f"Torch archs:  {torch.cuda.get_arch_list()}")   # sm_120 expected, not sm_121
print(f"Torch CUDA:   {torch.version.cuda}")
print(f"CUDA avail:   {torch.cuda.is_available()}")
print(f"CuPy:         {cp.__version__}")
print(f"CuPy runtime: {cp.cuda.runtime.runtimeGetVersion()}")   # 13xxx

# Actually execute on the device
a = torch.randn(1000, 1000, device="cuda")
print(f"Torch matmul: {(a @ a).sum().item():.2f}")
print(f"CuPy matmul:  {float((cp.random.randn(1000, 1000) ** 2).sum()):.2f}")
```

Record the output in `docs/environment.md` with the date. This stack has been changing month to month; a dated record of what worked is what you'll want when something breaks later.

### Platform notes (DGX Spark / GB10)

**sm_121 vs sm_120.** PyTorch wheels compile kernels up to sm_120, and the GB10 is sm_121. This is fine — **sm_121 is binary compatible with sm_120**, so those kernels run natively. `get_arch_list()` showing `sm_120` without `sm_121` is expected, not a fault. You may see a warning on first CUDA call; it is safe to ignore for ordinary training. The well-documented sm_121 breakages (Flash Attention, CUTLASS MoE kernels, CUDA graph capture) affect high-throughput LLM serving, not the convolutional and small-transformer workloads here.

CuPy avoids the problem entirely: its `ElementwiseKernel` / `RawKernel` code is JIT-compiled by NVRTC against the device actually present, so the preprocessing kernels target sm_121 natively. If using Triton or `torch.compile`, set `export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas` — the bundled ptxas doesn't handle `sm_121a`.

<a name="silent-cpu-fallback"></a>
**Silent CPU fallback.** Any dependency pinning an older torch (`torch==2.6.0` and friends) finds no matching aarch64+cu130 wheel and quietly resolves to a CPU build — no error, just no GPU. If a third-party package must be installed, use `pip install <pkg> --no-deps` and add its real dependencies by hand, omitting the torch/numpy pins. Re-run the verification snippet after installing anything new.

**Unified memory: OOM is system-wide.** The 128 GB is shared between CPU and GPU — there is no separate VRAM pool. Consequences that matter for training runs here:

- "GPU out of memory" is *system* out of memory. Instead of a clean `RuntimeError: CUDA out of memory`, the machine can go unresponsive — SSH hangs, the job appears alive from outside, and a hard reboot is the only exit.
- `nvidia-smi` reports `Memory-Usage: N/A` on unified systems. Use the DGX Dashboard or `tegrastats` for real telemetry.
- DataLoader workers fork full process state; with a model already resident this exhausts memory fast. Start with `num_workers=0` and raise it only with memory monitored.
- Grow batch size upward from something conservative and record the ceiling in `docs/environment.md`.

**Don't upgrade CUDA on the host.** The DGX Spark ships a long-term-supported stack where OS, driver and toolkit move together. To get newer CUDA, run an NGC container (`nvcr.io/nvidia/pytorch:25.12-py3` or later, validated for DGX Spark) rather than modifying the host.

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

Hardware: NVIDIA GB10 (sm_121), 20-core ARM CPU, 128 GB unified memory. Batch size <!-- N -->, dtype <!-- complex64 / float32 -->.

Two things to get right before quoting any of these numbers:

- **Synchronise.** CuPy kernels launch asynchronously. Time with `cupy.cuda.Device().synchronize()` or `cupyx.profiler.benchmark`, never a bare `time.perf_counter()` around the call — otherwise you're timing the launch, not the work.
- **Unified memory flatters host↔device transfer.** There's no PCIe hop here, so the CPU→GPU copy cost that dominates CuPy-vs-NumPy comparisons on a discrete GPU is largely absent. That makes the speedups real *on this machine* but not transferable — state the platform whenever you quote them, and say so in interviews before someone else points it out.

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
