# Geometric Alignment Tax
<p align="center">
    <a style="text-decoration:none !important;" href="https://arxiv.org/abs/2604.04155" alt="arXiv"><img src="https://img.shields.io/badge/paper-arXiv-blue" /></a>
    <a style="text-decoration:none !important;" href="https://huggingface.co/papers/2604.04155" alt="Hugging Face Papers"><img src="https://img.shields.io/badge/paper-Hugging%20Face-FFD21E?logo=huggingface&logoColor=black" /></a>
</p>

Code accompanying the paper **"The Geometric Alignment Tax: Tokenization vs. Continuous Geometry in Scientific Foundation Models"**.

## Overview

This repository contains all experiments measuring **geometric stability** of biological sequence foundation models under standardized perturbations.

## Repository Structure

```
geometric-alignment-tax/
├── utils/
│   ├── evaluation_harness.py      # StabilityHarness: Shesha-based stability metrics
│   └── perturbation_protocol.py   # PerturbationSuite: standardized sequence perturbations
├── Section 2/                     # Mechanisms & synthetic controls
│   ├── Track A+B/                 # Track A (physics interpolation) & Track B (BRCA1 walk)
│   ├── Architectural Control/     # SmallBERT vs SmallMamba on synthetic dynamics
│   ├── Coupled Lorenz/            # d_M scaling / Procrustes rate-distortion
│   ├── Learned Quantization/      # Continuous vs VQ vs categorical head ablation
│   └── Ablation/                  # Variants A–F (head type, Jacobian, LSTM, Hyena sweeps)
├── Section 3/                     # Foundation model experiments
│   ├── Models/                    # Per-model scaling + real-data experiments
│   │   ├── ESM-2/                 # Protein: ESM-2 (8M–15B)
│   │   ├── OpenFold/              # Protein: OpenFold Evoformer
│   │   ├── SaProt/                # Protein: SaProt
│   │   ├── DNABERT:2/             # DNA: DNABERT (k-mer) + DNABERT-2
│   │   ├── NT/                    # DNA: Nucleotide Transformer (50M–2.5B)
│   │   ├── Caduceus/              # DNA: Caduceus (SSM, RC-equivariant)
│   │   ├── HyenaDNA/              # DNA: HyenaDNA (long-convolution)
│   │   ├── GPN/                   # DNA: GPN (convolutional)
│   │   └── Evo2/                  # DNA/Genomic: Evo 2 (7B–40B)
│   ├── Ghost Detection/           # Spin test + frozen-head context tax
│   ├── RCCR/                      # Reverse-complement consistency regularization
│   ├── Texture/                   # Evo 2 texture vs structure hypothesis
│   ├── Nonlinear Probes/          # Linear vs MLP probe validation
│   └── Phase Transition/          # Brittle glass vs untethered gel classification
└── Section 4/mine/                # MINE mutual information analysis (three regimes)
```

## Setup

All notebooks are designed to run on **Google Colab** with a GPU runtime. Upload `utils/evaluation_harness.py` and `utils/perturbation_protocol.py` to `/content/` before running any experiment notebook.

```bash
# Core dependencies (installed automatically by each notebook)
pip install torch transformers shesha-geometry matplotlib seaborn pandas scipy
```

For Mamba/SSM experiments:
```bash
pip install causal-conv1d mamba-ssm --no-build-isolation
```

For Evo 2 experiments:
```bash
pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
pip install flash-attn==2.8.0.post2 --no-build-isolation
pip install evo2
```

## Key Utilities

### `StabilityHarness` (`utils/evaluation_harness.py`)

Wraps the [Shesha](https://pypi.org/project/shesha-geometry/) geometry library to compute:
- Feature split / sample split scores
- Anchor stability
- RDM similarity & drift (clean vs. perturbed)
- Perturbation stability & effect size
- Composite stability score

### `PerturbationSuite` (`utils/perturbation_protocol.py`)

Standardized perturbations applied identically across all models:
- **SNP test**: random point mutations at 1%, 2%, 5%, 10% rates
- **Motif shift/jitter**: CTCF/TATA/SP1 motif shifted ±5/10/20 bp
- **Reverse-complement flip**: full RC transformation

## Reproducibility

All experiments use `seed=320`. Results are saved locally as `.npy` caches, `.csv` summaries, and `.png` figures under `./results/<experiment_name>/` relative to where the notebook is run. No Google Drive or cloud storage required — outputs go wherever the notebook's working directory is.



## Citation

If you use this code, please cite:

```bibtex
@article{raju2026tax,
  title   = {{The Geometric Alignment Tax: Tokenization vs. Continuous Geometry in Scientific Foundation Models}},
  author  = {Raju, Prashant C.},
  journal = {arXiv preprint arXiv:2604.04155},
  year    = {2026},
}
```
