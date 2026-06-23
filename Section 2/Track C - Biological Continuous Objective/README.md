# Track C — Biological Continuous Objective (ESM-2 / Evo 2 analogs)

The biological counterpart of **synthetic Variant A**. Variant A showed that swapping
a categorical cross-entropy (CE) head for a continuous MSE head — *same backbone,
same discrete inputs, only the output objective changes* — reduces geometric
distortion. Track C carries that controlled, causal comparison onto **real protein
and DNA sequence models**.

## The one hard part

A biological token (a residue, a nucleotide) has no underlying continuous quantity
the way an oscillator sample does. So the continuous condition cannot regress to "the
raw value." It regresses to a **fixed, multi-dimensional, well-separated continuous
code per token**. Inputs stay discrete (identical to CE); only the head + loss change.
Discrete predictions for evaluation come from **nearest-prototype lookup**, giving a
recovery accuracy directly comparable to CE's argmax.

Two code flavours are used together to make the claim causal:

- **`physchem`** — standardized physicochemical / structural property vectors
  (biologically meaningful, well-separated, non-monotone in token index).
- **`random`** — a fixed random orthonormal code (biologically arbitrary but
  identity-preserving). The control: if it *also* beats CE on geometry, the effect is
  the **form of the objective** (continuous regression, no softmax, no hard decision
  boundary), not biology smuggled in through the target. This closes the CE→MSE
  confound directly.

## Notebooks

| Notebook | Analog | Backbone | Objective | Data | Extra |
|---|---|---|---|---|---|
| `Track_C_Protein_Continuous.ipynb` | ESM-2 | `SmallBERT` (bidirectional) | masked-LM | UniRef50/SwissProt subset | SCOP-class probe |
| `Track_C_DNA_Continuous.ipynb` | Evo 2 | `SmallStripedHyena` (causal hybrid) | next-token | human chr22 windows | **reverse-complement** test + NT-benchmark probe |

Each notebook runs three conditions — **CE**, **Cont-physchem**, **Cont-random** —
matched on backbone, data, optimizer, schedule, precision, and `seed=320`.

## What is measured

1. **Geometry** (Shesha `StabilityHarness` + Procrustes distortion **D**) under a
   biological perturbation suite (point substitution at 1/2/5/10%, sequence reversal;
   DNA adds reverse-complement). *Prediction:* continuous arms show lower **D** and
   higher RDM similarity than CE.
2. **Task performance on frozen representations:** intrinsic recovery accuracy
   (argmax vs nearest-prototype) **and** a linear probe on frozen embeddings for a
   real downstream task. *Bar to clear:* comparable accuracy **+** better geometry.
3. **Matched-performance check:** the geometry gap persists even at checkpoints where
   CE and Cont reach the same recovery accuracy (closes the optimization-artifact
   objection).

## Shared utility

`utils/bio_continuous_codes.py` — the genuinely new, reusable logic:
`build_protein_codes` / `build_dna_codes` (physchem + random codebooks),
`nearest_prototype_decode`, `recovery_accuracy`, `procrustes_distortion`,
`rdm_similarity`. Pure NumPy/SciPy; run `python utils/bio_continuous_codes.py` for the
smoke test.

## Setup

Colab + GPU. Upload `utils/evaluation_harness.py`, `utils/perturbation_protocol.py`,
and `utils/bio_continuous_codes.py` (or keep the notebooks under
`geometric-alignment-tax/`, which the `sys.path` shims already cover). Set `PHASE='quick'`
for a fast smoke run, `PHASE='full'` for the headline result. Outputs land under
`results/track_c_{protein,dna}_continuous/`.
