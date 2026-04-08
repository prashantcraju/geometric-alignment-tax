from __future__ import annotations

"""
Geometric Tax - Perturbation Protocol
===================================================
Standardized perturbation suite applied identically to all models.
Ensures fair comparison between Transformers and SSMs.

Three perturbation types:
  1. Point Mutation Noise (SNP test) -- local stability
  2. Motif Shift/Jitter (Grammar test) -- positional stability
  3. Reverse-Complement Flip (Symmetry test) -- global invariance

Usage:
    from perturbation_protocol import PerturbationSuite

    suite = PerturbationSuite(seed=320)
    perturbed_seqs = suite.run_all(sequences)
    # Returns dict: {"snp_1pct": [...], "snp_5pct": [...], "motif_shift_5bp": [...], ...}
"""

import numpy as np
import os
from typing import Optional
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DNA_BASES = ["A", "C", "G", "T"]
COMPLEMENT = {"A": "T", "T": "A", "C": "G", "G": "C", "N": "N"}

# Common regulatory motifs (IUPAC -> concrete examples)
KNOWN_MOTIFS = {
    "CTCF": "CCGCGNGGNGGCAG",   # CTCF consensus (N = any)
    "TATA": "TATAAAA",           # TATA box
    "SP1":  "GGGCGG",            # SP1 binding site
    "CAAT": "CCAAT",             # CAAT box
    "E-box": "CACGTG",           # E-box (bHLH binding)
}


# ---------------------------------------------------------------------------
# Perturbation result container
# ---------------------------------------------------------------------------

@dataclass
class PerturbedSet:
    """One set of perturbed sequences with metadata."""
    name: str
    category: str           # "snp", "motif_shift", "reverse_complement"
    sequences: list[str]
    params: dict = field(default_factory=dict)
    description: str = ""


# ---------------------------------------------------------------------------
# Core perturbation functions
# ---------------------------------------------------------------------------

def mutate_snp(
    sequence: str,
    mutation_rate: float,
    rng: np.random.Generator,
) -> str:
    """
    Apply random point mutations at the given rate.

    Parameters
    ----------
    sequence : str
        Input DNA sequence (uppercase A/C/G/T).
    mutation_rate : float
        Fraction of bases to mutate (0.0 to 1.0).
    rng : np.random.Generator
        Random number generator.

    Returns
    -------
    str
        Mutated sequence.
    """
    seq = list(sequence.upper())
    n_mutations = max(1, int(len(seq) * mutation_rate))
    positions = rng.choice(len(seq), size=n_mutations, replace=False)

    for pos in positions:
        original = seq[pos]
        if original in DNA_BASES:
            alternatives = [b for b in DNA_BASES if b != original]
            seq[pos] = rng.choice(alternatives)

    return "".join(seq)


def shift_motif(
    sequence: str,
    motif: str,
    shift_bp: int,
    rng: np.random.Generator,
) -> str:
    """
    Find a motif in the sequence and shift it by the specified number of base pairs.
    If the motif is not found, insert it at the center and then shift.

    Parameters
    ----------
    sequence : str
        Input DNA sequence.
    motif : str
        The motif pattern to shift (concrete bases, no IUPAC ambiguity).
    shift_bp : int
        Number of base pairs to shift (positive = right, negative = left).
    rng : np.random.Generator
        Random number generator.

    Returns
    -------
    str
        Sequence with motif shifted.
    """
    seq = list(sequence.upper())
    # Remove IUPAC ambiguity codes for matching
    motif_clean = motif.replace("N", "")
    motif_len = len(motif)

    # Find all occurrences of the motif (fuzzy: skip N positions)
    found_pos = _find_motif(sequence, motif)

    if found_pos:
        # Pick the first occurrence
        pos = found_pos[0]
    else:
        # Insert motif at center
        pos = len(seq) // 2 - motif_len // 2
        for i, base in enumerate(motif):
            if base != "N" and pos + i < len(seq):
                seq[pos + i] = base

    # Now shift: remove motif from old position, insert at new position
    new_pos = pos + shift_bp
    new_pos = max(0, min(new_pos, len(seq) - motif_len))

    # Extract the motif bases
    motif_bases = seq[pos:pos + motif_len]

    # Remove from old position (replace with random bases)
    for i in range(motif_len):
        if pos + i < len(seq):
            seq[pos + i] = rng.choice(DNA_BASES)

    # Insert at new position
    for i, base in enumerate(motif_bases):
        if new_pos + i < len(seq):
            seq[new_pos + i] = base

    return "".join(seq)


def _find_motif(sequence: str, motif: str) -> list[int]:
    """Find all positions where motif matches (N matches anything)."""
    seq = sequence.upper()
    motif = motif.upper()
    positions = []
    for i in range(len(seq) - len(motif) + 1):
        match = True
        for j, m in enumerate(motif):
            if m != "N" and seq[i + j] != m:
                match = False
                break
        if match:
            positions.append(i)
    return positions


def reverse_complement(sequence: str) -> str:
    """
    Return the reverse complement of a DNA sequence.

    Parameters
    ----------
    sequence : str
        Input DNA sequence.

    Returns
    -------
    str
        Reverse complement.
    """
    return "".join(COMPLEMENT.get(b, "N") for b in reversed(sequence.upper()))


def single_point_mutation_walk(
    seq_start: str,
    seq_end: str,
    n_steps: Optional[int] = None,
) -> list[str]:
    """
    Create a mutation walk from seq_start to seq_end, changing one base at a time.
    This is the "natural perturbation path" for the manifold continuity visualization.

    Parameters
    ----------
    seq_start : str
        Starting sequence.
    seq_end : str
        Target sequence.
    n_steps : int, optional
        If None, uses the Hamming distance (one step per differing position).

    Returns
    -------
    list[str]
        List of intermediate sequences from start to end.
    """
    assert len(seq_start) == len(seq_end), "Sequences must be same length"
    start = list(seq_start.upper())
    end = list(seq_end.upper())

    # Find all differing positions
    diff_positions = [i for i in range(len(start)) if start[i] != end[i]]

    if n_steps is not None and n_steps < len(diff_positions):
        # Subsample positions evenly
        indices = np.linspace(0, len(diff_positions) - 1, n_steps, dtype=int)
        diff_positions = [diff_positions[i] for i in indices]

    walk = [seq_start]
    current = start.copy()
    for pos in diff_positions:
        current[pos] = end[pos]
        walk.append("".join(current))

    return walk


# ---------------------------------------------------------------------------
# Perturbation Suite
# ---------------------------------------------------------------------------

class PerturbationSuite:
    """
    Standardized perturbation protocol for the Geometric Tax experiments.

    Parameters
    ----------
    seed : int
        Random seed for reproducibility.
    snp_rates : list[float]
        Mutation rates for the SNP test.
    motif_shifts : list[int]
        Shift amounts (in bp) for the motif jitter test.
    motif_name : str
        Which motif to use for the shift test.
    include_reverse_complement : bool
        Whether to include the RC flip test.
    """

    def __init__(
        self,
        seed: int = 320,
        snp_rates: Optional[list[float]] = None,
        motif_shifts: Optional[list[int]] = None,
        motif_name: str = "CTCF",
        include_reverse_complement: bool = True,
    ):
        self.rng = np.random.default_rng(seed)
        self.snp_rates = snp_rates or [0.01, 0.02, 0.05, 0.10]
        self.motif_shifts = motif_shifts or [5, 10, 20]
        self.motif_name = motif_name
        self.motif_seq = KNOWN_MOTIFS.get(motif_name, "CCGCGNGGNGGCAG")
        self.include_rc = include_reverse_complement

    def run_all(self, sequences: list[str]) -> dict[str, PerturbedSet]:
        """
        Apply all perturbations to the input sequences.

        Parameters
        ----------
        sequences : list[str]
            Input DNA sequences.

        Returns
        -------
        dict[str, PerturbedSet]
            Mapping of perturbation name to PerturbedSet.
        """
        results = {}

        # 1. SNP tests
        for rate in self.snp_rates:
            name = f"snp_{int(rate * 100)}pct"
            perturbed = [mutate_snp(s, rate, self.rng) for s in sequences]
            results[name] = PerturbedSet(
                name=name,
                category="snp",
                sequences=perturbed,
                params={"mutation_rate": rate},
                description=f"Random point mutations at {rate*100:.0f}% rate",
            )

        # 2. Motif shift tests
        for shift in self.motif_shifts:
            for direction in [1, -1]:
                shift_val = shift * direction
                sign = "plus" if direction > 0 else "minus"
                name = f"motif_shift_{sign}{shift}bp"
                perturbed = [
                    shift_motif(s, self.motif_seq, shift_val, self.rng)
                    for s in sequences
                ]
                results[name] = PerturbedSet(
                    name=name,
                    category="motif_shift",
                    sequences=perturbed,
                    params={
                        "motif": self.motif_name,
                        "shift_bp": shift_val,
                    },
                    description=(
                        f"{self.motif_name} motif shifted by {shift_val:+d} bp"
                    ),
                )

        # 3. Reverse complement
        if self.include_rc:
            name = "reverse_complement"
            perturbed = [reverse_complement(s) for s in sequences]
            results[name] = PerturbedSet(
                name=name,
                category="reverse_complement",
                sequences=perturbed,
                params={},
                description="Full reverse complement flip",
            )

        return results

    def get_perturbation_names(self) -> list[str]:
        """Return the names of all perturbations that will be generated."""
        names = []
        for rate in self.snp_rates:
            names.append(f"snp_{int(rate * 100)}pct")
        for shift in self.motif_shifts:
            names.append(f"motif_shift_plus{shift}bp")
            names.append(f"motif_shift_minus{shift}bp")
        if self.include_rc:
            names.append("reverse_complement")
        return names

    def summary(self) -> str:
        """Print a human-readable summary of the protocol."""
        lines = [
            "Perturbation Protocol Summary",
            "=" * 40,
            f"SNP rates: {self.snp_rates}",
            f"Motif: {self.motif_name} ({self.motif_seq})",
            f"Motif shifts: +/- {self.motif_shifts} bp",
            f"Reverse complement: {self.include_rc}",
            f"Total perturbation sets: {len(self.get_perturbation_names())}",
            "",
            "Perturbations:",
        ]
        for name in self.get_perturbation_names():
            lines.append(f"  - {name}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Integration: Full pipeline from sequences to Shesha scores
# ---------------------------------------------------------------------------

def run_full_pipeline(
    model_name: str,
    sequences: list[str],
    embedding_fn,
    harness=None,
    suite: Optional[PerturbationSuite] = None,
    labels: Optional[np.ndarray] = None,
    batch_size: int = 64,
    cache_dir: Optional[str] = None,
):
    """
    End-to-end pipeline: sequences -> perturbations -> embeddings -> Shesha scores.

    Parameters
    ----------
    model_name : str
        Model identifier.
    sequences : list[str]
        Input DNA sequences.
    embedding_fn : callable
        Function that takes list[str] and returns np.ndarray of embeddings.
        Should return shape (n_sequences, embed_dim) or (n_sequences, seq_len, embed_dim).
    harness : StabilityHarness, optional
        Pre-configured harness. If None, uses defaults.
    suite : PerturbationSuite, optional
        Pre-configured perturbation suite. If None, uses defaults.
    labels : np.ndarray, optional
        Class labels for supervised metrics.
    batch_size : int
        Number of sequences to embed per batch. Prevents OOM for large datasets.
    cache_dir : str, optional
        If provided, cache embeddings to disk as .npy files.
        Useful for resuming interrupted runs on 10k+ datasets.

    Returns
    -------
    ModelReport
    """
    # Lazy import to avoid circular dependency
    from evaluation_harness import StabilityHarness, ModelReport

    if harness is None:
        harness = StabilityHarness()
    if suite is None:
        suite = PerturbationSuite()

    print(f"\n{'='*60}")
    print(f"Running full pipeline for: {model_name}")
    print(f"  Sequences: {len(sequences)}")
    print(f"  Batch size: {batch_size}")
    print(f"{'='*60}")
    print(suite.summary())

    def _batched_embed(seqs: list[str], name: str) -> np.ndarray:
        """Compute embeddings in batches, with optional disk caching."""
        cache_path = None
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            safe_name = name.replace("/", "_").replace(" ", "_")
            cache_path = os.path.join(cache_dir, f"{model_name}_{safe_name}.npy")
            if os.path.exists(cache_path):
                print(f"  Loading cached embeddings: {cache_path}")
                return np.load(cache_path)

        n_total = len(seqs)
        all_embeddings = []
        for i in range(0, n_total, batch_size):
            batch = seqs[i:i + batch_size]
            emb = embedding_fn(batch)
            all_embeddings.append(emb)
            done = min(i + batch_size, n_total)
            print(f"    {name}: {done}/{n_total} sequences embedded", end="\r")
        print()  # newline after progress

        result = np.concatenate(all_embeddings, axis=0)

        if cache_path:
            np.save(cache_path, result)
            print(f"  Cached to: {cache_path}")

        return result

    # Get clean embeddings
    print(f"\nComputing clean embeddings...")
    embeddings_clean = _batched_embed(sequences, "clean")
    print(f"  Shape: {embeddings_clean.shape}")

    # Generate all perturbations
    print("\nGenerating perturbations...")
    perturbed_sets = suite.run_all(sequences)

    # Compute perturbed embeddings and evaluate
    perturbed_embeddings = {}
    for pert_name, pert_set in perturbed_sets.items():
        print(f"\nComputing embeddings for {pert_name}...")
        perturbed_embeddings[pert_name] = _batched_embed(
            pert_set.sequences, pert_name
        )

    # Run evaluation
    print("\nRunning Shesha evaluation...")
    report = harness.evaluate_all_perturbations(
        model_name=model_name,
        embeddings_clean=embeddings_clean,
        perturbed_dict=perturbed_embeddings,
        labels=labels,
    )

    return report


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Perturbation Protocol - Smoke Test ===\n")

    rng = np.random.default_rng(320)

    # Generate some random DNA sequences
    n_seqs = 5
    seq_len = 200
    sequences = [
        "".join(rng.choice(DNA_BASES, size=seq_len))
        for _ in range(n_seqs)
    ]

    print(f"Generated {n_seqs} random sequences of length {seq_len}")
    print(f"Sample: {sequences[0][:50]}...\n")

    # Run suite
    suite = PerturbationSuite(seed=320)
    print(suite.summary())
    print()

    perturbed = suite.run_all(sequences)

    for name, pset in perturbed.items():
        # Calculate Hamming distance from original
        distances = []
        for orig, pert in zip(sequences, pset.sequences):
            d = sum(a != b for a, b in zip(orig, pert)) / len(orig)
            distances.append(d)
        mean_dist = np.mean(distances)
        print(f"{name:<30} mean_hamming_dist={mean_dist:.4f}")

    # Test mutation walk
    print("\n=== Mutation Walk Test ===")
    seq_a = "".join(rng.choice(DNA_BASES, size=50))
    seq_b = "".join(rng.choice(DNA_BASES, size=50))
    walk = single_point_mutation_walk(seq_a, seq_b, n_steps=10)
    print(f"Walk from A to B: {len(walk)} steps")
    for i, s in enumerate(walk):
        hamming = sum(a != b for a, b in zip(seq_a, s)) / len(seq_a)
        print(f"  Step {i:2d}: hamming_from_start={hamming:.3f}  ...{s[:20]}")

    # Test reverse complement
    print("\n=== Reverse Complement Test ===")
    test_seq = "ATCGATCG"
    rc = reverse_complement(test_seq)
    print(f"Original: {test_seq}")
    print(f"RevComp:  {rc}")
    assert rc == "CGATCGAT", f"Expected CGATCGAT, got {rc}"
    # Double RC should recover original
    assert reverse_complement(rc) == test_seq
    print("RC round-trip: PASSED")

    print("\nSmoke test passed.")