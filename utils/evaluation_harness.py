
"""
Geometric Tax - Evaluation Harness
===============================================
Reusable pipeline that takes any model's embeddings, runs the full Shesha
stability suite, and returns structured results.

Usage:
    from evaluation_harness import StabilityHarness

    harness = StabilityHarness(window_size=2000)
    results = harness.evaluate(
        model_name="AlphaGenome",
        embeddings_clean=X_clean,        # (n_sequences, seq_len, embed_dim)
        embeddings_perturbed=X_perturbed, # same shape, after perturbation
        perturbation_name="snp_1pct",
        labels=y,                         # optional class labels
    )
"""

import numpy as np
import json
import os
from dataclasses import dataclass, field, asdict
from typing import Optional, Literal
from datetime import datetime

import shesha
from shesha import (
    feature_split,
    sample_split,
    anchor_stability,
    variance_ratio,
    supervised_alignment,
    rdm_similarity,
    rdm_drift,
    compute_rdm,
)
from shesha.bio import perturbation_stability, perturbation_effect_size


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class StabilityResult:
    """Results for a single (model, perturbation) evaluation."""
    model_name: str
    perturbation_name: str
    timestamp: str = ""

    # Core Shesha metrics (on clean embeddings)
    feature_split_score: float = 0.0
    sample_split_score: float = 0.0
    anchor_stability_score: float = 0.0

    # Supervised metrics (if labels provided)
    variance_ratio_score: Optional[float] = None
    supervised_alignment_score: Optional[float] = None

    # Perturbation-specific metrics (clean vs. perturbed)
    rdm_similarity_score: float = 0.0
    rdm_drift_score: float = 0.0
    perturbation_stability_score: float = 0.0
    perturbation_magnitude: float = 0.0

    # Composite
    composite_stability: float = 0.0

    # Metadata
    n_sequences: int = 0
    embed_dim: int = 0
    window_size: int = 0

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ModelReport:
    """Aggregated results across all perturbations for one model."""
    model_name: str
    results: list[StabilityResult] = field(default_factory=list)

    def summary(self) -> dict:
        """Aggregate mean scores across perturbations."""
        if not self.results:
            return {}
        keys = [
            "feature_split_score", "sample_split_score",
            "anchor_stability_score", "rdm_similarity_score",
            "rdm_drift_score", "perturbation_stability_score",
            "perturbation_magnitude", "composite_stability",
        ]
        summary = {"model_name": self.model_name, "n_perturbations": len(self.results)}
        for k in keys:
            vals = [getattr(r, k) for r in self.results]
            summary[f"mean_{k}"] = float(np.mean(vals))
            summary[f"std_{k}"] = float(np.std(vals))
        return summary

    def perturbation_breakdown(self) -> dict:
        """Get per-perturbation results."""
        return {r.perturbation_name: r.to_dict() for r in self.results}


# ---------------------------------------------------------------------------
# Main harness
# ---------------------------------------------------------------------------

class StabilityHarness:
    """
    Standardized evaluation pipeline for the Geometric Tax experiments.

    Parameters
    ----------
    window_size : int
        Center window (in tokens/bp) to extract from embeddings.
        Neutralizes context-length differences between architectures.
        Set to 0 to use full embeddings.
    metric : str
        Distance metric for RDM computation ('cosine' or 'correlation').
    n_splits : int
        Number of random splits for feature_split / sample_split.
    seed : int
        Random seed for reproducibility.
    max_samples : int
        Max samples for Shesha computation (memory management).
        For 10k+ datasets, Shesha internally subsamples to this limit
        for pairwise RDM computation. Higher values = more coverage
        but O(n^2) memory. 2500 is a good balance for 10k datasets.
    n_bootstrap : int
        Number of bootstrap rounds for confidence intervals.
        Set to 0 to disable bootstrapping (faster, no CIs).
    """

    def __init__(
        self,
        window_size: int = 2000,
        metric: Literal["cosine", "correlation"] = "cosine",
        n_splits: int = 30,
        seed: int = 320,
        max_samples: int = 2500,
        n_bootstrap: int = 0,
    ):
        self.window_size = window_size
        self.metric = metric
        self.n_splits = n_splits
        self.seed = seed
        self.max_samples = max_samples
        self.n_bootstrap = n_bootstrap

    def _extract_center_window(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Extract the center window from sequence embeddings.

        Parameters
        ----------
        embeddings : np.ndarray
            Shape (n_sequences, seq_len, embed_dim) or (n_sequences, embed_dim).

        Returns
        -------
        np.ndarray
            Shape (n_sequences, window_size * embed_dim) -- flattened for Shesha,
            or (n_sequences, embed_dim) if already 2D.
        """
        if embeddings.ndim == 2:
            # Already pooled -- (n_sequences, embed_dim)
            return embeddings

        n_seq, seq_len, embed_dim = embeddings.shape

        if self.window_size <= 0 or self.window_size >= seq_len:
            # Use full sequence, pool by mean
            return embeddings.mean(axis=1)

        # Extract center window
        center = seq_len // 2
        half_w = self.window_size // 2
        start = max(0, center - half_w)
        end = min(seq_len, center + half_w)

        windowed = embeddings[:, start:end, :]  # (n_seq, window, embed_dim)

        # Mean-pool over the window to get (n_seq, embed_dim)
        # This preserves comparability across models with different embed dims
        return windowed.mean(axis=1)

    def _subsample_stratified(
        self,
        X_clean: np.ndarray,
        X_pert: np.ndarray,
        labels: Optional[np.ndarray],
        n_samples: int,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Stratified subsampling for large datasets.
        If labels are provided, subsample proportionally from each class.
        Otherwise, uniform random subsampling.
        """
        n_total = X_clean.shape[0]
        if n_total <= n_samples:
            return X_clean, X_pert, labels

        if labels is not None:
            # Stratified: proportional sampling from each class
            unique_labels = np.unique(labels)
            indices = []
            for lab in unique_labels:
                lab_idx = np.where(labels == lab)[0]
                n_from_class = max(1, int(n_samples * len(lab_idx) / n_total))
                chosen = rng.choice(lab_idx, size=min(n_from_class, len(lab_idx)), replace=False)
                indices.extend(chosen)
            indices = np.array(indices[:n_samples])
        else:
            indices = rng.choice(n_total, size=n_samples, replace=False)

        sub_labels = labels[indices] if labels is not None else None
        return X_clean[indices], X_pert[indices], sub_labels

    @staticmethod
    def _safe_metric(fn, *args, **kwargs):
        """Call a shesha metric function safely, returning NaN on failure."""
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                val = fn(*args, **kwargs)
                result = float(val)
                if np.isnan(result) or np.isinf(result):
                    return float('nan')
                return result
            except Exception:
                return float('nan')

    def _compute_single_run(
        self,
        X_clean: np.ndarray,
        X_pert: np.ndarray,
        labels: Optional[np.ndarray],
    ) -> dict:
        """Run all Shesha metrics once on a (possibly subsampled) set."""
        n_sequences = X_clean.shape[0]

        fs_score = self._safe_metric(
            feature_split,
            X_clean,
            n_splits=self.n_splits,
            metric=self.metric,
            seed=self.seed,
            max_samples=self.max_samples,
        )
        ss_score = self._safe_metric(
            sample_split,
            X_clean,
            n_splits=self.n_splits,
            metric=self.metric,
            seed=self.seed,
            max_samples=self.max_samples,
        )
        anc_score = self._safe_metric(
            anchor_stability,
            X_clean,
            n_splits=self.n_splits,
            metric="cosine",
            seed=self.seed,
            max_samples=self.max_samples,
        )

        vr_score = None
        sa_score = None
        if labels is not None:
            vr_score = self._safe_metric(variance_ratio, X_clean, labels)
            sa_score = self._safe_metric(
                supervised_alignment,
                X_clean, labels,
                metric="correlation",
                seed=self.seed,
                max_samples=min(300, n_sequences),
            )

        rdm_sim = self._safe_metric(rdm_similarity, X_clean, X_pert, metric=self.metric)
        rdm_d = self._safe_metric(rdm_drift, X_clean, X_pert, metric=self.metric)

        pert_stab = self._safe_metric(
            perturbation_stability,
            X_clean, X_pert,
            metric="cosine",
            seed=self.seed,
            max_samples=min(1000, n_sequences),
        )
        pert_mag = self._safe_metric(perturbation_effect_size, X_clean, X_pert, metric="euclidean")


        return {
            "feature_split_score": fs_score,
            "sample_split_score": ss_score,
            "anchor_stability_score": anc_score,
            "variance_ratio_score": vr_score,
            "supervised_alignment_score": sa_score,
            "rdm_similarity_score": rdm_sim,
            "rdm_drift_score": rdm_d,
            "perturbation_stability_score": pert_stab,
            "perturbation_magnitude": pert_mag,
        }

    def evaluate(
        self,
        model_name: str,
        embeddings_clean: np.ndarray,
        embeddings_perturbed: np.ndarray,
        perturbation_name: str,
        labels: Optional[np.ndarray] = None,
    ) -> StabilityResult:
        """
        Run the full Shesha stability suite for one (model, perturbation) pair.

        For datasets > max_samples, stratified subsampling is applied.
        If n_bootstrap > 0, runs multiple bootstrap rounds and reports
        mean +/- std for each metric.

        Parameters
        ----------
        model_name : str
            Name of the model (e.g., "AlphaGenome", "Caduceus").
        embeddings_clean : np.ndarray
            Clean embeddings. Shape (n_seq, seq_len, embed_dim) or (n_seq, embed_dim).
        embeddings_perturbed : np.ndarray
            Perturbed embeddings. Same shape as clean.
        perturbation_name : str
            Label for the perturbation (e.g., "snp_1pct", "motif_shift_5bp").
        labels : np.ndarray, optional
            Class labels for supervised metrics.

        Returns
        -------
        StabilityResult
        """
        # 1. Window extraction
        X_clean = self._extract_center_window(embeddings_clean)
        X_pert = self._extract_center_window(embeddings_perturbed)

        # Filter zero-norm or NaN embeddings before passing to Shesha (guards against Mamba padding artifacts)
        norms_c = np.linalg.norm(X_clean.reshape(X_clean.shape[0], -1), axis=1)
        norms_p = np.linalg.norm(X_pert.reshape(X_pert.shape[0], -1), axis=1)

        valid_c = (norms_c > 1e-6) & ~np.isnan(norms_c)
        valid_p = (norms_p > 1e-6) & ~np.isnan(norms_p)
        nan_mask_c = np.isnan(X_clean).reshape(X_clean.shape[0], -1).any(axis=1)
        nan_mask_p = np.isnan(X_pert).reshape(X_pert.shape[0], -1).any(axis=1)
        
        valid_idx = np.where(valid_c & valid_p & ~nan_mask_c & ~nan_mask_p)[0]
        
        if len(valid_idx) < len(X_clean):
            X_clean = X_clean[valid_idx]
            X_pert = X_pert[valid_idx]
            if labels is not None:
                labels = labels[valid_idx]

        n_sequences, embed_dim = X_clean.shape
        rng = np.random.default_rng(self.seed)

        if self.n_bootstrap > 0:
            # Bootstrap: run multiple rounds with different subsamples
            all_runs = []
            for i in range(self.n_bootstrap):
                boot_rng = np.random.default_rng(self.seed + i)
                Xc, Xp, lab = self._subsample_stratified(
                    X_clean, X_pert, labels, self.max_samples, boot_rng
                )
                run = self._compute_single_run(Xc, Xp, lab)
                all_runs.append(run)

            # Aggregate: mean across bootstrap rounds
            metrics = {}
            for key in all_runs[0]:
                vals = [r[key] for r in all_runs if r[key] is not None]
                metrics[key] = float(np.mean(vals)) if vals else None
        else:
            # Single run (subsample if needed)
            Xc, Xp, lab = self._subsample_stratified(
                X_clean, X_pert, labels, self.max_samples, rng
            )
            metrics = self._compute_single_run(Xc, Xp, lab)

        # Composite stability
        core_metrics = [
            metrics["feature_split_score"],
            metrics["sample_split_score"],
            metrics["anchor_stability_score"],
            metrics["rdm_similarity_score"],
        ]
        valid_metrics = [m for m in core_metrics if m is not None and not np.isnan(m)]
        if len(valid_metrics) == 0:
            composite = float('nan')
        else:
            composite = float(np.mean(valid_metrics))

        return StabilityResult(
            model_name=model_name,
            perturbation_name=perturbation_name,
            feature_split_score=metrics["feature_split_score"],
            sample_split_score=metrics["sample_split_score"],
            anchor_stability_score=metrics["anchor_stability_score"],
            variance_ratio_score=metrics["variance_ratio_score"],
            supervised_alignment_score=metrics["supervised_alignment_score"],
            rdm_similarity_score=metrics["rdm_similarity_score"],
            rdm_drift_score=metrics["rdm_drift_score"],
            perturbation_stability_score=metrics["perturbation_stability_score"],
            perturbation_magnitude=metrics["perturbation_magnitude"],
            composite_stability=composite,
            n_sequences=n_sequences,
            embed_dim=embed_dim,
            window_size=self.window_size,
        )

    def evaluate_all_perturbations(
        self,
        model_name: str,
        embeddings_clean: np.ndarray,
        perturbed_dict: dict[str, np.ndarray],
        labels: Optional[np.ndarray] = None,
    ) -> ModelReport:
        """
        Run evaluation across all perturbations for a single model.

        Parameters
        ----------
        model_name : str
            Model identifier.
        embeddings_clean : np.ndarray
            Clean embeddings.
        perturbed_dict : dict[str, np.ndarray]
            Mapping of perturbation_name -> perturbed embeddings.
        labels : np.ndarray, optional
            Class labels.

        Returns
        -------
        ModelReport
        """
        report = ModelReport(model_name=model_name)
        for pert_name, X_pert in perturbed_dict.items():
            result = self.evaluate(
                model_name=model_name,
                embeddings_clean=embeddings_clean,
                embeddings_perturbed=X_pert,
                perturbation_name=pert_name,
                labels=labels,
            )
            report.results.append(result)
            print(f"  [{model_name}] {pert_name}: composite={result.composite_stability:.4f}")
        return report


# ---------------------------------------------------------------------------
# Comparison utilities
# ---------------------------------------------------------------------------

def compare_models(reports: list[ModelReport], output_dir: str = "results") -> dict:
    """
    Compare multiple models and save results.

    Parameters
    ----------
    reports : list[ModelReport]
        One report per model.
    output_dir : str
        Directory to save JSON results.

    Returns
    -------
    dict
        Combined summary for all models.
    """
    os.makedirs(output_dir, exist_ok=True)

    comparison = {
        "timestamp": datetime.now().isoformat(),
        "models": {},
    }

    for report in reports:
        comparison["models"][report.model_name] = {
            "summary": report.summary(),
            "perturbations": report.perturbation_breakdown(),
        }

    # Save
    out_path = os.path.join(output_dir, "model_comparison.json")
    with open(out_path, "w") as f:
        json.dump(comparison, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

    return comparison


def compute_lipschitz_profile(
    embeddings_sequence: np.ndarray,
    epsilon: float = 1e-8,
) -> np.ndarray:
    """
    Compute local Lipschitz constants along an interpolation path.
    This is for the "Lipschitz Spike" plot (Panel C of the manifold visualization).

    Parameters
    ----------
    embeddings_sequence : np.ndarray
        Shape (n_steps, embed_dim) -- embeddings at each interpolation step.
    epsilon : float
        Small constant to avoid division by zero.

    Returns
    -------
    np.ndarray
        Shape (n_steps - 1,) -- local Lipschitz constant at each step.
    """
    diffs = np.diff(embeddings_sequence, axis=0)
    norms = np.linalg.norm(diffs, axis=1)
    # Each step is 1/(n_steps-1) apart in alpha space
    step_size = 1.0 / (len(embeddings_sequence) - 1)
    lipschitz = norms / (step_size + epsilon)
    return lipschitz


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Shesha Evaluation Harness - Smoke Test ===\n")

    rng = np.random.default_rng(320)
    n_seq, embed_dim = 200, 256

    # Simulate "stable" model (SSM-like): small perturbation = small shift
    clean = rng.standard_normal((n_seq, embed_dim))
    perturbed_stable = clean + rng.standard_normal((n_seq, embed_dim)) * 0.05

    # Simulate "unstable" model (Transformer-like): perturbation causes big jumps
    perturbed_unstable = clean + rng.standard_normal((n_seq, embed_dim)) * 0.8

    harness = StabilityHarness(window_size=0, seed=320)

    print("Evaluating 'Stable Model' (SSM-like)...")
    r_stable = harness.evaluate(
        model_name="SSM_mock",
        embeddings_clean=clean,
        embeddings_perturbed=perturbed_stable,
        perturbation_name="snp_1pct",
    )

    print("Evaluating 'Unstable Model' (Transformer-like)...")
    r_unstable = harness.evaluate(
        model_name="Transformer_mock",
        embeddings_clean=clean,
        embeddings_perturbed=perturbed_unstable,
        perturbation_name="snp_1pct",
    )

    print(f"\n{'Metric':<35} {'SSM_mock':>12} {'Transformer':>12}")
    print("-" * 60)
    for field_name in [
        "feature_split_score", "sample_split_score", "anchor_stability_score",
        "rdm_similarity_score", "rdm_drift_score",
        "perturbation_stability_score", "perturbation_magnitude",
        "composite_stability",
    ]:
        v1 = getattr(r_stable, field_name)
        v2 = getattr(r_unstable, field_name)
        print(f"{field_name:<35} {v1:>12.4f} {v2:>12.4f}")

    # Lipschitz profile demo
    print("\n=== Lipschitz Profile Demo ===")
    n_steps = 50
    alpha = np.linspace(0, 1, n_steps)

    # Smooth path (SSM)
    start, end = rng.standard_normal(embed_dim), rng.standard_normal(embed_dim)
    smooth_path = np.array([start * (1 - a) + end * a for a in alpha])
    lip_smooth = compute_lipschitz_profile(smooth_path)

    # Jagged path (Transformer) -- stays at start then jumps to end
    jagged_path = np.array([
        start if a < 0.45 else (end if a > 0.55 else start * (1 - (a - 0.45) / 0.1) + end * ((a - 0.45) / 0.1))
        for a in alpha
    ])
    lip_jagged = compute_lipschitz_profile(jagged_path)

    print(f"Smooth path  - Lipschitz max: {lip_smooth.max():.2f}, std: {lip_smooth.std():.4f}")
    print(f"Jagged path  - Lipschitz max: {lip_jagged.max():.2f}, std: {lip_jagged.std():.4f}")
    print(f"Spike ratio (jagged/smooth max): {lip_jagged.max() / lip_smooth.max():.1f}x")

    print("\nSmoke test passed.")