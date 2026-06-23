"""
Geometric Tax - Biological Continuous Target Codes
==================================================
The "one hard part" of the biological Variant-A experiment.

A protein residue or nucleotide has no underlying continuous quantity the way an
oscillator sample does. There is no "raw float" a masked alanine is a rounded
version of. So the continuous-objective condition has to regress toward a
*fixed, multi-dimensional, well-separated continuous code per token*. Inputs stay
discrete (identical to the categorical condition); only the output objective swaps
from softmax-CE-over-vocab to MSE-toward-the-code. Discrete predictions for
evaluation come from nearest-prototype lookup.

This module provides three things:

  1. ``build_protein_codes`` / ``build_dna_codes`` -- the fixed code tables, in two
     flavours that together close the CE-vs-MSE confound:
       * ``physchem``: standardized physicochemical property vectors (biologically
         meaningful, well-separated, NON-monotone in token index).
       * ``random``:   a fixed random *orthonormal* code per token -- biologically
         arbitrary but identity-preserving. If BOTH flavours reduce distortion
         relative to CE, the effect comes from the *form* of the objective
         (continuous regression, no softmax, no hard decision boundary) and not
         from biological information smuggled in through the target.

  2. ``nearest_prototype_decode`` -- turns continuous predictions into discrete
     token ids by nearest-prototype lookup, giving a masked-token recovery
     accuracy directly comparable to the CE model's argmax accuracy.

  3. ``procrustes_distortion`` -- the geometric distortion D between clean and
     perturbed representations (lower = more stable), the headline geometry metric
     mirroring synthetic Variant A.

Everything here is pure NumPy/SciPy (no torch) so it can be unit-tested without a
GPU and imported identically by the protein and DNA notebooks.

Reproducibility: every randomized construction is seeded; the project-wide seed is
320.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Sequence

# ---------------------------------------------------------------------------
# Alphabets
# ---------------------------------------------------------------------------

# Canonical 20 amino acids (ESM-2 / SwissProt order is arbitrary; we fix one).
AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")
# Canonical 4 nucleotides.
NUCLEOTIDES = list("ACGT")


# ---------------------------------------------------------------------------
# Amino-acid physicochemical property table
# ---------------------------------------------------------------------------
# Each entry is a well-known biophysical scalar. They are intentionally
# heterogeneous (hydropathy, size, charge, polarity, pI, mass, aromaticity,
# H-bonding) so that the resulting code is high-rank and the 20 residues are
# well separated. Crucially, none of these is monotone in the (arbitrary)
# token-index order, which is what prevents the "staircase" leak that a trivial
# index-rescaling target would create.

# Kyte-Doolittle hydropathy index.
_AA_HYDROPATHY = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "Q": -3.5, "E": -3.5,
    "G": -0.4, "H": -3.2, "I": 4.5, "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8,
    "P": -1.6, "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}
# Side-chain volume (Zamyatnin 1972), cubic angstroms.
_AA_VOLUME = {
    "A": 88.6, "R": 173.4, "N": 114.1, "D": 111.1, "C": 108.5, "Q": 143.8,
    "E": 138.4, "G": 60.1, "H": 153.2, "I": 166.7, "L": 166.7, "K": 168.6,
    "M": 162.9, "F": 189.9, "P": 112.7, "S": 89.0, "T": 116.1, "W": 227.8,
    "Y": 193.6, "V": 140.0,
}
# Net side-chain charge at pH 7 (His weakly positive).
_AA_CHARGE = {
    "A": 0.0, "R": 1.0, "N": 0.0, "D": -1.0, "C": 0.0, "Q": 0.0, "E": -1.0,
    "G": 0.0, "H": 0.1, "I": 0.0, "L": 0.0, "K": 1.0, "M": 0.0, "F": 0.0,
    "P": 0.0, "S": 0.0, "T": 0.0, "W": 0.0, "Y": 0.0, "V": 0.0,
}
# Grantham polarity.
_AA_POLARITY = {
    "A": 8.1, "R": 10.5, "N": 11.6, "D": 13.0, "C": 5.5, "Q": 10.5, "E": 12.3,
    "G": 9.0, "H": 10.4, "I": 5.2, "L": 4.9, "K": 11.3, "M": 5.7, "F": 5.2,
    "P": 8.0, "S": 9.2, "T": 8.6, "W": 5.4, "Y": 6.2, "V": 5.9,
}
# Isoelectric point (pI) of the free amino acid.
_AA_PI = {
    "A": 6.00, "R": 10.76, "N": 5.41, "D": 2.77, "C": 5.07, "Q": 5.65,
    "E": 3.22, "G": 5.97, "H": 7.59, "I": 6.02, "L": 5.98, "K": 9.74,
    "M": 5.74, "F": 5.48, "P": 6.30, "S": 5.68, "T": 5.60, "W": 5.89,
    "Y": 5.66, "V": 5.96,
}
# Residue molecular weight (Da).
_AA_WEIGHT = {
    "A": 89.09, "R": 174.20, "N": 132.12, "D": 133.10, "C": 121.16,
    "Q": 146.15, "E": 147.13, "G": 75.07, "H": 155.16, "I": 131.17,
    "L": 131.17, "K": 146.19, "M": 149.21, "F": 165.19, "P": 115.13,
    "S": 105.09, "T": 119.12, "W": 204.23, "Y": 181.19, "V": 117.15,
}
# Aromaticity (F/W/Y aromatic; H weakly so).
_AA_AROMATIC = {
    "A": 0.0, "R": 0.0, "N": 0.0, "D": 0.0, "C": 0.0, "Q": 0.0, "E": 0.0,
    "G": 0.0, "H": 0.5, "I": 0.0, "L": 0.0, "K": 0.0, "M": 0.0, "F": 1.0,
    "P": 0.0, "S": 0.0, "T": 0.0, "W": 1.0, "Y": 1.0, "V": 0.0,
}
# Side-chain hydrogen-bond donor+acceptor count (approximate).
_AA_HBOND = {
    "A": 0, "R": 5, "N": 4, "D": 4, "C": 1, "Q": 4, "E": 4, "G": 0, "H": 3,
    "I": 0, "L": 0, "K": 2, "M": 0, "F": 0, "P": 0, "S": 2, "T": 2, "W": 1,
    "Y": 2, "V": 0,
}

_AA_PROPERTY_TABLES = [
    ("hydropathy", _AA_HYDROPATHY),
    ("volume", _AA_VOLUME),
    ("charge", _AA_CHARGE),
    ("polarity", _AA_POLARITY),
    ("isoelectric_point", _AA_PI),
    ("weight", _AA_WEIGHT),
    ("aromaticity", _AA_AROMATIC),
    ("hbond", _AA_HBOND),
]


# ---------------------------------------------------------------------------
# Nucleotide structural / physicochemical features
# ---------------------------------------------------------------------------
# Three classic, mutually-orthogonal biochemical axes uniquely identify each
# base and are non-monotone in (A,C,G,T) index order:
#   purine        : A,G = 1   (two rings)   vs  C,T = 0 (one ring)
#   amino         : A,C = 1   (amino group) vs  G,T = 0 (keto group)
#   strong_hbond  : G,C = 1   (3 H-bonds)   vs  A,T = 0 (2 H-bonds)
# Augmented with molecular weight to break any residual ties / add scale.
_NT_PURINE = {"A": 1.0, "C": 0.0, "G": 1.0, "T": 0.0}
_NT_AMINO = {"A": 1.0, "C": 1.0, "G": 0.0, "T": 0.0}
_NT_STRONG_HBOND = {"A": 0.0, "C": 1.0, "G": 1.0, "T": 0.0}
_NT_WEIGHT = {"A": 135.13, "C": 111.10, "G": 151.13, "T": 126.11}

_NT_PROPERTY_TABLES = [
    ("purine", _NT_PURINE),
    ("amino", _NT_AMINO),
    ("strong_hbond", _NT_STRONG_HBOND),
    ("weight", _NT_WEIGHT),
]


# ---------------------------------------------------------------------------
# Code container
# ---------------------------------------------------------------------------

@dataclass
class ContinuousCodeBook:
    """A fixed continuous target code per token.

    Attributes
    ----------
    codes : np.ndarray
        Shape ``(vocab_size, d)``. Row ``i`` is the regression target for token
        ``i``. Special tokens (mask/pad/...) get an all-zero row and are never
        used as regression targets or prototypes.
    d : int
        Code dimensionality.
    kind : str
        ``'physchem'`` or ``'random'`` (the control).
    alphabet : list[str]
        The real (decodable) tokens, in token-id order ``0..len(alphabet)-1``.
    token_to_id : dict[str, int]
    prototype_ids : np.ndarray
        Token ids that participate in nearest-prototype decoding (the real
        alphabet only, never specials).
    feature_names : list[str]
        Human-readable description of each code dimension (physchem only).
    """

    codes: np.ndarray
    d: int
    kind: str
    alphabet: list
    token_to_id: dict
    prototype_ids: np.ndarray
    feature_names: list = field(default_factory=list)

    @property
    def vocab_size(self) -> int:
        return self.codes.shape[0]

    @property
    def prototypes(self) -> np.ndarray:
        """The ``(len(alphabet), d)`` matrix of decodable prototype codes."""
        return self.codes[self.prototype_ids]

    def min_pairwise_distance(self) -> float:
        """Smallest Euclidean distance between any two prototype codes.

        A sanity check that identity is recoverable: if this collapses toward 0,
        nearest-prototype decoding cannot separate residues and the objective
        goes "inert" (the geometrically-clean-but-useless failure mode).
        """
        P = self.prototypes
        diff = P[:, None, :] - P[None, :, :]
        d2 = np.sqrt((diff ** 2).sum(-1))
        iu = np.triu_indices(len(P), k=1)
        return float(d2[iu].min())


# ---------------------------------------------------------------------------
# Construction helpers
# ---------------------------------------------------------------------------

def _zscore_property_matrix(alphabet: Sequence[str], tables) -> tuple:
    """Build a (len(alphabet), n_features) z-scored property matrix."""
    feats = []
    names = []
    for name, table in tables:
        col = np.array([float(table[a]) for a in alphabet], dtype=np.float64)
        mu, sd = col.mean(), col.std()
        if sd < 1e-12:
            sd = 1.0
        feats.append((col - mu) / sd)
        names.append(name)
    M = np.stack(feats, axis=1)  # (K, n_features)
    return M, names


def _random_orthonormal_codes(n_tokens: int, d: int, seed: int) -> np.ndarray:
    """``n_tokens`` rows of a random (semi-)orthonormal matrix in R^d.

    Built from a QR decomposition so rows are unit-norm and mutually orthogonal
    (when ``n_tokens <= d``). This is the biologically-arbitrary but
    identity-preserving control target.
    """
    rng = np.random.default_rng(seed)
    if n_tokens <= d:
        a = rng.standard_normal((d, d))
        q, r = np.linalg.qr(a)
        # Fix signs for determinism (so rng/QR sign conventions don't matter).
        q = q * np.sign(np.diag(r))
        return q[:n_tokens]  # orthonormal rows
    # More tokens than dimensions: can't be fully orthonormal; use normalized
    # Gaussian rows (still well-separated, identity-preserving).
    a = rng.standard_normal((n_tokens, d))
    a /= np.linalg.norm(a, axis=1, keepdims=True)
    return a


def _scale_to_unit_rms(codes: np.ndarray) -> np.ndarray:
    """Scale a code matrix so its per-token vectors have ~unit RMS norm.

    Keeps the MSE target magnitude on the same order across ``physchem`` and
    ``random`` flavours, so the loss landscapes are comparable.
    """
    rms = np.sqrt((codes ** 2).sum(axis=1).mean())
    if rms < 1e-12:
        return codes
    return codes / rms


def _assemble_codebook(
    alphabet, real_codes, d, kind, specials, feature_names,
):
    """Place real-token codes at ids 0..K-1 and append zero rows for specials."""
    K = len(alphabet)
    vocab_size = K + len(specials)
    codes = np.zeros((vocab_size, d), dtype=np.float32)
    codes[:K] = real_codes.astype(np.float32)
    token_to_id = {tok: i for i, tok in enumerate(alphabet)}
    for j, s in enumerate(specials):
        token_to_id[s] = K + j
    return ContinuousCodeBook(
        codes=codes,
        d=d,
        kind=kind,
        alphabet=list(alphabet),
        token_to_id=token_to_id,
        prototype_ids=np.arange(K),
        feature_names=list(feature_names),
    )


def _build_codes(
    alphabet, tables, kind, d, seed, specials,
):
    """Shared builder for protein & DNA codebooks."""
    K = len(alphabet)
    if kind == "physchem":
        M, names = _zscore_property_matrix(alphabet, tables)
        n_feat = M.shape[1]
        if d is None:
            d = n_feat
        if d == n_feat:
            real = M
        elif d < n_feat:
            # Compact to d dims via PCA (preserves separation, drops redundancy).
            Mc = M - M.mean(0, keepdims=True)
            u, s, vt = np.linalg.svd(Mc, full_matrices=False)
            real = (u[:, :d] * s[:d])
            names = [f"pc{i+1}" for i in range(d)]
        else:  # d > n_feat: pad with zeros (no fabricated information)
            real = np.concatenate([M, np.zeros((K, d - n_feat))], axis=1)
            names = names + [f"pad{i+1}" for i in range(d - n_feat)]
        real = _scale_to_unit_rms(real)
        return _assemble_codebook(alphabet, real, d, kind, specials, names)
    elif kind == "random":
        if d is None:
            d = len(tables)  # match physchem dim by default
        real = _random_orthonormal_codes(K, d, seed)
        real = _scale_to_unit_rms(real)
        names = [f"rand{i+1}" for i in range(d)]
        return _assemble_codebook(alphabet, real, d, kind, specials, names)
    else:
        raise ValueError(f"Unknown code kind: {kind!r} (use 'physchem' or 'random')")


def build_protein_codes(
    kind: str = "physchem",
    d: Optional[int] = None,
    seed: int = 320,
    specials: Sequence[str] = ("<mask>", "<pad>", "<cls>", "<eos>", "<unk>"),
) -> ContinuousCodeBook:
    """Fixed continuous target code per amino acid.

    Parameters
    ----------
    kind : {'physchem', 'random'}
        ``physchem`` -> standardized 8-D biophysical property vector.
        ``random``   -> fixed random orthonormal code (the confound control).
    d : int, optional
        Code dimensionality. Defaults to 8 (the number of physchem features) so
        the two flavours share an output dimension and are directly comparable.
    seed : int
        Seed for the random-orthonormal construction.
    specials : sequence of str
        Special tokens appended after the 20 amino acids. They get zero codes and
        are excluded from nearest-prototype decoding.
    """
    return _build_codes(AMINO_ACIDS, _AA_PROPERTY_TABLES, kind, d, seed, specials)


def build_dna_codes(
    kind: str = "physchem",
    d: Optional[int] = None,
    seed: int = 320,
    specials: Sequence[str] = ("<mask>", "<pad>", "<cls>", "<eos>", "<unk>"),
) -> ContinuousCodeBook:
    """Fixed continuous target code per nucleotide.

    Parameters
    ----------
    kind : {'physchem', 'random'}
        ``physchem`` -> standardized structural feature vector (purine, amino,
        strong-H-bond, weight): a non-monotone, identity-preserving code.
        ``random``   -> fixed random orthonormal code (the confound control).
    d, seed, specials : see :func:`build_protein_codes`.
    """
    return _build_codes(NUCLEOTIDES, _NT_PROPERTY_TABLES, kind, d, seed, specials)


# ---------------------------------------------------------------------------
# Nearest-prototype decoding (the Cont-condition analog of CE's argmax)
# ---------------------------------------------------------------------------

def nearest_prototype_decode(
    preds: np.ndarray,
    codebook: ContinuousCodeBook,
) -> np.ndarray:
    """Map continuous predictions to token ids by nearest prototype.

    Parameters
    ----------
    preds : np.ndarray
        Shape ``(..., d)`` continuous predictions from the regression head.
    codebook : ContinuousCodeBook

    Returns
    -------
    np.ndarray
        Token ids (into the full vocab; restricted to the decodable alphabet),
        shape ``preds.shape[:-1]``.
    """
    preds = np.asarray(preds)
    flat = preds.reshape(-1, preds.shape[-1])
    P = codebook.prototypes                      # (K, d)
    # Squared Euclidean distance to every prototype.
    # ||x||^2 - 2 x.P^T + ||P||^2  ; the ||x||^2 term is constant per row.
    cross = flat @ P.T                           # (N, K)
    pnorm = (P ** 2).sum(axis=1)[None, :]        # (1, K)
    d2 = pnorm - 2.0 * cross                      # argmin is unaffected by ||x||^2
    nearest = np.argmin(d2, axis=1)              # index into alphabet
    ids = codebook.prototype_ids[nearest]
    return ids.reshape(preds.shape[:-1])


def recovery_accuracy(
    preds: np.ndarray,
    target_ids: np.ndarray,
    codebook: ContinuousCodeBook,
) -> float:
    """Masked/next-token recovery accuracy for the continuous head.

    Directly comparable to CE argmax accuracy: the fraction of target positions
    whose nearest prototype equals the true token.
    """
    pred_ids = nearest_prototype_decode(preds, codebook)
    target_ids = np.asarray(target_ids)
    return float((pred_ids.reshape(-1) == target_ids.reshape(-1)).mean())


# ---------------------------------------------------------------------------
# Geometric distortion (the headline geometry metric)
# ---------------------------------------------------------------------------

def procrustes_distortion(
    X_clean: np.ndarray,
    X_pert: np.ndarray,
    scaling: bool = True,
) -> float:
    """Procrustes distortion D between clean and perturbed representations.

    Both point sets are centred and Frobenius-normalised, then optimally aligned
    by an orthogonal transform (plus optional isotropic scaling). D is the
    residual sum-of-squares after alignment, normalised to ``[0, 1]``:

        D = 0  -> perturbation acts as a rigid motion of the representation
                  (geometry perfectly preserved up to rotation/scale).
        D -> 1 -> the representation is reshuffled / fractured.

    Lower D = more geometrically stable. The synthetic Variant-A prediction is
    that the continuous-objective condition shows materially lower D than CE
    under matched perturbations.

    Parameters
    ----------
    X_clean, X_pert : np.ndarray
        Shape ``(n_samples, d)``. Row ``i`` of each is the representation of the
        same item, clean vs perturbed.
    scaling : bool
        If True, allow an optimal isotropic scale in the alignment (standard
        Procrustes; matches ``scipy.spatial.procrustes``).
    """
    A = np.asarray(X_clean, dtype=np.float64)
    B = np.asarray(X_pert, dtype=np.float64)
    if A.shape != B.shape:
        raise ValueError(f"shape mismatch: {A.shape} vs {B.shape}")

    A = A - A.mean(axis=0, keepdims=True)
    B = B - B.mean(axis=0, keepdims=True)
    normA = np.linalg.norm(A)
    normB = np.linalg.norm(B)
    if normA < 1e-12 or normB < 1e-12:
        return float("nan")
    A = A / normA
    B = B / normB

    # Optimal orthogonal alignment of B onto A.
    M = B.T @ A
    u, s, vt = np.linalg.svd(M, full_matrices=False)
    R = u @ vt
    if scaling:
        scale = s.sum()           # optimal isotropic scale for unit-norm sets
        B_aligned = scale * (B @ R)
    else:
        B_aligned = B @ R
    D = float(((A - B_aligned) ** 2).sum())
    return D


def rdm(X: np.ndarray, metric: str = "cosine") -> np.ndarray:
    """Representational dissimilarity matrix (upper-triangle vectorisable)."""
    X = np.asarray(X, dtype=np.float64)
    if metric == "cosine":
        Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
        return 1.0 - Xn @ Xn.T
    elif metric == "euclidean":
        sq = (X ** 2).sum(1)
        d2 = sq[:, None] + sq[None, :] - 2.0 * (X @ X.T)
        return np.sqrt(np.clip(d2, 0, None))
    raise ValueError(metric)


def rdm_similarity(X_clean: np.ndarray, X_pert: np.ndarray, metric: str = "cosine") -> float:
    """Spearman-free RDM similarity: Pearson r between clean & perturbed RDMs.

    Higher = perturbation preserves the relational geometry. (A lightweight,
    torch-free companion to the Shesha harness metric of the same name, so the
    matched-accuracy checkpoint analysis can run cheaply per checkpoint.)
    """
    Rc = rdm(X_clean, metric)
    Rp = rdm(X_pert, metric)
    iu = np.triu_indices(Rc.shape[0], k=1)
    a, b = Rc[iu], Rp[iu]
    if a.std() < 1e-12 or b.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Biological Continuous Codes - Smoke Test (seed=320) ===\n")

    for builder, name, alpha in [
        (build_protein_codes, "PROTEIN", AMINO_ACIDS),
        (build_dna_codes, "DNA", NUCLEOTIDES),
    ]:
        print(f"--- {name} ---")
        phys = builder(kind="physchem", seed=320)
        rand = builder(kind="random", d=phys.d, seed=320)
        print(f"  physchem: codes {phys.codes.shape}, d={phys.d}, "
              f"min_pairwise={phys.min_pairwise_distance():.4f}, "
              f"feats={phys.feature_names}")
        print(f"  random:   codes {rand.codes.shape}, d={rand.d}, "
              f"min_pairwise={rand.min_pairwise_distance():.4f}")

        # Identity must be perfectly recoverable from the exact codes.
        for cb in (phys, rand):
            ids = cb.prototype_ids
            dec = nearest_prototype_decode(cb.codes[ids], cb)
            assert np.array_equal(dec, ids), f"{name}/{cb.kind}: exact decode failed"
        print("  exact-code nearest-prototype decode: 100% (identity recoverable)")

        # Noisy decode should degrade gracefully, not collapse.
        rng = np.random.default_rng(320)
        ids = phys.prototype_ids
        noisy = phys.codes[ids] + 0.15 * rng.standard_normal(phys.prototypes.shape)
        acc = (nearest_prototype_decode(noisy, phys) == ids).mean()
        print(f"  noisy (sigma=0.15) recovery accuracy: {acc*100:.1f}%")

        # Non-monotonicity check: physchem code dim-0 must NOT be a monotone
        # function of token index (that would be the staircase leak).
        c0 = phys.prototypes[:, 0]
        monotone = np.all(np.diff(c0) > 0) or np.all(np.diff(c0) < 0)
        assert not monotone, f"{name}: physchem code is monotone in index (leak!)"
        print("  non-monotone in token index: OK (no staircase leak)\n")

    # Procrustes distortion sanity: rigid rotation -> ~0, reshuffle -> larger.
    rng = np.random.default_rng(320)
    X = rng.standard_normal((200, 16))
    Q, _ = np.linalg.qr(rng.standard_normal((16, 16)))
    X_rot = X @ Q                          # rigid -> D ~ 0
    X_shuf = X + 0.9 * rng.standard_normal((200, 16))  # large noise -> D up
    d_rot = procrustes_distortion(X, X_rot)
    d_shuf = procrustes_distortion(X, X_shuf)
    print(f"Procrustes D: rigid-rotation={d_rot:.2e}, heavy-noise={d_shuf:.4f}")
    assert d_rot < 1e-6 < d_shuf, "Procrustes distortion sanity failed"
    print(f"RDM similarity: rigid-rotation={rdm_similarity(X, X_rot):.4f}, "
          f"heavy-noise={rdm_similarity(X, X_shuf):.4f}")

    print("\nAll smoke tests passed.")
