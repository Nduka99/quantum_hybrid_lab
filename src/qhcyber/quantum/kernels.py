"""Bandwidth-scaled fidelity quantum kernels, their diagnostics and their resource cost.

The kernel is K(x, y) = |<psi(b * x) | psi(b * y)>|^2 for a ZZ feature map and bandwidth b. It is
computed from exact statevectors, so it is positive semidefinite by construction: it equals the
Hilbert-Schmidt inner product of the density matrices |psi><psi|.

Bandwidth matters. Without it, kernel values concentrate towards zero as qubits are added and the
classifier degrades to guessing (Shaydulin & Wild, arXiv:2111.05451; Thanasilp et al.,
doi:10.1038/s41467-024-49287-w).
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import zz_feature_map
from qiskit.quantum_info import Statevector

BASIS_GATES = ["rz", "sx", "x", "cx"]


def build_feature_map(n_qubits: int, reps: int, entanglement: str) -> QuantumCircuit:
    """ZZ feature map: one qubit per input feature."""

    return zz_feature_map(n_qubits, reps=reps, entanglement=entanglement)


def statevectors(circuit: QuantumCircuit, angles: np.ndarray) -> np.ndarray:
    """Exact statevector for each row of angles, as a (rows, 2**qubits) complex matrix."""

    return np.array([Statevector(circuit.assign_parameters(row)).data for row in angles])


def fidelity_kernel(states_a: np.ndarray, states_b: np.ndarray) -> np.ndarray:
    """K[i, j] = |<a_i | b_j>|^2."""

    return np.abs(states_a @ states_b.conj().T) ** 2


def kernel_diagnostics(kernel: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """Summaries that expose concentration and how well the kernel matches the labels.

    - off-diagonal mean/std: values near zero with tiny spread mean every point looks unrelated
      to every other (concentration); values near one mean every point looks the same.
    - kernel-target alignment (Cristianini et al., 2001): cosine between K and y y^T, y in {-1, 1}.
    - effective rank (Roy & Vetterli, 2007): exp(entropy of the normalised eigenvalues).
    """

    n = kernel.shape[0]
    off_diagonal = kernel[~np.eye(n, dtype=bool)]
    signs = np.where(np.asarray(labels) == 1, 1.0, -1.0)
    target = np.outer(signs, signs)
    alignment = np.sum(kernel * target) / (np.linalg.norm(kernel) * np.linalg.norm(target))

    eigenvalues = np.clip(np.linalg.eigvalsh(kernel), 0.0, None)
    weights = eigenvalues[eigenvalues > 0] / eigenvalues.sum()
    effective_rank = np.exp(-np.sum(weights * np.log(weights)))
    return {
        "offdiagonal_mean": float(off_diagonal.mean()),
        "offdiagonal_std": float(off_diagonal.std()),
        "kernel_target_alignment": float(alignment),
        "effective_rank": float(effective_rank),
    }


def circuit_resources(
    n_qubits: int,
    reps: int,
    entanglement: str,
    *,
    n_train: int,
    n_test: int,
    n_bandwidths: int,
) -> dict[str, int]:
    """Gate counts and the work a simulator did versus what quantum hardware would need.

    Hardware estimates each fidelity with a compute-uncompute circuit U(x) U(y)^dagger: every
    training pair for each bandwidth tried, then every test-train pair once.
    """

    feature_map = build_feature_map(n_qubits, reps, entanglement)
    # Separate parameters for U(y) so the transpiler cannot cancel it against U(x).
    other = zz_feature_map(n_qubits, reps=reps, entanglement=entanglement, parameter_prefix="y")
    compiled_map = transpile(feature_map, basis_gates=BASIS_GATES, optimization_level=1)
    compiled_pair = transpile(
        feature_map.compose(other.inverse()), basis_gates=BASIS_GATES, optimization_level=1
    )
    train_pairs = n_train * (n_train - 1) // 2
    return {
        "qubits": n_qubits,
        "feature_map_depth": compiled_map.depth(),
        "feature_map_two_qubit_gates": compiled_map.count_ops().get("cx", 0),
        "fidelity_circuit_depth": compiled_pair.depth(),
        "fidelity_circuit_two_qubit_gates": compiled_pair.count_ops().get("cx", 0),
        "statevectors_simulated": n_bandwidths * n_train + n_test,
        "hardware_fidelity_circuits": n_bandwidths * train_pairs + n_test * n_train,
    }
