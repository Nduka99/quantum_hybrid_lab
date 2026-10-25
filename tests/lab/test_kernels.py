"""The statevector fidelity kernel has the properties a kernel must have."""

import numpy as np
import pytest
from qiskit.circuit.library import z_feature_map

from qhcyber.quantum.kernels import (
    build_feature_map,
    circuit_resources,
    fidelity_kernel,
    kernel_diagnostics,
    statevectors,
)


@pytest.fixture(scope="module")
def inputs() -> np.ndarray:
    return np.random.default_rng(7).random((12, 3))


def test_kernel_is_a_valid_gram_matrix(inputs) -> None:
    states = statevectors(build_feature_map(3, 2, "linear"), inputs)
    kernel = fidelity_kernel(states, states)
    np.testing.assert_allclose(kernel, kernel.T, atol=1e-12)
    np.testing.assert_allclose(np.diag(kernel), 1.0, atol=1e-12)
    assert kernel.min() >= -1e-12 and kernel.max() <= 1 + 1e-12
    assert np.linalg.eigvalsh(kernel).min() >= -1e-10


def test_zero_bandwidth_makes_every_point_identical(inputs) -> None:
    states = statevectors(build_feature_map(3, 2, "full"), 0.0 * inputs)
    np.testing.assert_allclose(fidelity_kernel(states, states), 1.0, atol=1e-12)


def test_single_qubit_kernel_matches_closed_form() -> None:
    # H then P(2x) prepares (|0> + e^{2ix}|1>)/sqrt(2), so K(x, y) = cos^2(x - y).
    x = np.array([[0.1], [0.7], [2.0]])
    states = statevectors(z_feature_map(1, reps=1), x)
    expected = np.cos(x - x.T) ** 2
    np.testing.assert_allclose(fidelity_kernel(states, states), expected, atol=1e-12)


def test_matches_qiskit_machine_learning(inputs) -> None:
    kernels = pytest.importorskip("qiskit_machine_learning.kernels")
    feature_map = build_feature_map(3, 2, "linear")
    reference = kernels.FidelityStatevectorKernel(feature_map=feature_map, enforce_psd=False)
    states = statevectors(feature_map, inputs)
    np.testing.assert_allclose(
        fidelity_kernel(states, states[:5]), reference.evaluate(inputs, inputs[:5]), atol=1e-10
    )


def test_diagnostics_detect_concentration_and_alignment() -> None:
    labels = np.array([0, 0, 1, 1])
    identity = np.eye(4)
    concentrated = kernel_diagnostics(identity, labels)
    assert concentrated["offdiagonal_mean"] == 0.0
    assert concentrated["effective_rank"] == pytest.approx(4.0)

    ideal = (np.outer(labels, labels) + np.outer(1 - labels, 1 - labels)).astype(float)
    aligned = kernel_diagnostics(ideal, labels)
    assert aligned["kernel_target_alignment"] == pytest.approx(1 / np.sqrt(2))
    assert aligned["effective_rank"] == pytest.approx(2.0)


def test_resources_count_hardware_work() -> None:
    resources = circuit_resources(4, 1, "linear", n_train=10, n_test=5, n_bandwidths=6)
    assert resources["qubits"] == 4
    assert resources["statevectors_simulated"] == 6 * 10 + 5
    assert resources["hardware_fidelity_circuits"] == 6 * 45 + 5 * 10
    # U(x) U(y)^dagger needs up to twice the gates; the transpiler cancels some at the seam.
    single = resources["feature_map_two_qubit_gates"]
    assert single < resources["fidelity_circuit_two_qubit_gates"] <= 2 * single
