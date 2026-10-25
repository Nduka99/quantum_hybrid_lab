"""Figures for the hybrid lab reference runs (used by build_lab_reference_runs.py).

Colour follows the entity in every figure: quantum kernel with tuned bandwidth, quantum kernel at
the fixed [0, pi] scale, and the RBF SVM. The palette is the first three slots of a validated
categorical palette (all-pairs colour-vision-deficiency check passed); lines are direct-labelled.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e4e3df"
TUNED = "#2a78d6"
FIXED_PI = "#eb6834"
RBF = "#1baf7a"

DATASET_TITLES = {
    "network_flows": "Network flows (cybersecurity)",
    "breast_cancer": "Breast cancer (medicine)",
}
QUANTUM = "Quantum-kernel SVM"


def style_axes(axes) -> None:
    axes.set_facecolor(SURFACE)
    axes.grid(axis="y", color=GRID, linewidth=0.8)
    axes.set_axisbelow(True)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axes.spines[side].set_color(GRID)
    axes.tick_params(colors=INK_SECONDARY, labelsize=9)


def series(runs: list[dict], dataset: str, variant: str, value) -> tuple[list, np.ndarray]:
    """Per-qubit array of seed values for one dataset and variant."""

    qubits = sorted({run["qubits"] for run in runs})
    values = [
        [
            value(run)
            for run in runs
            if (run["dataset"], run["variant"], run["qubits"]) == (dataset, variant, q)
        ]
        for q in qubits
    ]
    return qubits, np.array(values)


MIN_LABEL_GAP = 0.07  # axes fraction between stacked end labels


def place_end_labels(axes, ends: list[tuple[str, float, float]]) -> None:
    """Direct labels just right of each line's last point, nudged apart so none overlap."""

    to_axes = axes.transAxes.inverted()
    placed = []
    for label, x, y in ends:
        fx, fy = to_axes.transform(axes.transData.transform((x, y)))
        placed.append([fy, fx, label, x, y])
    placed.sort()
    for index in range(1, len(placed)):
        placed[index][0] = max(placed[index][0], placed[index - 1][0] + MIN_LABEL_GAP)
    for fy, fx, label, x, y in placed:
        axes.annotate(
            label,
            xy=(x, y),
            xytext=(fx + 0.02, fy),
            textcoords="axes fraction",
            va="center",
            fontsize=8.5,
            color=INK,
        )


def draw_lines(
    axes,
    lines: list[tuple[str, str, list, np.ndarray]],
    *,
    log: bool = False,
    ylim: tuple[float, float] | None = None,
) -> None:
    ends = []
    for label, colour, qubits, values in lines:
        mean = values.mean(axis=1)
        axes.fill_between(
            qubits, values.min(axis=1), values.max(axis=1), color=colour, alpha=0.12, linewidth=0
        )
        axes.plot(qubits, mean, color=colour, linewidth=2, marker="o", markersize=5, label=label)
        ends.append((label, qubits[-1], mean[-1]))
    qubits = lines[0][2]
    axes.set_xticks(qubits)
    axes.set_xlim(qubits[0] - 0.5, qubits[-1] + 5.5)
    if log:
        axes.set_yscale("log")
    if ylim is not None:
        axes.set_ylim(*ylim)
    place_end_labels(axes, ends)


def finish(figure, panels, note: str, path: Path, legend: bool = True) -> None:
    if legend:
        handles, labels = panels[0].get_legend_handles_labels()
        figure.legend(
            handles, labels, loc="upper center", ncol=len(labels), frameon=False, fontsize=9
        )
    figure.text(0.5, 0.015, note, ha="center", va="bottom", fontsize=8.5, color=INK_SECONDARY)
    figure.tight_layout(rect=(0, 0.06, 1, 0.9 if legend else 1))
    figure.savefig(path, dpi=160, facecolor=SURFACE, metadata={"Software": None})
    plt.close(figure)


def accuracy_figure(runs: list[dict], path: Path) -> None:
    figure, panels = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True, facecolor=SURFACE)
    for axes, dataset in zip(panels, DATASET_TITLES, strict=True):
        style_axes(axes)
        accuracy = lambda name: lambda run: run["models"][name]["balanced_accuracy"]  # noqa: E731
        qubits, tuned = series(runs, dataset, "tuned", accuracy(QUANTUM))
        _, fixed = series(runs, dataset, "fixed_pi", accuracy(QUANTUM))
        _, rbf = series(runs, dataset, "tuned", accuracy("RBF SVM"))
        _, control = series(runs, dataset, "tuned", accuracy("LightGBM (all features)"))
        # Stops at the last qubit count so it never crosses the direct labels.
        axes.plot(
            [qubits[0] - 0.5, qubits[-1]],
            [control.mean()] * 2,
            color=INK_SECONDARY,
            linewidth=1.2,
            linestyle=(0, (4, 3)),
            label="LightGBM, all features (control)",
        )
        draw_lines(
            axes,
            [
                ("Quantum, tuned bandwidth", TUNED, qubits, tuned),
                ("Quantum, fixed [0, π]", FIXED_PI, qubits, fixed),
                ("RBF SVM", RBF, qubits, rbf),
            ],
            ylim=(0.45, 1.0),
        )
        axes.set_title(DATASET_TITLES[dataset], fontsize=11, color=INK, loc="left")
        axes.set_xlabel("Qubits (= PCA components)", fontsize=9, color=INK_SECONDARY)
    panels[0].set_ylabel("Held-out balanced accuracy", fontsize=9, color=INK_SECONDARY)
    note = "Mean of 5 seeds (band = min to max); 200 train / 200 test rows per run"
    finish(figure, panels, note, path)


def concentration_figure(runs: list[dict], path: Path) -> None:
    figure, panels = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True, facecolor=SURFACE)
    for axes, dataset in zip(panels, DATASET_TITLES, strict=True):
        style_axes(axes)
        quantum = lambda run: run["quantum_kernel"]["offdiagonal_mean"]  # noqa: E731
        qubits, tuned = series(runs, dataset, "tuned", quantum)
        _, fixed = series(runs, dataset, "fixed_pi", quantum)
        _, rbf = series(runs, dataset, "tuned", lambda run: run["rbf_kernel"]["offdiagonal_mean"])
        draw_lines(
            axes,
            [
                ("Quantum, tuned bandwidth", TUNED, qubits, tuned),
                ("Quantum, fixed [0, π]", FIXED_PI, qubits, fixed),
                ("RBF SVM", RBF, qubits, rbf),
            ],
            log=True,
            ylim=(3e-4, 2.0),
        )
        axes.set_title(DATASET_TITLES[dataset], fontsize=11, color=INK, loc="left")
        axes.set_xlabel("Qubits (= PCA components)", fontsize=9, color=INK_SECONDARY)
    panels[0].set_ylabel("Mean off-diagonal kernel value (log)", fontsize=9, color=INK_SECONDARY)
    note = "Values near 0 mean every pair of points looks unrelated: the kernel has concentrated"
    finish(figure, panels, note, path)


def paired_difference_figure(runs: list[dict], path: Path) -> None:
    """Per seed: tuned quantum kernel minus the best matched classical model on that seed."""

    def difference(run: dict) -> float:
        matched = [
            model["balanced_accuracy"]
            for model in run["models"].values()
            if model["tier"] == "matched classical"
        ]
        return run["models"][QUANTUM]["balanced_accuracy"] - max(matched)

    figure, panels = plt.subplots(1, 2, figsize=(11, 3.9), sharey=True, facecolor=SURFACE)
    for axes, dataset in zip(panels, DATASET_TITLES, strict=True):
        style_axes(axes)
        qubits, values = series(runs, dataset, "tuned", difference)
        axes.axhline(0, color=INK_SECONDARY, linewidth=1)
        for position, seeds in enumerate(values):
            jitter = np.linspace(-0.12, 0.12, len(seeds))
            axes.scatter(
                position + jitter,
                seeds,
                s=36,
                color=TUNED,
                edgecolor=SURFACE,
                linewidth=1.2,
                zorder=3,
            )
            axes.plot(
                [position - 0.25, position + 0.25], [seeds.mean()] * 2, color=INK, linewidth=2
            )
        axes.set_xticks(range(len(qubits)), [str(q) for q in qubits])
        axes.set_title(DATASET_TITLES[dataset], fontsize=11, color=INK, loc="left")
        axes.set_xlabel("Qubits", fontsize=9, color=INK_SECONDARY)
    panels[0].set_ylabel(
        "Quantum − best matched classical\n(balanced accuracy)", fontsize=9, color=INK_SECONDARY
    )
    note = "Dots = seeds, bar = mean. Below zero: the quantum kernel scored lower on that seed."
    finish(figure, panels, note, path, legend=False)


def draw_figures(runs: list[dict], directory: Path) -> list[str]:
    """Write the three PNGs and return their paths relative to the asset directory."""

    directory.mkdir(parents=True, exist_ok=True)
    plt.rcParams["font.family"] = "DejaVu Sans"
    names = {
        "accuracy_by_qubits.png": accuracy_figure,
        "kernel_concentration.png": concentration_figure,
        "paired_differences.png": paired_difference_figure,
    }
    for name, draw in names.items():
        draw(runs, directory / name)
    return [f"{directory.name}/{name}" for name in names]
