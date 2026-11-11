from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from Manage_files import gaussian, time_intervals


def _save_figure(filename):
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()


def plot_gaussian_fit(centers, counts, parameters, mu, sigma, title, filename):
    """Point 1: Plot time-difference histogram in nanoseconds with Gaussian fit."""
    centers_ns = centers * 1e9
    x_ns = np.linspace(centers.min(), centers.max(), 1500) * 1e9
    mu_ns = mu * 1e9
    sigma_ns = sigma * 1e9

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.step(centers_ns, counts, where="mid", label="Histogram")
    ax.plot(x_ns, gaussian(np.linspace(centers.min(), centers.max(), 1500), *parameters), label="Gaussian fit")

    ax.set_xlabel("Detector - Herald time difference (ns)")
    ax.set_ylabel("Counts")
    # Usiamo una raw string (r"...") per evitare i SyntaxWarning con \mu e \sigma
    ax.set_title(r"{0}"
                 "\n$\mu$ = {1:.3f} ns, $\sigma$ = {2:.3f} ns".format(title, mu_ns, sigma_ns))
    ax.legend()
    ax.grid(True, alpha=0.3)

    _save_figure(filename)


def plot_alpha_window(w_values, alpha_values, title, filename):
    """Point 3: Plot alpha vs window width w together with classical limit alpha = 1."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(w_values, alpha_values, "o-", markersize=5, label=r"$\alpha(w)$")
    ax.axhline(1.0, color="red", linestyle="--", label=r"Classical limit ($\alpha=1$)")

    ax.set_xlabel(r"Window multiplier $w$ (where Window = $\mu \pm w \cdot \sigma$)")
    ax.set_ylabel(r"Anticorrelation parameter $\alpha$")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    _save_figure(filename)


def plot_time_intervals(time_tags_sec, detector_name, title, filename, bins=250):
    """Point 4: Plot histogram of consecutive click intervals on a logarithmic y-axis."""
    intervals_sec = time_intervals(time_tags_sec)

    if intervals_sec.size == 0:
        raise ValueError(f"Not enough events to plot intervals for {detector_name}.")

    intervals_ns = intervals_sec * 1e9

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(intervals_ns, bins=bins, histtype="step", linewidth=1.2)
    ax.set_yscale("log")

    ax.set_xlabel("Time between consecutive clicks (ns)")
    ax.set_ylabel("Counts (log scale)")
    ax.set_title(f"Inter-arrival time distribution: {title} - {detector_name}")
    ax.grid(True, which="both", alpha=0.3)

    _save_figure(filename)


def plot_alpha_comparison(w_values, alpha_original, alpha_filtered, title, filename):
    """Point 5: Compare alpha(w) before and after afterpulse removal."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(w_values, alpha_original, "o-", markersize=5, label="Raw Data")
    ax.plot(w_values, alpha_filtered, "s-", markersize=5, label="Afterpulse-Filtered Data")
    ax.axhline(1.0, color="red", linestyle="--", label=r"Classical limit ($\alpha=1$)")

    ax.set_xlabel(r"Window multiplier $w$")
    ax.set_ylabel(r"Anticorrelation parameter $\alpha$")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    _save_figure(filename)