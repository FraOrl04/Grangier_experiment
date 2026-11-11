from pathlib import Path
import numpy as np

from Manage_files import (
    read_file,
    converter,
    divided_by_channel,
    herald_differences,
    gaussian_fit,
    coincidences,
    probabilities,
    anticorrelation,
    uncertainty_alpha,
    alpha_vs_window,
    remove_afterpulses
)

from Plot_files import (
    plot_gaussian_fit,
    plot_alpha_window,
    plot_time_intervals,
    plot_alpha_comparison
)

# Set file name according to your student number (e.g. N=15 -> TimeTags_15.txt)
FILES = [
    "TimeTags_15.txt"
]

# Parameters defined in seconds:
MAX_LAG_SEC = 15e-9  # Range +/- 15 ns to look for coincidences
FIT_HALF_WIDTH_SEC = 2.5e-9  # Half-width region around the peak for Gaussian fit (2.5 ns)
W_VALUES = np.linspace(0.5, 5.0, 10)  # Window multipliers w

ASK_AFTERPULSE_THRESHOLDS = True


def fit_channel_pair(detector_tags_sec, herald_tags_sec, detector_label, output_dir):
    differences_sec = herald_differences(
        detector_tags_sec, herald_tags_sec, max_lag_sec=MAX_LAG_SEC
    )

    if differences_sec.size == 0:
        raise ValueError(f"No {detector_label}-herald pairs found within lag window.")

    mu, sigma, centers, counts, parameters, covariance = gaussian_fit(
        differences_sec,
        bins=1000,
        fit_half_width=FIT_HALF_WIDTH_SEC
    )

    plot_gaussian_fit(
        centers, counts, parameters, mu, sigma,
        f"{detector_label} - Herald coincidences",
        output_dir / f"gaussian_{detector_label.lower().replace(' ', '_')}.png"
    )

    return mu, sigma


def analyse_one_file(path):
    path = Path(path)
    output_dir = Path("Results") / path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- STEP 0: Read raw time-tags & CONVERT TO SECONDS FIRST ---
    raw_time_tags, channels = read_file(path)
    time_tags_sec = converter(raw_time_tags)

    tags_1, tags_2, tags_3 = divided_by_channel(time_tags_sec, channels)

    print(f"\n==========================================")
    print(f"       ANALYSIS: {path.name}")
    print(f"==========================================")
    print(f"Channel 1 (Transmitted) events: {len(tags_1)}")
    print(f"Channel 2 (Reflected) events:   {len(tags_2)}")
    print(f"Channel 3 (Herald) events:      {len(tags_3)}")

    # --- POINT 1: Differences & Gaussian Fits ---
    print("\n[POINT 1] Fitting Gaussian peaks for coincidence windows...")
    mu_T, sigma_T = fit_channel_pair(tags_1, tags_3, "Transmitted", output_dir)
    mu_R, sigma_R = fit_channel_pair(tags_2, tags_3, "Reflected", output_dir)

    print(f"  Transmitted: mu = {mu_T * 1e9:.3f} ns, sigma = {sigma_T * 1e9:.3f} ns")
    print(f"  Reflected:   mu = {mu_R * 1e9:.3f} ns, sigma = {sigma_R * 1e9:.3f} ns")

    # --- POINT 2: Coincidences & Alpha at 3 Sigma ---
    print("\n[POINT 2] Counting coincidences with 3-sigma windows...")
    N1, Nt, Nr, Nc = coincidences(tags_1, tags_2, tags_3, mu_T, sigma_T, mu_R, sigma_R)

    print(f"  N1 (Heralds):    {N1}")
    print(f"  Nt (Transmitted): {Nt}")
    print(f"  Nr (Reflected):   {Nr}")
    print(f"  Nc (Triples):     {Nc}")

    p_t, p_r, p_c = probabilities(N1, Nt, Nr, Nc)
    alpha_3sigma = anticorrelation(N1, Nt, Nr, Nc)
    err_alpha = uncertainty_alpha(N1, Nt, Nr, Nc)

    print(f"  p_t = {p_t:.6e}")
    print(f"  p_r = {p_r:.6e}")
    print(f"  p_c = {p_c:.6e}")
    print(f"  --> Alpha (3 sigma) = {alpha_3sigma:.4f} +/- {err_alpha:.4f}")

    # --- POINT 3: Alpha vs Window Width w ---
    print("\n[POINT 3] Computing Alpha vs Window Width w...")
    alpha_original = alpha_vs_window(tags_1, tags_2, tags_3, mu_T, sigma_T, mu_R, sigma_R, W_VALUES)
    plot_alpha_window(W_VALUES, alpha_original, f"{path.stem} - Raw Alpha(w)", output_dir / "alpha_vs_window.png")

    # --- POINT 4: Inter-arrival Time Histograms ---
    print("\n[POINT 4] Plotting consecutive time interval histograms...")
    plot_time_intervals(tags_1, "Transmitted", path.stem, output_dir / "intervals_ch1.png")
    plot_time_intervals(tags_2, "Reflected", path.stem, output_dir / "intervals_ch2.png")
    plot_time_intervals(tags_3, "Herald", path.stem, output_dir / "intervals_ch3.png")

    # --- POINT 5: Afterpulse Filter & Re-Analysis ---
    if not ASK_AFTERPULSE_THRESHOLDS:
        return

    print("\n[POINT 5] Afterpulse Removal")
    print("  Inspect plots 'intervals_ch1.png', 'intervals_ch2.png', 'intervals_ch3.png'.")
    print("  Enter minimum separation threshold in NANOSECONDS (e.g. 50):")

    thresh1_ns = float(input("  Threshold Ch1 (ns): "))
    thresh2_ns = float(input("  Threshold Ch2 (ns): "))
    thresh3_ns = float(input("  Threshold Ch3/Herald (ns): "))

    filt_1 = remove_afterpulses(tags_1, thresh1_ns * 1e-9)
    filt_2 = remove_afterpulses(tags_2, thresh2_ns * 1e-9)
    filt_3 = remove_afterpulses(tags_3, thresh3_ns * 1e-9)

    print("\n  Refitting and recalculating with filtered data...")
    mu_T_f, sigma_T_f = fit_channel_pair(filt_1, filt_3, "Filtered Transmitted", output_dir)
    mu_R_f, sigma_R_f = fit_channel_pair(filt_2, filt_3, "Filtered Reflected", output_dir)

    N1_f, Nt_f, Nr_f, Nc_f = coincidences(filt_1, filt_2, filt_3, mu_T_f, sigma_T_f, mu_R_f, sigma_R_f)
    alpha_f = anticorrelation(N1_f, Nt_f, Nr_f, Nc_f)
    err_f = uncertainty_alpha(N1_f, Nt_f, Nr_f, Nc_f)

    print(f"  Filtered N1={N1_f}, Nt={Nt_f}, Nr={Nr_f}, Nc={Nc_f}")
    print(f"  --> Filtered Alpha (3 sigma) = {alpha_f:.4f} +/- {err_f:.4f}")

    alpha_filtered = alpha_vs_window(filt_1, filt_2, filt_3, mu_T_f, sigma_T_f, mu_R_f, sigma_R_f, W_VALUES)
    plot_alpha_comparison(W_VALUES, alpha_original, alpha_filtered, f"{path.stem} - Alpha Comparison",
                          output_dir / "alpha_comparison.png")


def main():
    for filename in FILES:
        if not Path(filename).exists():
            print(f"File non trovato: {filename}. Verifica la variabile FILES in main.py.")
            continue
        analyse_one_file(filename)

    print("\nAnalisi completata con successo! I grafici sono salvati nella cartella Results.")


if __name__ == "__main__":
    main()