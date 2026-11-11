import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

TIME_TAG_SECONDS = 80.955e-12  # 1 unit = 80.955 ps


def read_file(path):
    """
    Read a time-tag file with two columns: time tag ; channel.
    Lines starting with '#' are treated as comments.
    """
    df = pd.read_csv(
        path,
        comment="#",
        sep=";",
        header=None,
        skipinitialspace=True,
        dtype={0: "int64", 1: "int64"}
    )

    if df.shape[1] < 2:
        raise ValueError(f"{path}: expected two columns (time tag ; channel).")

    time_tags = df.iloc[:, 0].to_numpy(dtype=np.int64)
    channels = df.iloc[:, 1].to_numpy(dtype=np.int64)

    if not np.all(np.isin(channels, [1, 2, 3])):
        raise ValueError(f"{path}: channel values must be 1, 2, or 3.")

    order = np.argsort(time_tags)
    return time_tags[order], channels[order]


def converter(time_tags):
    """Convert raw integer time tags to seconds (float64)."""
    return np.asarray(time_tags, dtype=np.float64) * TIME_TAG_SECONDS


def time_different(time_tag_1, time_tag_2):
    """Return time_tag_2 - time_tag_1 in seconds."""
    return time_tag_2 - time_tag_1


def divided_by_channel(time_tags_sec, channels):
    """Separate converted time tags (in seconds) into channels 1, 2, and 3."""
    time_tags_sec = np.asarray(time_tags_sec, dtype=np.float64)
    channels = np.asarray(channels, dtype=np.int64)

    if len(time_tags_sec) != len(channels):
        raise ValueError("time_tags and channels must have the same length.")

    tags_1 = time_tags_sec[channels == 1]
    tags_2 = time_tags_sec[channels == 2]
    tags_3 = time_tags_sec[channels == 3]

    return tags_1, tags_2, tags_3


def herald_differences(time_tags_detector, time_tags_herald, max_lag_sec=20e-9):
    """
    Calculate detector - herald differences (in seconds) for pairs within +/- max_lag_sec.
    """
    detector = np.sort(np.asarray(time_tags_detector, dtype=np.float64))
    heralds = np.sort(np.asarray(time_tags_herald, dtype=np.float64))

    if len(detector) == 0 or len(heralds) == 0:
        return np.array([], dtype=np.float64)

    chunks = []

    for herald in heralds:
        left = np.searchsorted(detector, herald - max_lag_sec, side="left")
        right = np.searchsorted(detector, herald + max_lag_sec, side="right")

        if right > left:
            chunks.append(detector[left:right] - herald)

    if not chunks:
        return np.array([], dtype=np.float64)

    return np.concatenate(chunks)


def gaussian(x, amplitude, mu, sigma, offset):
    """Gaussian peak plus a constant background."""
    return amplitude * np.exp(-0.5 * ((x - mu) / sigma) ** 2) + offset


def gaussian_fit(time_differences, bins=1000, fit_half_width=2e-9):
    """
    Histogram the herald-detector differences (in seconds) and fit the peak.
    """
    differences = np.asarray(time_differences, dtype=np.float64)

    if differences.size < 10:
        raise ValueError("Too few time differences for a reliable Gaussian fit.")

    counts, edges = np.histogram(differences, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2

    peak_index = int(np.argmax(counts))
    peak_position = centers[peak_index]

    mask = np.abs(centers - peak_position) <= fit_half_width
    x_fit = centers[mask]
    y_fit = counts[mask]

    if len(x_fit) < 5 or np.max(y_fit) <= 0:
        raise ValueError("Not enough histogram bins around the peak for fitting. Increase fit_half_width.")

    bin_width = edges[1] - edges[0]
    initial_sigma = max(bin_width * 2, fit_half_width / 10)

    p0 = [
        max(float(np.max(y_fit) - np.min(y_fit)), 1.0),
        float(peak_position),
        float(initial_sigma),
        float(max(np.min(y_fit), 0.0))
    ]

    parameters, covariance = curve_fit(
        gaussian,
        x_fit,
        y_fit,
        p0=p0,
        bounds=(
            [0, peak_position - fit_half_width, bin_width / 2, 0],
            [np.inf, peak_position + fit_half_width, fit_half_width * 2, np.inf]
        ),
        maxfev=20_000
    )

    amplitude, mu, sigma, offset = parameters
    return float(mu), abs(float(sigma)), centers, counts, parameters, covariance


def _count_coincidences(tags_1, tags_2, tags_3,
                        mu_T, sigma_T, mu_R, sigma_R,
                        window_scale=3.0):
    """
    Count coincidences in windows defined in seconds: [mu - w*sigma, mu + w*sigma].
    """
    detector_1 = np.sort(np.asarray(tags_1, dtype=np.float64))
    detector_2 = np.sort(np.asarray(tags_2, dtype=np.float64))
    heralds = np.sort(np.asarray(tags_3, dtype=np.float64))

    N1 = len(heralds)
    Nt = Nr = Nc = 0

    if N1 == 0:
        return N1, Nt, Nr, Nc

    T_low = mu_T - window_scale * sigma_T
    T_high = mu_T + window_scale * sigma_T
    R_low = mu_R - window_scale * sigma_R
    R_high = mu_R + window_scale * sigma_R

    for herald in heralds:
        T_left = np.searchsorted(detector_1, herald + T_low, side="left")
        T_right = np.searchsorted(detector_1, herald + T_high, side="right")
        coincidence_T = T_right > T_left

        R_left = np.searchsorted(detector_2, herald + R_low, side="left")
        R_right = np.searchsorted(detector_2, herald + R_high, side="right")
        coincidence_R = R_right > R_left

        Nt += int(coincidence_T)
        Nr += int(coincidence_R)
        Nc += int(coincidence_T and coincidence_R)

    return N1, Nt, Nr, Nc


def coincidences(tags_1, tags_2, tags_3, mu_T, sigma_T, mu_R, sigma_R):
    """Point 2: Count coincidences using +/- 3 sigma windows."""
    return _count_coincidences(tags_1, tags_2, tags_3, mu_T, sigma_T, mu_R, sigma_R, window_scale=3.0)


def probabilities(N1, Nt, Nr, Nc):
    """Point 2: Compute p_t, p_r, p_c."""
    if N1 == 0:
        raise ValueError("N1 is zero: there are no herald events.")
    return Nt / N1, Nr / N1, Nc / N1


def anticorrelation(N1, Nt, Nr, Nc):
    """Point 2: Calculate alpha = (N1 * Nc) / (Nt * Nr)."""
    if N1 <= 0 or Nt <= 0 or Nr <= 0:
        raise ValueError("N1, Nt, and Nr must be positive to calculate alpha.")
    return (N1 * Nc) / (Nt * Nr)


def uncertainty_alpha(N1, Nt, Nr, Nc):
    """Point 2: Poisson uncertainty propagation for alpha."""
    if min(N1, Nt, Nr) <= 0 or Nc < 0:
        raise ValueError("Counts must be non-negative for uncertainty estimation.")
    if Nc == 0:
        return 0.0  # Se Nc è 0, l'incertezza non può essere propagata con questa formula
    alpha = anticorrelation(N1, Nt, Nr, Nc)
    return alpha * np.sqrt(1/N1 + 1/Nc + 1/Nt + 1/Nr)


def alpha_vs_window(tags_1, tags_2, tags_3, mu_T, sigma_T, mu_R, sigma_R, w_values):
    """Point 3: Calculate alpha as a function of window width multiplier w."""
    alpha_values = []
    for w in w_values:
        N1, Nt, Nr, Nc = _count_coincidences(
            tags_1, tags_2, tags_3, mu_T, sigma_T, mu_R, sigma_R, window_scale=w
        )
        if Nt > 0 and Nr > 0:
            alpha_values.append((N1 * Nc) / (Nt * Nr))
        else:
            alpha_values.append(np.nan)

    return np.asarray(alpha_values, dtype=np.float64)


def time_intervals(time_tags_sec):
    """Point 4: Differences between consecutive sorted clicks in seconds."""
    sorted_tags = np.sort(np.asarray(time_tags_sec, dtype=np.float64))
    return np.diff(sorted_tags)


def remove_afterpulses(time_tags_sec, threshold_sec):
    """
    Point 5: Remove afterpulses by enforcing a minimum time separation (in seconds).
    """
    sorted_tags = np.sort(np.asarray(time_tags_sec, dtype=np.float64))

    if len(sorted_tags) == 0:
        return np.array([], dtype=np.float64)

    if threshold_sec < 0:
        raise ValueError("threshold_sec must be non-negative.")

    filtered = [sorted_tags[0]]
    for tag in sorted_tags[1:]:
        if tag - filtered[-1] >= threshold_sec:
            filtered.append(tag)

    return np.asarray(filtered, dtype=np.float64)