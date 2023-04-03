import numpy as np
from scipy.stats import t as student_variable
from scipy.stats import bootstrap


def single_confidence_interval(mean, std, n_samples, confidence_level):
    t_crit = np.abs(student_variable.ppf((1 - confidence_level) / 2, n_samples - 1))
    lower_bound = mean - t_crit * (std / np.sqrt(n_samples))
    upper_bound = mean + t_crit * (std / np.sqrt(n_samples))

    return lower_bound, upper_bound


def confidence_interval_student(means, stds, n_samples, confidence_level=0.95):
    confidence_intervals = np.zeros((2, len(means)))

    if n_samples == 1:
        confidence_intervals[0] = means
        confidence_intervals[1] = means
    else:
        for idx_iteration in range(len(means)):
            confidence_intervals[0, idx_iteration], confidence_intervals[1, idx_iteration] = single_confidence_interval(
                means[idx_iteration], stds[idx_iteration], n_samples, confidence_level
            )

    return confidence_intervals


def confidence_interval_bootstrap(scores: np.ndarray) -> np.ndarray:
    confidence_intervals = np.zeros((2, scores.shape[1]))

    if scores.shape[0] == 1:
        confidence_intervals[0] = scores
        confidence_intervals[1] = scores
    else:
        for idx_iteration in range(scores.shape[1]):
            confidence_intervals[0, idx_iteration], confidence_intervals[1, idx_iteration] = bootstrap(
                (scores[:, idx_iteration],), np.mean
            ).confidence_interval

    return confidence_intervals
