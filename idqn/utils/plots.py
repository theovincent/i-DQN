from typing import Callable, Dict, Tuple
import numpy as np
import scipy

from experiments.atari import ALL_HUMAN_SCORES, ALL_RANDOM_SCORES


def single_confidence_interval(mean, std, n_samples, confidence_level):
    t_crit = np.abs(scipy.stats.t.ppf((1 - confidence_level) / 2, n_samples - 1))
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


def confidence_interval_bootstrap(scores: np.ndarray, func: Callable) -> np.ndarray:
    confidence_intervals = np.zeros((2, scores.shape[1]))

    if scores.shape[0] == 1:
        confidence_intervals[0] = scores
        confidence_intervals[1] = scores
    else:
        for idx_iteration in range(scores.shape[1]):
            confidence_intervals[0, idx_iteration], confidence_intervals[1, idx_iteration] = scipy.stats.bootstrap(
                (scores[:, idx_iteration],), func
            ).confidence_interval

    return confidence_intervals


def normalize_scores(scores: Dict) -> Dict:
    normalized_scores = {}

    for game_seed, scores in scores.items():
        game = game_seed.split("_")[0]
        normalized_scores[game_seed] = (scores - ALL_RANDOM_SCORES[game]) / (
            ALL_HUMAN_SCORES[game] - ALL_RANDOM_SCORES[game]
        )

    return normalized_scores


def stack_dictionary_values(dict_: Dict) -> np.ndarray:
    return np.stack([dict_[k] for k in dict_.keys()], axis=0)


def iqm(scores: np.ndarray) -> np.ndarray:
    return scipy.stats.trim_mean(scores, proportiontocut=0.25, axis=0)


def compute_iqm_and_confidence_interval(scores: Dict, selected_epochs: np.ndarray) -> Tuple:
    normalized_scores = normalize_scores(scores)
    stacked_scores = stack_dictionary_values(normalized_scores)

    iqms = iqm(stacked_scores[:, selected_epochs])
    iqms_confidence_interval = confidence_interval_bootstrap(stacked_scores[:, selected_epochs], iqm)

    return iqms, iqms_confidence_interval
