from typing import Dict, Tuple, List
from functools import partial
import numpy as np
import scipy
import arch.bootstrap as arch_bs

from experiments.atari import ALL_HUMAN_SCORES, ALL_RANDOM_SCORES


# Taken without any change from https://github.com/google-research/rliable.
class StratifiedBootstrap(arch_bs.IIDBootstrap):
    """Bootstrap using stratified resampling.
    Supports numpy arrays. Data returned has the same type as the input data.
    Data entered using keyword arguments is directly accessibly as an attribute.
    To ensure a reproducible bootstrap, you must set the `random_state`
    attribute after the bootstrap has been created. See the example below.
    Note that `random_state` is a reserved keyword and any variable
    passed using this keyword must be an instance of `RandomState`.
    Examples
    --------
    Data can be accessed in a number of ways.  Positional data is retained in
    the same order as it was entered when the bootstrap was initialized.
    Keyword data is available both as an attribute or using a dictionary syntax
    on kw_data.
    >>> from rliable.library import StratifiedBootstrap
    >>> from numpy.random import standard_normal
    >>> x = standard_normal((5, 50))
    >>> bs = StratifiedBootstrap(x)
    >>> for data in bs.bootstrap(100):
    ...     bs_x = data[0][0]
    >>> bs.conf_int(np.mean, method='percentile', reps=50000)  # 95% CIs for mean
    Set the random_state if reproducibility is required.
    >>> from numpy.random import RandomState
    >>> rs = RandomState(1234)
    >>> bs = StratifiedBootstrap(x, random_state=rs)
    See also: `arch.bootstrap.IIDBootstrap`
    Attributes:
      data: tuple, Two-element tuple with the pos_data in the first position and
        kw_data in the second (pos_data, kw_data). Derived from `IIDBootstrap`.
      pos_data: tuple, Tuple containing the positional arguments (in the order
        entered). Derived from `IIDBootstrap`.
      kw_data: dict, Dictionary containing the keyword arguments. Derived from
        `IIDBootstrap`.
    """

    _name = "Stratified Bootstrap"

    def __init__(
        self,
        *args: np.ndarray,
        random_state=None,
        task_bootstrap: bool = False,
        **kwargs: np.ndarray,
    ) -> None:
        """Initializes StratifiedBootstrap.
        Args:
          *args: Positional arguments to bootstrap. Typically used for the
            performance on a suite of tasks with multiple runs/episodes. The inputs
            are assumed to be of the shape `(num_runs, num_tasks, ..)`.
          random_state: If specified, ensures reproducibility in uncertainty
            estimates.
          task_bootstrap: Whether to perform bootstrapping (a) over runs or (b) over
            both runs and tasks. Defaults to False which corresponds to (a). (a)
            captures the statistical uncertainty in the aggregate performance if the
            experiment is repeated using a different set of runs (e.g., changing
            seeds) on the same set of tasks. (b) captures the sensitivity of the
            aggregate performance to a given task and provides the performance
            estimate if we had used a larger unknown population of tasks.
          **kwargs: Keyword arguments, passed directly to `IIDBootstrap`.
        """

        super().__init__(*args, random_state=random_state, **kwargs)
        self._args_shape = args[0].shape
        self._num_tasks = self._args_shape[1]
        self._parameters = [self._num_tasks, task_bootstrap]
        self._task_bootstrap = task_bootstrap
        self._strata_indices = self._get_strata_indices()

    def _get_strata_indices(self) -> List[np.ndarray]:
        """Samples partial indices for bootstrap resamples.
        Returns:
          A list of arrays of size N x 1 x 1 x .., 1 x M x 1 x ..,
          1 x 1 x L x .. and so on, where the `args_shape` is `N x M x L x ..`.
        """
        ogrid_indices = tuple(slice(x) for x in (0, *self._args_shape[1:]))
        strata_indices = np.ogrid[ogrid_indices]
        return strata_indices[1:]

    def update_indices(
        self,
    ) -> Tuple[np.ndarray, ...]:
        """Selects the indices to sample from the bootstrap distribution."""
        # `self._num_items` corresponds to the number of runs
        indices = np.random.choice(self._num_items, self._args_shape, replace=True)
        if self._task_bootstrap:
            task_indices = np.random.choice(self._num_tasks, self._strata_indices[0].shape, replace=True)
            return (indices, task_indices, *self._strata_indices[1:])
        return (indices, *self._strata_indices)


def normalize_scores(scores: Dict) -> Dict:
    normalized_scores = {}

    for game, scores in scores.items():
        normalized_scores[game] = (scores - ALL_RANDOM_SCORES[game]) / (
            ALL_HUMAN_SCORES[game] - ALL_RANDOM_SCORES[game]
        )

    return normalized_scores


def stack_dictionary_values(dict_: Dict) -> np.ndarray:
    return np.stack([dict_[k] for k in dict_.keys()], axis=0)


def iqm(scores: np.ndarray) -> np.ndarray:
    return scipy.stats.trim_mean(scores.reshape((-1, scores.shape[-1])), proportiontocut=0.25, axis=0)


def compute_iqm_and_confidence_interval(scores: Dict, selected_epochs: np.ndarray, normalize: bool = True) -> Tuple:
    """
    scores: "algorithm": "game": 200 x n_seeds
    """
    if normalize:
        scores = normalize_scores(scores)
    stacked_scores = stack_dictionary_values(scores)
    # transpose to have n_seeds x n_games x n_selected_epochs: format required by StratifiedBootstrap
    selected_scores = stacked_scores[:, selected_epochs, :].transpose((2, 0, 1))

    # Take IQM over games and seeds.
    iqms = iqm(selected_scores)
    # Take the confidence interval over seeds only.
    iqms_confidence_interval = StratifiedBootstrap(selected_scores).conf_int(
        iqm, reps=2000, size=0.95, method="percentile"
    )

    return iqms, iqms_confidence_interval


@partial(np.vectorize, excluded=[0])
def scores_distribution(scores: np.ndarray, tau: float) -> float:
    """Evaluates how many `scores` are above `tau` averaged across all runs."""
    return np.mean(scores > tau)


def compute_performance_profile_and_confidence_interval(scores: Dict, taus: np.ndarray) -> Tuple:
    """
    scores: "algorithm": "game": 200 x n_seeds
    """
    normalized_scores = normalize_scores(scores)
    stacked_scores = stack_dictionary_values(normalized_scores)
    # transpose to have n_seeds x n_games : format required by StratifiedBootstrap
    selected_scores = stacked_scores[:, 199, :].transpose()

    # Take the score distribution over games and seeds.
    performance_profile = scores_distribution(selected_scores, taus)
    # Take the confidence interval over seeds only.
    performance_profile_confidence_interval = StratifiedBootstrap(selected_scores).conf_int(
        partial(scores_distribution, tau=taus), reps=2000, size=0.95, method="percentile"
    )

    return performance_profile, performance_profile_confidence_interval
