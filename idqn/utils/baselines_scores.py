from typing import Dict, List
import numpy as np


def get_baselines_scores(baselines: List[int], games: List[int]) -> Dict:
    with open("baselines_scores/atari_200_iters_scores_plus_dopamine.npy", "rb") as f:
        atari_200m_scores_ = np.load(f, allow_pickle=True)
        atari_200m_scores = atari_200m_scores_.tolist()

    baselines_scores = {}

    for baseline in baselines:
        baselines_scores[baseline] = {}
        for game in games:
            baselines_scores[baseline][game] = atari_200m_scores[baseline][game]

    return baselines_scores
