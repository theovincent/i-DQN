from typing import Dict
import json
import numpy as np


def get_dopamine_scores(game: str, baseline: str) -> Dict:
    game_scores = json.load(open(f"../../../dopamine/baselines/atari/data/{game.lower()}.json"))

    scores = {}
    for idx_seed in range(5):
        scores[f"{game}_{idx_seed}"] = np.zeros(200) * np.nan
    seeds = np.zeros(200, dtype=int)

    for point in game_scores:
        if point["Agent"] == baseline:
            scores[f"{game}_{seeds[point['Iteration']]}"][point["Iteration"]] = point["Value"]
            seeds[point["Iteration"]] += 1

    return scores
