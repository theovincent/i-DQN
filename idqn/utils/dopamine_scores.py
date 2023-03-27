import json
import numpy as np


def get_dopamine_scores(game: str):
    breakout = json.load(open(f"../../../dopamine/baselines/atari/data/{game.lower()}.json"))

    scores = np.zeros((5, 200)) * np.nan
    seeds = np.zeros(200, dtype=int)

    for point in breakout:
        if point["Agent"] == "DQN (Adam + MSE in JAX)":
            scores[seeds[point["Iteration"]], point["Iteration"]] = point["Value"]
            seeds[point["Iteration"]] += 1

    return scores
