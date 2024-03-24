import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


def count_samples(
    dimension_one: np.ndarray,
    dimension_two: np.ndarray,
    discrete_dim_one_boxes: np.ndarray,
    discrete_dim_two_boxes: np.ndarray,
    rewards: np.ndarray,
) -> tuple:
    # for each element of dimension one, get the index where it is located in the discrete dimension.
    dimension_one = np.array(dimension_one).reshape(-1)
    indexes_dim_one_boxes = np.searchsorted(discrete_dim_one_boxes, dimension_one) - 1

    # for each element of dimension two, get the index where it is located in the discrete dimension.
    dimension_two = np.array(dimension_two).reshape(-1)
    indexes_dim_two_boxes = np.searchsorted(discrete_dim_two_boxes, dimension_two) - 1

    # only count the element pairs that are in the boxes
    dim_one_inside_boxes = np.logical_and(
        dimension_one >= discrete_dim_one_boxes[0], dimension_one <= discrete_dim_one_boxes[-1]
    )
    dim_two_inside_boxes = np.logical_and(
        dimension_two >= discrete_dim_two_boxes[0], dimension_two <= discrete_dim_two_boxes[-1]
    )
    dimensions_inside_boxes = np.logical_and(dim_one_inside_boxes, dim_two_inside_boxes)

    pruned_rewards = rewards[dimensions_inside_boxes]

    samples_count = np.zeros((len(discrete_dim_one_boxes) - 1, len(discrete_dim_two_boxes) - 1))
    rewards_count = np.zeros((len(discrete_dim_one_boxes) - 1, len(discrete_dim_two_boxes) - 1))

    indexes_dim = np.vstack(
        (indexes_dim_one_boxes[dimensions_inside_boxes], indexes_dim_two_boxes[dimensions_inside_boxes])
    ).T

    for idx_in_list, (idx_dim_one, idx_dim_two) in enumerate(tqdm(indexes_dim)):
        samples_count[idx_dim_one, idx_dim_two] += 1
        rewards_count[idx_dim_one, idx_dim_two] += pruned_rewards[idx_in_list]

    return samples_count, (~dimensions_inside_boxes).sum(), rewards_count


class TwoDimesionsMesh:
    def __init__(
        self, dimension_one, dimension_two, sleeping_time: float, axis_equal: bool = True, zero_centered: bool = False
    ) -> None:
        self.dimension_one = dimension_one
        self.dimension_two = dimension_two
        self.grid_dimension_one, self.grid_dimension_two = np.meshgrid(self.dimension_one, self.dimension_two)

        self.sleeping_time = sleeping_time
        self.axis_equal = axis_equal
        self.zero_centered = zero_centered

        self.values = np.zeros((len(self.dimension_one), len(self.dimension_two)))

    def set_values(self, values: np.ndarray, zeros_to_nan: bool = False) -> None:
        assert values.shape == (
            len(self.dimension_one),
            len(self.dimension_two),
        ), f"given shape values: {values.shape} don't match with environment values: {(len(self.dimension_one), len(self.dimension_two))}"

        self.values = values
        if zeros_to_nan:
            self.values = np.where(self.values == 0, np.nan, self.values)

    def show(
        self,
        title: str = "",
        xlabel: str = "States",
        ylabel: str = "Actions",
        ticks_freq: int = 1,
    ) -> None:
        fig, ax = plt.subplots(figsize=(5.7, 5))
        plt.rc("font", size=18)
        plt.rc("lines", linewidth=3)

        if self.zero_centered:
            abs_max = np.max(np.abs(self.values))
            kwargs = {"cmap": "PRGn", "vmin": -abs_max, "vmax": abs_max}
        else:
            kwargs = {}

        colors = ax.pcolormesh(
            self.grid_dimension_one, self.grid_dimension_two, self.values.T, shading="nearest", **kwargs
        )

        ax.set_xticks(self.dimension_one[::ticks_freq])
        ax.set_xticklabels(np.around(self.dimension_one[::ticks_freq], 1), rotation="vertical")
        ax.set_xlim(self.dimension_one[0], self.dimension_one[-1])
        ax.set_xlabel(xlabel)

        ax.set_yticks(self.dimension_two[::ticks_freq])
        ax.set_yticklabels(np.around(self.dimension_two[::ticks_freq], 1))
        ax.set_ylim(self.dimension_two[0], self.dimension_two[-1])
        ax.set_ylabel(ylabel)

        if self.axis_equal:
            ax.set_aspect("equal", "box")
        if title != "":
            ax.set_title(title)

        fig.colorbar(colors, ax=ax)
        fig.tight_layout()
        fig.canvas.draw()
        plt.show()
