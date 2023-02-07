import matplotlib.pyplot as plt
import numpy as np


class TwoDimesionsMesh:
    def __init__(self, dimension_one, dimension_two) -> None:
        self.dimension_one = dimension_one
        self.dimension_two = dimension_two
        self.grid_dimension_one, self.grid_dimension_two = np.meshgrid(self.dimension_one, self.dimension_two)
        self.ticks_freq = 2

        self.values = np.zeros((len(self.dimension_one), len(self.dimension_two)))

    def set_values(self, values: np.ndarray) -> None:
        assert values.shape == (
            len(self.dimension_one),
            len(self.dimension_two),
        ), f"given shape values: {values.shape} don't match with the given values: {(len(self.dimension_one), len(self.dimension_two))}"

        self.values = values

    def show(
        self,
        title: str,
        show_color_bar: bool = True,
        xlabel: str = "x",
        ylabel: str = "v",
    ) -> None:
        if show_color_bar:
            fig, ax = plt.subplots(figsize=(7, 5))
        else:
            fig, ax = plt.subplots(figsize=(5.7, 5))

        plt.rc("font", size=18)
        plt.rc("lines", linewidth=3)

        # colors are zero centered
        abs_max = np.max(np.abs(self.values))
        kwargs = {"cmap": "PRGn", "vmin": -abs_max, "vmax": abs_max}

        colors = ax.pcolormesh(
            self.grid_dimension_one, self.grid_dimension_two, self.values.T, shading="nearest", **kwargs
        )

        ax.set_xticks(self.dimension_one[:: self.ticks_freq])
        ax.set_xticklabels(np.around(self.dimension_one[:: self.ticks_freq], 1), rotation="vertical")
        ax.set_xlim(self.dimension_one[0], self.dimension_one[-1])
        ax.set_xlabel(xlabel)

        ax.set_yticks(self.dimension_two[:: self.ticks_freq])
        ax.set_yticklabels(np.around(self.dimension_two[:: self.ticks_freq], 1))
        ax.set_ylim(self.dimension_two[0], self.dimension_two[-1])
        ax.set_ylabel(ylabel)

        ax.set_title(title)

        if show_color_bar:
            fig.colorbar(colors, ax=ax)
            fig.tight_layout()
        fig.canvas.draw()
