

from typing import Optional, Union, Any, Sequence
from numpy.typing import ArrayLike
import functools
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from KDEpy import FFTKDE
def bisection(array: Sequence[Any], value: Any) -> int:
    """Given an ``array`` , and given a ``value`` , returns an index j such that ``value`` is between array[j]
    and array[j+1]. ``array`` must be monotonic increasing. j=-1 or j=len(array) is returned
    to indicate that ``value`` is out of range below and above respectively.
    From https://stackoverflow.com/a/41856629"""
    n = len(array)
    if value < array[0]:
        return -1
    elif value > array[n - 1]:
        return n
    jl = 0  # Initialize lower
    ju = n - 1  # and upper limits.
    while ju - jl > 1:  # If we are not yet done,
        jm = (ju + jl) >> 1  # compute a midpoint with a bitshift
        if value >= array[jm]:
            jl = jm  # and replace either the lower limit
        else:
            ju = jm  # or the upper limit, as appropriate.
        # Repeat until the test condition is satisfied.
    if value == array[0]:  # edge cases at bottom
        return 0
    elif value == array[n - 1]:  # and top
        return n - 1
    else:
        return jl
    

class GetKDE:
    r"""
    This is badsed on the code from denseweight.
    But instead it only returns the dense probabilities

    """

    def __init__(
        self,
        alpha: float = 1.0,
        bandwidth: Optional[Union[float, str]] = None,
        eps: float = 1e-6,
    ):
        self.alpha = alpha
        self.bandwidth = bandwidth
        self.eps = eps

    def fit(self, y: ArrayLike, grid_points=4096) -> np.ndarray:

        if self.bandwidth is None:
            silverman_bandwidth = 1.06 * np.std(y) * np.power(len(y), (-1.0 / 5.0))
            self.bandwidth = silverman_bandwidth

        self.kernel = FFTKDE(bw=self.bandwidth).fit(y, weights=None)

        x, y_dens_grid = self.kernel.evaluate(grid_points)  # Default precision is 1024
        self.x = x

        # Min-Max Scale to 0-1 since pdf's can actually exceed 1
        # See: https://stats.stackexchange.com/questions/5819/kernel-density-estimate-takes-values-larger-than-1
        self.y_dens_grid = (
            MinMaxScaler().fit_transform(y_dens_grid.reshape(-1, 1)).flatten()
        )

        self.y_dens = np.vectorize(self.get_density)(y)
        return self.y_dens #self.weights

    def get_density(self, y: ArrayLike) -> np.ndarray:
        try:
            idx = bisection(self.x, y)
        except AttributeError:
            raise ValueError("Must call fit first!")
        try:
            dens = self.y_dens_grid[idx]
        except IndexError:
            if idx <= -1:
                idx = 0
            elif idx >= len(self.x):
                idx = len(self.x) - 1
            dens = self.y_dens_grid[idx]
        return dens
