from typing import Annotated, TypeAlias

from numpy import float64
from numpy.typing import NDArray

Grid2D: TypeAlias = Annotated[NDArray[float64], "2D shape (nx, ny)"]
