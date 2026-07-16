from pathlib import Path
from typing import Protocol

from numpy.typing import NDArray


class ModelAdapter(Protocol):
    def fit(self, X: NDArray, y: NDArray) -> "ModelAdapter": ...

    def predict(self, X: NDArray) -> NDArray: ...

    def save(self, path: Path) -> None: ...
