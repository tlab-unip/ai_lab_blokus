import cv2
from enum import Enum


class SquareColor(Enum):
    EMPTY = 0
    RED = 1
    GREEN = 2
    BLUE = 3
    YELLOW = 4


class TileShape:
    def __init__(self):
        pass


class Tile:
    def __init__(
        self,
        color: SquareColor,
        shape: TileShape,
        n_squares: int,
    ):
        self.color = color
        self.shape = shape
        self.n_squares = n_squares
