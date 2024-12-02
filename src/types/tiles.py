from enum import Enum
from typing import Self


class SquareColor(Enum):
    EMPTY = 0
    RED = 1
    GREEN = 2
    BLUE = 3
    YELLOW = 4

    def fromGBR(
        color: tuple[int, int, int],
    ) -> Self:
        e = [bool(val) for val in color]
        match e:
            case [False, False, True]:
                return SquareColor.RED
            case [False, True, False]:
                return SquareColor.GREEN
            case [False, True, True]:
                return SquareColor.YELLOW
            case [True, False, False]:
                return SquareColor.BLUE
            case _:
                return SquareColor.EMPTY

    def toGBR(value) -> tuple[int, int, int]:
        if not isinstance(value, SquareColor):
            value = SquareColor(int(value))
        match value:
            case SquareColor.RED:
                return (0, 0, 255)
            case SquareColor.GREEN:
                return (0, 255, 0)
            case SquareColor.YELLOW:
                return (0, 255, 255)
            case SquareColor.BLUE:
                return (255, 0, 0)
            case _:
                return (255, 255, 255)

    def __repr__(self):
        return self.name

    def __str__(self):
        return self.name


class PieceType(Enum):
    I1 = 1

    I2 = 2

    V3 = 3
    I3 = 9

    T4 = 10
    O4 = 16
    L4 = 22
    I4 = 28
    Z4 = 34

    F5 = 35
    X5 = 41
    P5 = 47
    W5 = 53
    Z5 = 59
    Y5 = 65
    L5 = 71
    U5 = 77
    T5 = 83
    V5 = 89
    N5 = 95
    I5 = 101

    def fits_piece(points: list) -> Self:
        """Finds suitable piece by rotating and flipping"""
        for typ in Self.value:
            pass
        return
