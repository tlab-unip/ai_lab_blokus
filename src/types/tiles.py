from enum import Enum
from typing import Self

import numpy as np


class SquareColor(Enum):
    EMPTY = 0
    RED = 1
    GREEN = 2
    BLUE = 3
    YELLOW = 4

    def fromColor(
        color: tuple[int, int, int],
        encoding: str = "RGB",
    ) -> Self:
        encoding = encoding.upper()
        if encoding not in ["RGB", "GBR", "BGR"]:
            raise ValueError("Unsupported encoding type")

        r, g, b = color
        if encoding == "GBR":
            g, b, r = color
        elif encoding == "BGR":
            b, g, r = color

        if (r, g, b) == (255, 0, 0):
            return SquareColor.RED
        elif (r, g, b) == (0, 255, 0):
            return SquareColor.GREEN
        elif (r, g, b) == (255, 255, 0):
            return SquareColor.YELLOW
        elif (r, g, b) == (0, 0, 255):
            return SquareColor.BLUE
        else:
            return SquareColor.EMPTY

    def toColor(self, encoding: str = "RGB") -> tuple[int, int, int]:
        encoding = encoding.upper()
        if encoding not in ["RGB", "GBR", "BGR"]:
            raise ValueError("Unsupported encoding type")

        rgb = {
            SquareColor.RED: (255, 0, 0),
            SquareColor.GREEN: (0, 255, 0),
            SquareColor.YELLOW: (255, 255, 0),
            SquareColor.BLUE: (0, 0, 255),
            SquareColor.EMPTY: (255, 255, 255),
        }

        r, g, b = rgb[self]
        if encoding == "RGB":
            return (r, g, b)
        elif encoding == "GBR":
            return (g, b, r)
        elif encoding == "BGR":
            return (b, g, r)

    def __repr__(self):
        return self.name

    def __str__(self):
        return self.name


def decode_bitboard(
    bitboard: int,
    shape=(20, 20),
) -> np.ndarray:
    """decode bitboard into 2d array, from the highest bit"""
    board = np.zeros(shape, dtype=int)
    total_bits = shape[0] * shape[1]
    for i in range(shape[0]):
        for j in range(shape[1]):
            bit_index = total_bits - (i * shape[1] + j) - 1
            if bitboard & (1 << bit_index):
                board[i, j] = 1
    return board


def encode_bitboard(
    board: np.ndarray,
) -> int:
    """encode 2d array into bitboard, from the highest bit"""
    height, width = board.shape
    bitboard = 0
    total_bits = width * height
    for i in range(height):
        for j in range(width):
            bit_index = total_bits - (i * width + j) - 1
            if board[i, j] == 1:
                bitboard |= 1 << bit_index
    return bitboard


class PieceType(Enum):
    I1 = (
        """
        1
        """,
    )
    I2 = (
        """
        11
        """,
    )

    V3 = (
        """
        11
        10
        """,
    )
    I3 = (
        """
        111
        """,
    )

    T4 = (
        """
        111
        010
        """,
    )
    O4 = (
        """
        11
        11
        """,
    )
    L4 = (
        """
        111
        100
        """,
    )
    I4 = (
        """
        1111
        """,
    )
    Z4 = (
        """
        110
        011
        """,
    )

    F5 = (
        """
        111
        110
        """,
    )
    X5 = (
        """
        010
        111
        010
        """,
    )
    P5 = (
        """
        111
        110
        """,
    )
    W5 = (
        """
        110
        011
        001
        """,
    )
    Z5 = (
        """
        110
        010
        011
        """,
    )
    Y5 = (
        """
        1111
        0100
        """,
    )
    L5 = (
        """
        1111
        1000
        """,
    )
    U5 = (
        """
        111
        101
        """,
    )
    T5 = (
        """
        111
        010
        010
        """,
    )
    V5 = (
        """
        111
        100
        100
        """,
    )
    N5 = (
        """
        1100
        0111
        """,
    )
    I5 = (
        """
        11111
        """,
    )

    def decode(
        self,
        translation=(0, 0),
        rotation=0,
        flipping=False,
        size=(20, 20),
    ) -> np.ndarray:
        lines = self.value[0].strip().split("\n")
        height = len(lines)
        width = len(lines[0])
        board = np.zeros((height, width), dtype=int)
        for i, line in enumerate(lines):
            for j, char in enumerate(line.strip()):
                if char == "1":
                    board[i, j] = 1

        if flipping:
            board = np.fliplr(board)

        for _ in range(rotation % 4):
            board = np.rot90(board, -1)

        height, width = board.shape
        translated_board = np.zeros(size, dtype=int)
        trans_x, trans_y = translation
        if trans_x + height <= size[0] and trans_y + width <= size[1]:
            translated_board[trans_x : trans_x + height, trans_y : trans_y + width] = (
                board
            )
        return translated_board
