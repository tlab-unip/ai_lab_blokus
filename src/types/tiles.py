from enum import Enum
from functools import lru_cache
from typing import Any, Dict, Self, Tuple
from joblib import Memory

import numpy as np

memory = Memory("cache/", verbose=0)


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
        vals = [val > 200 for val in color]
        r, g, b = vals
        if encoding == "GBR":
            g, b, r = vals
        elif encoding == "BGR":
            b, g, r = vals

        if (r, g, b) == (True, False, False):
            return SquareColor.RED
        elif (r, g, b) == (False, True, False):
            return SquareColor.GREEN
        elif (r, g, b) == (True, True, False):
            return SquareColor.YELLOW
        elif (r, g, b) == (False, False, True):
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

    def __repr__(self):
        return self.name

    @staticmethod
    @memory.cache
    def get_cache(shape=(20, 20)) -> Dict[Any, set[int]]:
        """get cached transformed pieces, indexed by [PieceType]"""
        cache = {}
        piece_types = sorted(PieceType, key=lambda p: -np.sum(p.decode()))
        for piece_type in piece_types:
            cache[piece_type] = set()
            for flipping in [False, True]:
                for rotation in range(4):
                    for i in range(shape[0]):
                        for j in range(shape[1]):
                            translation = (i, j)
                            bitboard = piece_type.transform_and_padding(
                                translation=translation,
                                rotation=rotation,
                                flipping=flipping,
                            )
                            if bitboard != 0:
                                cache[piece_type].add(bitboard)
        return cache

    @lru_cache(maxsize=None)
    def decode(
        self,
        rotation=0,
        flipping=False,
    ) -> np.ndarray | None:
        lines = self.value[0].strip().split("\n")
        height = len(lines)
        width = len(lines[0])
        piece = np.zeros((height, width), dtype=int)
        for i, line in enumerate(lines):
            for j, char in enumerate(line.strip()):
                if char == "1":
                    piece[i, j] = 1

        if flipping:
            piece = np.fliplr(piece)

        for _ in range(rotation % 4):
            piece = np.rot90(piece, -1)

        return piece

    @lru_cache(maxsize=None)
    def transform_and_padding(
        self,
        translation=(0, 0),
        rotation=0,
        flipping=False,
        shape=(20, 20),
    ) -> int:
        board = self.decode(rotation=rotation, flipping=flipping)
        height, width = board.shape

        trans_x, trans_y = translation
        if trans_x + height > shape[0] or trans_y + width > shape[1]:
            return 0

        transformed = np.zeros(shape, dtype=int)
        transformed[trans_x : trans_x + height, trans_y : trans_y + width] = board
        return encode_bitboard(transformed)


def extract_pieces(
    bitboard: int,
    shape=(20, 20),
) -> list[PieceType]:
    """extract utilized pieces from a bitboard, assuming no duplicated pieces"""
    cache = PieceType.get_cache()

    used_pieces = []
    # remove pieces from larger ones
    piece_types = sorted(PieceType, key=lambda p: -np.sum(p.decode()))
    for piece_type in piece_types:
        for transformed in cache[piece_type]:
            if transformed != 0 and (bitboard & transformed) == transformed:
                used_pieces.append(piece_type)
                bitboard &= ~transformed
                break
    if bitboard != 0:
        print("Warning: Invalid tiles detected")
    return used_pieces
