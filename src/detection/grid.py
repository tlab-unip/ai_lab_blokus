import cv2
import os
import numpy as np
from ..types.tiles import Tile, SquareColor


def __get_dominant_color(
    pixels: np.ndarray,
    n_colors: int = 5,
) -> np.ndarray:
    """get one dominant color from pixels"""

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        200,
        0.1,
    )
    _, labels, palette = cv2.kmeans(
        np.float32(pixels.reshape(-1, 3)),
        n_colors,
        None,
        criteria,
        10,
        cv2.KMEANS_RANDOM_CENTERS,
    )
    _, counts = np.unique(labels, return_counts=True)
    dominant = palette[np.argmax(counts)]
    return dominant


def generate_grid(
    image: np.ndarray,
    size: tuple[int, int] = (20, 20),
) -> tuple[np.ndarray, np.ndarray]:
    """return a grid of passed-in size and a grid of original size"""
    width = image.shape[1]
    height = image.shape[0]
    grid_width = width // size[0]
    grid_height = height // size[1]

    img = image.copy()
    grid = np.zeros((size[1], size[0], 3), np.uint8)
    for i, y in enumerate(range(0, height, grid_height)):
        for j, x in enumerate(range(0, width, grid_width)):
            x1 = x + grid_width
            y1 = y + grid_height

            pixels = img[y:y1, x:x1]
            dominant = __get_dominant_color(pixels)
            grid[i, j] = dominant

            pixels[:] = dominant
            cv2.rectangle(
                img,
                pt1=(x, y),
                pt2=(x1, y1),
                color=(0, 0, 0),
            )

    return grid, img


def detect_colors(
    grid: np.ndarray,
):
    def mapper(elem):
        e = [bool(val) for val in elem]
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

    result = np.array([[mapper(elem) for elem in row] for row in grid])
    return result


def build_tiles(
    grid: np.ndarray,
):
    pass


if __name__ == "__main__":
    import argparse

    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", help="path to a normalized image", required=True)
    parser.add_argument("-o", help="output directory")
    args = parser.parse_args()

    image = cv2.imread(args.i)
    grid, img = generate_grid(image)
    if args.o != None:
        cv2.imwrite(os.path.join(args.o, "grid_mini.jpg"), grid)
        cv2.imwrite(os.path.join(args.o, "grid_full.jpg"), img)
