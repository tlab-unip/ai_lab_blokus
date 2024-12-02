import cv2
import os
import numpy as np
import numpy.typing as npt
from ..types.tiles import SquareColor


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
    """return a grid of color enum and a grid of original size"""
    width = image.shape[1]
    height = image.shape[0]
    grid_width = width // size[0]
    grid_height = height // size[1]

    img = image.copy()
    grid = np.zeros((size[1], size[0]), SquareColor)
    for i, y in enumerate(range(0, height, grid_height)):
        for j, x in enumerate(range(0, width, grid_width)):
            x1 = x + grid_width
            y1 = y + grid_height

            pixels = img[y:y1, x:x1]
            dominant = __get_dominant_color(pixels)
            grid[i, j] = SquareColor.fromGBR(dominant)
    return grid


def generate_image(
    grid: npt.NDArray,
    size: tuple[int, int] = (200, 200),
):
    width = size[0]
    height = size[1]
    grid_width = width // grid.shape[1]
    grid_height = width // grid.shape[0]

    img = np.empty((*size, 3), np.uint8)
    for i, y in enumerate(range(0, height, grid_height)):
        for j, x in enumerate(range(0, width, grid_width)):
            x1 = x + grid_width
            y1 = y + grid_height

            pixels = img[y:y1, x:x1]
            pixels[:] = SquareColor.toGBR(grid[i, j])
            cv2.rectangle(
                img,
                pt1=(x, y),
                pt2=(x1, y1),
                color=(0, 0, 0),
            )
    return img


def generate_bitboard(grid):
    pass


if __name__ == "__main__":
    import argparse

    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", help="path to a normalized image", required=True)
    parser.add_argument("-o", help="output directory")
    args = parser.parse_args()

    image = cv2.imread(args.i)
    color_grid, img_grid = generate_grid(image)
    if args.o != None:
        cv2.imwrite(os.path.join(args.o, "grid.jpg"), img_grid)
