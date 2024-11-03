import cv2
import os
import numpy as np
from ..types.tiles import Tile, SquareColor


def __get_dominant_color(
    pixels: np.ndarray,
    n_colors: int = 5,
) -> np.ndarray:
    """get dominant color from pixels

    Args:
        pixels (np.ndarray): pixels to be processed
        n_colors (int, optional): number of classified colors. Defaults to 5.

    Returns:
        np.ndarray: one dominant color (GBR)
    """

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
    output_dir: str | None = None,
) -> np.ndarray:
    """generate grid from an image

    Args:
        image (np.ndarray): image to be processed
        size (tuple[int, int], optional): size of the grid, (width, height). Defaults to (20, 20).
        output_dir (str | None, optional): output path of processed image. Defaults to None.

    Returns:
        np.ndarray: a grid of colors
    """

    width = image.shape[1]
    height = image.shape[0]
    grid_width = width // size[0]
    grid_height = height // size[1]

    grid = np.zeros((size[1], size[0], 3), np.uint8)
    for i, y in enumerate(range(0, height, grid_height)):
        for j, x in enumerate(range(0, width, grid_width)):
            x1 = x + grid_width
            y1 = y + grid_height

            pixels = image[y:y1, x:x1]
            dominant = __get_dominant_color(pixels)
            grid[i, j] = dominant

            if output_dir != None:
                pixels[:] = dominant
                cv2.rectangle(
                    image,
                    pt1=(x, y),
                    pt2=(x1, y1),
                    color=(0, 0, 0),
                )

    if output_dir != None:
        cv2.imwrite(os.path.join(output_dir, "grid_full.jpg"), image)
        cv2.imwrite(os.path.join(output_dir, "grid_mini.jpg"), grid)
    return grid


def detect_colors(
    grid: np.ndarray,
):
    pass


def build_tiles(
    grid: np.ndarray,
):
    pass


if __name__ == "__main__":
    import argparse

    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", help="path to a normalized image", required=True)
    args = parser.parse_args()

    image = cv2.imread(args.i)
    generate_grid(image, output_dir="data/outputs")
