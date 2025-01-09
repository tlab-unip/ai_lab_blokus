from .board_seg import *
from .normalization import *
from .grid import *


def detect(
    raw_image,
    norm_size=(200, 200),
    rgyb_thres=(167, 97, 167, 97),
    grid_size=(20, 20),
):
    img = normalize(image=raw_image, size=norm_size, rgyb_thres=rgyb_thres)
    grid = generate_grid(image=img, size=grid_size)
    return grid
