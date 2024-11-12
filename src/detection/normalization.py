import cv2
import os
import numpy as np
from .board_seg import board_seg_by_cv


def __color_correction(image):
    result = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    avg_a = np.average(result[:, :, 1])
    avg_b = np.average(result[:, :, 2])
    result[:, :, 1] = result[:, :, 1] - (
        (avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1
    )
    result[:, :, 2] = result[:, :, 2] - (
        (avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1
    )
    result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
    return result


def __color_mapping(
    image: np.ndarray,
    rgyb_thres=(167, 97, 167, 97),
):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    red_thres = cv2.threshold(a, rgyb_thres[0], 255, cv2.THRESH_BINARY)[1]
    green_thres = cv2.threshold(a, rgyb_thres[1], 255, cv2.THRESH_BINARY_INV)[1]
    yellow_thres = cv2.threshold(b, rgyb_thres[2], 255, cv2.THRESH_BINARY)[1]
    blue_thres = cv2.threshold(b, rgyb_thres[3], 255, cv2.THRESH_BINARY_INV)[1]

    result = np.empty_like(image)
    result[:] = np.uint8([255, 255, 255])
    result[red_thres > 0] = np.uint8([0, 0, 255])
    result[green_thres > 0] = np.uint8([0, 255, 0])
    result[yellow_thres > 0] = np.uint8([0, 255, 255])
    result[blue_thres > 0] = np.uint8([255, 0, 0])
    return result


def normalize(
    image,
    size=(200, 200),
    rgyb_thres=(167, 97, 167, 97),
):
    """apply normalization process on a image

    Args:
        image (_type_): image to be processed
        size (tuple, optional): expected size. Defaults to (200, 200).

    Returns:
        _type_: normalized image
    """
    img = image.copy()
    img = cv2.bilateralFilter(img, 9, 75, 75)
    img = board_seg_by_cv(img)
    img = cv2.resize(img, size)
    img = __color_correction(img)
    img = __color_mapping(img, rgyb_thres)
    return img


def batch_normalize(
    input_dir: str,
    output_dir: str,
    size: tuple[int, int] = (200, 200),
    max_count=20,
):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    exts = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")
    filenames = [f for f in os.listdir(input_dir) if f.endswith(exts)]
    for filename in filenames[:max_count]:
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        image = cv2.imread(input_path)
        if image is None:
            print(f"Error reading image {input_path}")
            continue

        normalized_image = normalize(image, size=size)
        cv2.imwrite(output_path, normalized_image)


if __name__ == "__main__":
    import argparse

    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", default="data/inputs", help="input directory")
    parser.add_argument("-o", default="data/outputs", help="output directory")
    args = parser.parse_args()
    batch_normalize(args.i, args.o)
