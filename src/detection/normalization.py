import cv2
import os
import numpy as np
from .board_seg import apply_local_model


def __get_mask_contour(mask: np.ndarray):
    """get contour from a mask

    Args:
        mask (np.ndarray): mask, value of boolean type

    Returns:
        _type_: a contour with max area from the mask
    """

    contours, _ = cv2.findContours(
        np.array(mask * 255, dtype=np.uint8),
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_NONE,
    )
    return max(contours, key=cv2.contourArea)


def __crop_and_transform(image, rect: cv2.RotatedRect):
    """crop a image by a rotated rect and do transformation

    Args:
        image (_type_): image to be processed
        rect (cv2.RotatedRect): _description_

    Returns:
        _type_: transformed image
    """

    box = cv2.boxPoints(rect)
    box = np.intp(box)
    width, height = int(rect[1][0]), int(rect[1][1])
    src_points = box.astype("float32")
    dst_points = np.array(
        [
            [0, height - 1],
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
        ],
        dtype="float32",
    )
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    cropped = cv2.warpPerspective(image, M, (width, height))
    return cropped


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


def __color_mapping(image: np.ndarray):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    red_thres = cv2.threshold(a, 167, 255, cv2.THRESH_BINARY)[1]
    green_thres = cv2.threshold(a, 97, 255, cv2.THRESH_BINARY_INV)[1]
    yellow_thres = cv2.threshold(b, 167, 255, cv2.THRESH_BINARY)[1]
    blue_thres = cv2.threshold(b, 97, 255, cv2.THRESH_BINARY_INV)[1]

    result = np.empty_like(image)
    result[:] = np.uint8([255, 255, 255])
    result[red_thres > 0] = np.uint8([0, 0, 255])
    result[green_thres > 0] = np.uint8([0, 255, 0])
    result[yellow_thres > 0] = np.uint8([0, 255, 255])
    result[blue_thres > 0] = np.uint8([255, 0, 0])
    return result


def normalization(
    image,
    size=(200, 200),
):
    """apply normalization process on a image

    Args:
        image (_type_): image to be processed
        size (tuple, optional): expected size. Defaults to (200, 200).

    Returns:
        _type_: normalized image
    """
    img = cv2.bilateralFilter(image, 9, 75, 75)
    mask = next(apply_local_model(img))
    contour = __get_mask_contour(mask)
    rect = cv2.minAreaRect(contour)

    img = __crop_and_transform(img, rect)
    img = cv2.resize(img, size)
    img = __color_correction(img)
    img = __color_mapping(img)
    return img


def batch_normalization(
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

        normalized_image = normalization(image, size=size)
        cv2.imwrite(output_path, normalized_image)


if __name__ == "__main__":
    import argparse

    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", default="data/inputs", help="input directory")
    parser.add_argument("-o", default="data/outputs", help="output directory")
    args = parser.parse_args()
    batch_normalization(args.i, args.o)
