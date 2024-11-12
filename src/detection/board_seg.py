import cv2
import os
import numpy as np
import supervision as sv
from collections.abc import Generator


def __get_contour_from_mask(mask: np.ndarray):
    contours, _ = cv2.findContours(
        np.array(mask * 255, dtype=np.uint8),
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_NONE,
    )
    return max(contours, key=cv2.contourArea)


def __crop_and_transform_rect(image, rect: cv2.RotatedRect):
    box = cv2.boxPoints(rect)
    box = np.intp(box)
    width, height = int(rect[1][0]), int(rect[1][1])
    src_points = box.astype("float32")
    dst_points = np.array(
        [
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1],
        ],
        dtype="float32",
    )
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    cropped = cv2.warpPerspective(image, M, (width, height))
    return cropped


def __lines_detection(image) -> list:
    edges = cv2.Canny(image, 40, 150, apertureSize=3)
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=10,
        minLineLength=1000,
        maxLineGap=50,
    )
    lines = [
        [
            (points[0][0], points[0][1]),
            (points[0][2], points[0][3]),
        ]
        for points in lines
    ]
    return lines


def __get_quad_from_lines(image, lines):
    img = image.copy()
    lines_l2r = sorted(lines, key=lambda x: (x[0][0] + x[1][0]))
    leftmost = lines_l2r[0]
    rightmost = lines_l2r[-1]
    cv2.line(img, leftmost[0], leftmost[1], (255, 0, 0), 20)
    cv2.line(img, rightmost[0], rightmost[1], (0, 0, 255), 20)

    lines_t2b = sorted(lines, key=lambda x: (x[0][1] + x[1][1]))
    topmost = lines_t2b[0]
    bottommost = lines_t2b[-1]
    cv2.line(img, topmost[0], topmost[1], (255, 0, 0), 20)
    cv2.line(img, bottommost[0], bottommost[1], (0, 0, 255), 20)

    return img


def seg_using_inference(
    image,
    model_id: str = "blokus_board_seg/1",
) -> sv.Detections:
    import inference

    # os.environ["ROBOFLOW_API_KEY"] = ""
    model = inference.get_model(model_id)
    result = model.infer(image)[0]
    detections = sv.Detections.from_inference(result)
    return detections


def seg_using_yolo(
    image,
    model_path: str = "models/board_seg_v1.pt",
) -> sv.Detections:
    """predict a mask from an image, using local model

    Args:
        image (_type_): image to be processed
        model_path (str, optional): path to a local model. Defaults to "models/board_seg_v1.pt".
        show_detections (bool, optional): _description_. Defaults to False.

    Yields:
        Generator[np.ndarray]: mask of boolean type value
    """

    from ultralytics import YOLO

    model = YOLO(model=model_path)
    result = model.predict(image, task="segment")[0]
    detections = sv.Detections.from_ultralytics(result)
    return detections


def board_seg_by_cv(image):
    img = image.copy()
    lines = __lines_detection(img)
    for line in lines:
        cv2.line(img, line[0], line[1], (0, 255, 0), 10)
    img = __get_quad_from_lines(img, lines)

    return img


def board_seg_by_model(image):
    img = image.copy()
    detections = seg_using_yolo(img)
    mask = list(detections)[0][1]
    contour = __get_contour_from_mask(mask)

    # only for no perspective
    rect = cv2.minAreaRect(contour)
    img = __crop_and_transform_rect(img, rect)
    return img


if __name__ == "__main__":
    import argparse

    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", help="input image path", required=True)
    parser.add_argument("-o", help="output image path")
    args = parser.parse_args()

    image = cv2.imread(args.i)
    image = board_seg_by_cv(image)
    if args.o != None:
        cv2.imwrite(args.o, image)
