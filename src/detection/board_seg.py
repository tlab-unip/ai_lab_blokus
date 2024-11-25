import cv2
import os
import numpy as np
import supervision as sv
from collections.abc import Generator


def __get_hull_from_mask(mask: np.ndarray):
    contours, _ = cv2.findContours(
        np.array(mask * 255, dtype=np.uint8),
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_NONE,
    )
    return cv2.convexHull(contours[0])


def __crop_and_transform(image, quad):
    src_points = np.float32(quad)
    rect = cv2.minAreaRect(src_points)
    box = cv2.boxPoints(rect)
    box = np.intp(box)
    width, height = int(rect[1][0]), int(rect[1][1])
    dst_points = np.array(
        [
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1],
            [0, 0],
        ],
        dtype="float32",
    )
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    cropped = cv2.warpPerspective(image, M, (width, height))
    return cropped


def __appx_best_fit_ngon(hull, n: int = 4) -> list[(int, int)]:
    """Modified from https://stackoverflow.com/a/74620323/21237436"""
    import sympy

    hull = np.array(hull).reshape((len(hull), 2))

    # to sympy land
    hull = [sympy.Point(*pt) for pt in hull]

    # run until we cut down to n vertices
    while len(hull) > n:
        best_candidate = None

        # for all edges in hull ( <edge_idx_1>, <edge_idx_2> ) ->
        for edge_idx_1 in range(len(hull)):
            edge_idx_2 = (edge_idx_1 + 1) % len(hull)

            adj_idx_1 = (edge_idx_1 - 1) % len(hull)
            adj_idx_2 = (edge_idx_1 + 2) % len(hull)

            edge_pt_1 = sympy.Point(*hull[edge_idx_1])
            edge_pt_2 = sympy.Point(*hull[edge_idx_2])
            adj_pt_1 = sympy.Point(*hull[adj_idx_1])
            adj_pt_2 = sympy.Point(*hull[adj_idx_2])

            subpoly = sympy.Polygon(adj_pt_1, edge_pt_1, edge_pt_2, adj_pt_2)
            angle1 = subpoly.angles[edge_pt_1]
            angle2 = subpoly.angles[edge_pt_2]

            # we need to first make sure that the sum of the interior angles the edge
            # makes with the two adjacent edges is more than 180°
            if sympy.N(angle1 + angle2) <= sympy.pi:
                continue

            # find the new vertex if we delete this edge
            adj_edge_1 = sympy.Line(adj_pt_1, edge_pt_1)
            adj_edge_2 = sympy.Line(edge_pt_2, adj_pt_2)
            intersect = adj_edge_1.intersection(adj_edge_2)[0]

            # the area of the triangle we'll be adding
            area = sympy.N(sympy.Triangle(edge_pt_1, intersect, edge_pt_2).area)
            # should be the lowest
            if best_candidate and best_candidate[1] < area:
                continue

            # delete the edge and add the intersection of adjacent edges to the hull
            better_hull = list(hull)
            better_hull[edge_idx_1] = intersect
            del better_hull[edge_idx_2]
            best_candidate = (better_hull, area)

        if not best_candidate:
            raise ValueError("Could not find the best fit n-gon!")

        hull = best_candidate[0]

    # back to python land
    hull = [(int(x), int(y)) for x, y in hull]

    return hull


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
    model_path,
) -> sv.Detections:
    from ultralytics import YOLO

    model = YOLO(model=model_path)
    result = model.predict(image, task="segment")[0]
    detections = sv.Detections.from_ultralytics(result)
    return detections


def board_seg_by_model(
    image,
    model_path: str = "models/board_seg.pt",
):
    img = image.copy()
    detections = seg_using_yolo(img, model_path)
    mask = list(detections)[0][1]

    hull = __get_hull_from_mask(mask)
    quad = __appx_best_fit_ngon(hull)
    img = __crop_and_transform(img, quad)
    return img


if __name__ == "__main__":
    import argparse

    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", help="input image path", required=True)
    parser.add_argument("-o", help="output image path")
    args = parser.parse_args()

    image = cv2.imread(args.i)
    image = board_seg_by_model(image)
    if args.o != None:
        cv2.imwrite(args.o, image)
