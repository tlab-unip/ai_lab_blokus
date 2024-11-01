import cv2
import os
import numpy as np
import supervision as sv
from collections.abc import Generator


def __annotate_image(image, detections):
    annotator = sv.MaskAnnotator()
    annotated = annotator.annotate(
        scene=image,
        detections=detections,
    )
    sv.plot_image(annotated)


def apply_inference_model(
    image,
    model_id: str = "blokus_board_seg/1",
    show_detections=False,
) -> Generator[np.ndarray]:
    import inference

    # os.environ["ROBOFLOW_API_KEY"] = ""
    model = inference.get_model(model_id)
    result = model.infer(image)[0]
    detections = sv.Detections.from_inference(result)
    if show_detections:
        __annotate_image(image, detections)
    for xyxy, mask, confidence, class_id, tracker_id, data in detections:
        yield mask


def apply_local_model(
    image,
    model_path: str = "models/board_seg_v1.pt",
    show_detections=False,
) -> Generator[np.ndarray]:
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
    if show_detections:
        __annotate_image(image, detections)
    for xyxy, mask, confidence, class_id, tracker_id, data in detections:
        yield mask


if __name__ == "__main__":
    import argparse

    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", help="input image path", required=True)
    args = parser.parse_args()

    image = cv2.imread(args.i)
    mask = next(apply_local_model(image, show_detections=True))
    print(mask.shape)
