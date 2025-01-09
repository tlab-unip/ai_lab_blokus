import cv2
import numpy as np


def get_coordinates(color_grid):
    """return a dict recording coordinates of each color"""
    from src.types.tiles import SquareColor

    return dict((color, np.argwhere(color_grid == color)) for color in SquareColor)


def validate_detection(color_grid, tile_set):
    """validate detected grid"""
    pass


def ax_grid_setting(ax):
    major_ticks = np.arange(0, 201, 20)
    minor_ticks = np.arange(0, 201, 10)
    ax.set_xticks(major_ticks)
    ax.set_xticks(minor_ticks, minor=True)
    ax.set_yticks(major_ticks)
    ax.set_yticks(minor_ticks, minor=True)
    ax.grid(which="both")


def img_editor(img):
    import gradio as gr

    wrapper = [img]

    def predict(res):
        wrapper.append(res["composite"])
        return wrapper[-1]

    with gr.Blocks() as demo:
        with gr.Row():
            im = gr.ImageEditor(
                value=wrapper[-1],
                type="numpy",
                crop_size="1:1",
            )
            im_preview = gr.Image()
        im.change(predict, outputs=im_preview, inputs=im, show_progress="hidden")

    demo.launch()


def mat_editor(mat):
    from src.detection import grid
    import gradio as gr

    wrapper = [mat]

    def _update_mat(mat):
        wrapper.append(mat)
        img = grid.generate_image(mat)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    with gr.Blocks() as demo:
        with gr.Row():
            mat = gr.Numpy(
                value=wrapper[-1],
                datatype="number",
                row_count=20,
                col_count=20,
                interactive=True,
            )
            img_container = gr.Image()
        mat.input(_update_mat, outputs=img_container, inputs=mat)
    demo.launch()
