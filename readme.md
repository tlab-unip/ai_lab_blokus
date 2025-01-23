# AI Lab: Blokus


## Tile Detection

- Detection Workflow (src.detection)
  1. Board Segmentation (board_seg)
  2. Normalization (normalization)
  3. Grid Generation (grid)
- For minimum coding, use `detection.detect(...)`



```python
%load_ext autoreload
%autoreload 2

import cv2
import numpy as np
import matplotlib.pyplot as plt

image_path = "readme_files/input.png"

image = cv2.imread(image_path)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
```




    <matplotlib.image.AxesImage at 0x19d9d0d4610>




    
![png](readme_files/readme_2_1.png)
    


### Board Segmentation

- Segmentation Process
  - Get mask from model
  - Crop image using the mask
  - Perspective transformation on the cropped image 


```python
from src.detection import board_seg


img = image.copy()
img = cv2.bilateralFilter(img, 9, 75, 75)
img_segmented = board_seg.board_seg_by_model(img, "models/board_seg.pt")
plt.imshow(cv2.cvtColor(img_segmented, cv2.COLOR_BGR2RGB))
```

    
    0: 640x640 1 board, 370.9ms
    Speed: 8.1ms preprocess, 370.9ms inference, 4.3ms postprocess per image at shape (1, 3, 640, 640)
    




    <matplotlib.image.AxesImage at 0x15092675f10>




    
![png](readme_files/readme_4_2.png)
    


### Image Normalization

- Normalization Process
  - Resize
  - Color Correction
  - Color Mapping


```python
from src.detection import normalization
import src.utils as utils

fig, axes = plt.subplots(1, 3, figsize=(12, 12))

img = cv2.resize(img_segmented, (200, 200))
ax = axes[0]
ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

img = normalization.__color_correction(img)
ax = axes[1]
ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

rgyb_thres = (167, 107, 167, 97)
img_normalized = normalization.__color_mapping(img, rgyb_thres)
ax = axes[2]
utils.ax_grid_setting(ax)
ax.imshow(cv2.cvtColor(img_normalized, cv2.COLOR_BGR2RGB))
```




    <matplotlib.image.AxesImage at 0x15093802050>




    
![png](readme_files/readme_6_1.png)
    


### Split Image into Grid

- Grid Generation Steps
  - Split the image into 20x20
  - Get dominant color from each grid


```python
from src.detection import grid
from collections import Counter

color_grid = grid.generate_grid(img_normalized)
print(Counter(color_grid.flatten()))

img_grid = grid.generate_image(color_grid)
plt.imshow(cv2.cvtColor(img_grid, cv2.COLOR_BGR2RGB))
```

    Counter({EMPTY: 284, GREEN: 34, YELLOW: 29, RED: 27, BLUE: 26})
    




    <matplotlib.image.AxesImage at 0x15094064d90>




    
![png](readme_files/readme_8_2.png)
    


## Recommendation

- Recommendation Process
  - Restore game state using detected grid
  - Apply algorithm to get next move


```python
%load_ext autoreload
%autoreload 2
import cv2
import numpy as np
import matplotlib.pyplot as plt
from src import detection
from src.types.tiles import *

image_path = "readme_files/input.png"
image = cv2.imread(image_path)

main_grid = detection.detect(image, rgyb_thres=(165, 107, 160, 97))
main_img = detection.generate_image(main_grid)
plt.imshow(cv2.cvtColor(main_img, cv2.COLOR_BGR2RGB))
```

    
    0: 640x640 1 board, 565.8ms
    Speed: 6.5ms preprocess, 565.8ms inference, 0.0ms postprocess per image at shape (1, 3, 640, 640)
    




    <matplotlib.image.AxesImage at 0x2a892b066d0>




    
![png](readme_files/readme_10_2.png)
    


### Interactive Gamer Runner


```python
%matplotlib widget
from src.game.ui import render_pyplot
from src.game.logic import *
from src.types.tiles import SquareColor

players = [
    SquareColor.RED,
    SquareColor.GREEN,
    SquareColor.BLUE,
    SquareColor.YELLOW,
]

color_masks = dict((color, 0) for color in players)
for color in players:
    color_mask = np.zeros_like(main_grid, dtype=int)
    color_mask[main_grid == color] = 1
    color_masks[color]= encode_bitboard(color_mask)

key_step_map= {
    "1": step_random,
    "2": step_greedy,
    "3": step_maxn,
    "z": log_available_tiles,
    "x": log_player_score,
}
render_pyplot(GameContext(players, color_masks), key_step_map)
```

    Warning: Invalid tiles detected
    



<div style="display: inline-block;">
    <div class="jupyter-widgets widget-label" style="text-align: center;">
        Figure
    </div>
    <img src='data:image/png;base64,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' width=640.0/>
</div>



    RED   	28
    GREEN 	39
    BLUE  	30
    YELLOW	35
    
    RED   	42
    GREEN 	48
    BLUE  	41
    YELLOW	43
    
    No more possible steps for GREEN
    No more possible steps for BLUE
    No more possible steps for YELLOW
    No more possible steps for RED
    No more possible steps for GREEN
    No more possible steps for BLUE
    RED   	42
    GREEN 	48
    BLUE  	41
    YELLOW	43
    
    No more possible steps for YELLOW
    No more possible steps for RED
    No more possible steps for GREEN
    No more possible steps for BLUE
    No more possible steps for YELLOW
    No more possible steps for RED
    RED   	42
    GREEN 	48
    BLUE  	41
    YELLOW	43
    
    No more possible steps for GREEN
    No more possible steps for BLUE
    No more possible steps for YELLOW
    No more possible steps for RED
    No more possible steps for GREEN
    RED   	42
    GREEN 	48
    BLUE  	41
    YELLOW	43
    
    RED   	42
    GREEN 	48
    BLUE  	41
    YELLOW	43
    
    


```python
!jupyter nbconvert --to markdown readme.ipynb
```

    [NbConvertApp] Converting notebook readme.ipynb to markdown
    [NbConvertApp] Support files will be in readme_files\
    [NbConvertApp] Writing 8817 bytes to readme.md
    
