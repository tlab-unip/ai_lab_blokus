# AI Lab: Blokus


## Tile Detection



```python
import cv2
import numpy as np
import matplotlib.pyplot as plt


image_path = "readme_files/input.png"

image = cv2.imread(image_path)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
```




    <matplotlib.image.AxesImage at 0x1adb50c6310>




    
![png](readme_files/readme_2_1.png)
    


### Board Segmentation



```python
from src.detection import board_seg


img = image.copy()
img = cv2.bilateralFilter(img, 9, 75, 75)
img_segmented = board_seg.board_seg_by_model(img, "models/board_seg.pt")
plt.imshow(cv2.cvtColor(img_segmented, cv2.COLOR_BGR2RGB))
```

    
    0: 640x640 1 board, 707.1ms
    Speed: 0.0ms preprocess, 707.1ms inference, 7.7ms postprocess per image at shape (1, 3, 640, 640)
    




    <matplotlib.image.AxesImage at 0x1ade1278c50>




    
![png](readme_files/readme_4_2.png)
    


### Image Normalization



```python
from src.detection import normalization


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
major_ticks = np.arange(0, 201, 20)
minor_ticks = np.arange(0, 201, 10)
ax.set_xticks(major_ticks)
ax.set_xticks(minor_ticks, minor=True)
ax.set_yticks(major_ticks)
ax.set_yticks(minor_ticks, minor=True)
ax.grid(which="both")
ax.imshow(cv2.cvtColor(img_normalized, cv2.COLOR_BGR2RGB))
```




    <matplotlib.image.AxesImage at 0x1adc993ed10>




    
![png](readme_files/readme_6_1.png)
    


### Split Image into Grid


```python
from src.detection import grid


img_mini, img_full = grid.generate_grid(img_normalized)
plt.imshow(cv2.cvtColor(img_full, cv2.COLOR_BGR2RGB))
```




    <matplotlib.image.AxesImage at 0x1adca346710>




    
![png](readme_files/readme_8_1.png)
    



```python
from collections import Counter
from src.types.tiles import SquareColor


result = grid.detect_colors(img_mini)
print(Counter(result.flatten()))

pos_dict = dict((color, np.argwhere(result == color)) for color in SquareColor)
for key in pos_dict:
    print(f"{key:<6}", *pos_dict[key][:10], sep="\t")
```

    Counter({EMPTY: 283, GREEN: 34, YELLOW: 29, RED: 28, BLUE: 26})
    EMPTY 	[0 0]	[0 4]	[0 5]	[0 6]	[0 7]	[0 8]	[0 9]	[ 0 10]	[ 0 11]	[ 0 12]
    RED   	[10 19]	[11 19]	[12 19]	[13  8]	[13  9]	[13 16]	[13 17]	[13 18]	[14  8]	[14  9]
    GREEN 	[ 0 13]	[ 0 14]	[ 0 15]	[ 0 16]	[ 0 19]	[ 1 11]	[ 1 12]	[ 1 17]	[ 1 18]	[ 1 19]
    BLUE  	[8 9]	[ 8 10]	[ 8 11]	[9 8]	[ 9 10]	[10  7]	[10  8]	[11  7]	[12  7]	[13  6]
    YELLOW	[0 1]	[0 2]	[0 3]	[1 0]	[1 4]	[2 4]	[3 3]	[3 4]	[4 3]	[5 1]
    


```python
!jupyter nbconvert --to markdown readme.ipynb
```

    [NbConvertApp] Converting notebook readme.ipynb to markdown
    [NbConvertApp] Support files will be in readme_files\
    [NbConvertApp] Writing 3016 bytes to readme.md
    
