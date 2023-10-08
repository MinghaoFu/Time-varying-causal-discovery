import numpy as np
import pandas as pd
import graphviz
import lingam
import os
import cv2
import glob

from lingam.utils import make_dot
from utils import makedir
from tqdm import tqdm


npy_path = '/home/minghao.fu/workspace/LatentTimeVaryingCausalDiscovery/results/2023-10-05_22-29-01/epoch_2000/prediction.npy'
save_path = './visual/'
video_path = './animation.mp4'
resize_path = './visual_resize/'
width, height = 1024, 1024

makedir(save_path, remove_exist=True)
makedir(resize_path, remove_exist=True)

# .npy to .png
array = np.load(npy_path)
for i in tqdm(range(array.shape[0])):
    dot = make_dot(array[i])
    dot.format = 'png'
    dot.render(os.path.join(save_path, f'Time_{i}'))

filepaths = glob.glob(f'{save_path}*.png')
# resize .png files
makedir(resize_path, remove_exist=True)
for path in tqdm(filepaths):
    img = cv2.imread(path)
    resized_img = cv2.resize(img, (width, height))  # Set your dimensions
    cv2.imwrite(resize_path + path.split('/')[-1], resized_img)

img_paths = sorted(glob.glob(f'{resize_path}*.png'))

# Create a VideoCapture object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(video_path, fourcc, 20, (width, height))  # FPS set to 1; adjust as needed

# Loop through image files to create video
for path in tqdm(img_paths):
    img = cv2.imread(path)
    if img is None:
        print(f"Warning: Image at {path} could not be read.")
        continue
    if img.shape[:2] != (height, width):
        print(f"Dimensions mismatch: {img.shape[:2]}")
    out.write(img)

out.release()
cv2.destroyAllWindows()