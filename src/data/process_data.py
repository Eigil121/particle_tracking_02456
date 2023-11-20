import glob
import lvpyio as lv
from PIL import Image
import numpy as np
from tqdm import tqdm

im7_dir = 'data/raw/batch1/'

target_dir = 'data/interim/batch1/'

camera_crops = {0: (100, 300), 1: (100, 300), 2: (300,500), 3:(300,500)}

crop_image = lambda img, camera: img[camera_crops[camera][0]:camera_crops[camera][1], :]
clip_image = lambda img: np.clip(img, 0, 255).astype(np.uint8)

# Wrap your iterable with tqdm for a progress bar
for im7_img in tqdm(glob.glob(im7_dir + '/B*.im7')):

    buffer = lv.read_buffer(im7_img)
    filename = im7_img[-10:]
    print(glob.glob(im7_dir + '/B*.im7'))
    for camera in range(4):
        img = buffer[camera].as_masked_array().data
        img = crop_image(img, camera)
        img = clip_image(img)
        img = Image.fromarray(img)
        img.save(target_dir + f"cam{camera}/" + filename + '.png')

        


