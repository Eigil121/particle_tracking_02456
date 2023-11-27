import json
import numpy as np
from PIL import Image
from typing import List
import numpy as np

# The first portion of this script is from:
# https://stackoverflow.com/questions/74339154/how-to-convert-rle-format-of-label-studio-to-black-and-white-image-masks

class InputStream:
    def __init__(self, data):
        self.data = data
        self.i = 0

    def read(self, size):
        out = self.data[self.i:self.i + size]
        self.i += size
        return int(out, 2)


def access_bit(data, num):
    """ from bytes array to bits by num position"""
    base = int(num // 8)
    shift = 7 - int(num % 8)
    return (data[base] & (1 << shift)) >> shift


def bytes2bit(data):
    """ get bit string from bytes data"""
    return ''.join([str(access_bit(data, i)) for i in range(len(data) * 8)])


def rle_to_mask(rle: List[int], height: int, width: int) -> np.array:
    """
    Converts rle to image mask
    Args:
        rle: your long rle
        height: original_height
        width: original_width

    Returns: np.array
    """

    rle_input = InputStream(bytes2bit(rle))

    num = rle_input.read(32)
    word_size = rle_input.read(5) + 1
    rle_sizes = [rle_input.read(4) + 1 for _ in range(4)]
    # print('RLE params:', num, 'values,', word_size, 'word_size,', rle_sizes, 'rle_sizes')

    i = 0
    out = np.zeros(num, dtype=np.uint8)
    while i < num:
        x = rle_input.read(1)
        j = i + 1 + rle_input.read(rle_sizes[rle_input.read(2)])
        if x:
            val = rle_input.read(word_size)
            out[i:j] = val
            i = j
        else:
            while i < j:
                val = rle_input.read(word_size)
                out[i] = val
                i += 1

    image = np.reshape(out, [height, width, 4])[:, :, 3]
    return image


def build_mask(json_path, mask_nr=0):

    with open(json_path, 'r') as f:
        json_file = json.load(f)[mask_nr]
    
    json_annotations = json_file['annotations'][0]["result"]

    height, width = json_annotations[0]["original_height"], json_annotations[0]["original_width"]

    mask = np.zeros((height, width))

    for annotation in json_annotations:
        partial_annotation = annotation["value"]["rle"]
        partial_mask = rle_to_mask(partial_annotation, height, width)
        mask += partial_mask
    
    # Clean mask for values different than 0 and 255
    mask[mask != 255] = 0
    return mask

def save_mask(mask, batch, camera, image_nr):
    # Convert the NumPy array to a Pillow Image
    image = Image.fromarray(mask.astype(np.uint8))

    # Construct the path to the data
    path = f"data/interim/masks/batch{batch}cam{camera}_image{image_nr}.png"

    # Save the image as a BMP file
    image.save(path)

    return None


if __name__ == "__main__":

    json_path = "src/data/data_labeller/project-2-at-2023-11-20-14-43-f6f813f4.json"

    batch = 1
    camera = 0
    image_nr = 2
    mask_nr = 0

    mask = build_mask(json_path, mask_nr=mask_nr)

    save_mask(mask, batch, camera, image_nr)