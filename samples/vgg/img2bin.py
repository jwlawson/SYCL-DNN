#! /usr/bin/env python3

import numpy as npy
from PIL import Image
from sys import argv

def main():
    vgg_average = npy.array([103.939, 116.779, 123.68], dtype=npy.float32)
    f = argv[1]
    img = Image.open(f).resize((224, 224))
    raw = np.frombuffer(img.tobytes(), dtype=np.uint8).astype(np.float32)
    arr = raw.reshape(224 * 224, 3)
    # Convert RGB image to BGR
    arr = arr[..., ::-1]
    arr = arr - vgg_average
    arr.tofile(f + ".bin")

if __name__ == "__main__":
    main()
