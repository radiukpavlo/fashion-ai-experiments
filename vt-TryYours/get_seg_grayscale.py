"""
get_seg_grayscale.py
"""
import cv2
from PIL import Image
import pickle
import json
import numpy as np

img = Image.open('./static_temp/resized_segmentation_img.png')

img_w, img_h = img.size

img = np.array(img)
gray_img = np.zeros((img_h, img_w))

for y_idx in range(img.shape[0]):
    for x_idx in range(img.shape[1]):
        tmp = img[y_idx][x_idx]
        if np.array_equal(tmp, [0, 0, 0]):
            gray_img[y_idx][x_idx] = 0
        if np.array_equal(tmp, [255, 0, 0]):
            gray_img[y_idx][x_idx] = 2  # hair
        elif np.array_equal(tmp, [0, 0, 255]):
            gray_img[y_idx][x_idx] = 13  # head
        elif np.array_equal(tmp, [85, 51, 0]):
            gray_img[y_idx][x_idx] = 10  # neck
        elif np.array_equal(tmp, [255, 85, 0]):
            gray_img[y_idx][x_idx] = 5  # body
        elif np.array_equal(tmp, [0, 255, 255]):
            gray_img[y_idx][x_idx] = 15  # left hand
        elif np.array_equal(tmp, [51, 170, 221]):
            gray_img[y_idx][x_idx] = 14  # right hand
        elif np.array_equal(tmp, [0, 85, 85]):
            gray_img[y_idx][x_idx] = 9  # pants
        elif np.array_equal(tmp, [0, 0, 85]):
            gray_img[y_idx][x_idx] = 6  # dresses
        elif np.array_equal(tmp, [0, 128, 0]):
            gray_img[y_idx][x_idx] = 12  # skirt
        elif np.array_equal(tmp, [177, 255, 85]):
            gray_img[y_idx][x_idx] = 17  # left leg
        elif np.array_equal(tmp, [85, 255, 170]):
            gray_img[y_idx][x_idx] = 16  # right leg
        elif np.array_equal(tmp, [0, 119, 221]):
            gray_img[y_idx][x_idx] = 5  # coat
        else:
            gray_img[y_idx][x_idx] = 0

img = cv2.resize(gray_img, (768, 1024), interpolation=cv2.INTER_NEAREST)
bg_img = Image.fromarray(np.uint8(img), "L")
bg_img.save("./HR-VITON-main/test/test/image-parse-v3/00001_00.png")

# # Convert the image from BGR to grayscale
# # OpenCV reads images in BGR format, not RGB
# img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
# # Resize the grayscale image
# img_resized = cv2.resize(img_gray, (768, 1024), interpolation=cv2.INTER_NEAREST)
#
# # Convert the numpy array to PIL image for better visualization or further manipulation
# img_pil = Image.fromarray(np.uint8(img_resized), "RGB")
#
# # You can then save or display the new image
# img_pil.save("./HR-VITON-main/test/test/image-parse-v3/00001_00.png")
