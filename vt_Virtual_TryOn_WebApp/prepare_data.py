import cv2
import numpy as np
import os


def generate_masks():
    for fn in os.listdir("dataset/test_clothes"):
        image = cv2.imread("dataset/test_clothes/"+fn)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros_like(image)
        cv2.drawContours(mask, contours, -1, (255, 255, 255), -1)
        cv2.imwrite("dataset/test_edge/"+fn, mask)


def generate_pairs():
    if os.path.exists("demo.txt"):
        os.remove("demo.txt")
    
    pairs_file = open("demo.txt", 'w')

    for ifn in os.listdir("dataset/test_img"):
        for cfn in os.listdir("dataset/test_clothes"):
            pairs_file.write(f"{ifn} {cfn}\n")
    
    pairs_file.close()
