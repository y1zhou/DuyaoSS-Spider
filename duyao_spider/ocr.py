from pathlib import Path

import cv2 as cv
import numpy as np
import pytesseract
from PIL import Image


def convert_to_grayscale(img: np.ndarray) -> np.ndarray:
    return cv.cvtColor(img, cv.COLOR_BGR2GRAY)


def convert_to_bw(img: np.ndarray, adaptive_threshold: bool = True) -> np.ndarray:
    if adaptive_threshold:
        blackAndWhiteImage = cv.adaptiveThreshold(
            img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2
        )
    else:
        (_, blackAndWhiteImage) = cv.threshold(img, 127, 255, cv.THRESH_BINARY)
    return blackAndWhiteImage


def read_img(img_path: str, crop_header: bool = True) -> np.ndarray:
    img = cv.imread(img_path)
    if crop_header:
        # each row should have height ~30px
        img = img[30:]
    return img


def show_img(fig: np.ndarray) -> Image.Image:
    return Image.fromarray(fig)


def detect_lines(img: np.ndarray, minLineLength: int = 500, maxLineGap: int = 10):
    # See https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_houghlines/py_houghlines.html
    edges = cv.Canny(img, 100, 150, apertureSize=3)
    lines = cv.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=400,
        lines=10,
        minLineLength=minLineLength,
        maxLineGap=maxLineGap,
    )

    return lines


def draw_lines(img: np.ndarray, lines):
    lines = lines.squeeze()
    for x1, y1, x2, y2 in lines:
        # draw thick green line for the identified lines
        cv.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return show_img(img)
