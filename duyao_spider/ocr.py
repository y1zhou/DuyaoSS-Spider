from typing import Tuple

import cv2 as cv
import numpy as np
import pytesseract
from PIL import Image
from tqdm import tqdm


def convert_to_grayscale(img: np.ndarray) -> np.ndarray:
    return cv.cvtColor(img, cv.COLOR_BGR2GRAY)


def convert_to_bw(img: np.ndarray, adaptive_threshold: bool = True) -> np.ndarray:
    if adaptive_threshold:
        blackAndWhiteImage = cv.adaptiveThreshold(
            img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 2
        )
    else:
        (_, blackAndWhiteImage) = cv.threshold(img, 127, 255, cv.THRESH_BINARY)
    return blackAndWhiteImage


def read_img(
    img_path: str, crop_header: bool = True, enhance_rows: bool = False
) -> np.ndarray:
    img = cv.imread(img_path)
    if crop_header:
        # each row should have height ~30px
        img = img[30:]
    return img


def enhance_borders(img: np.ndarray, naive: bool = False) -> np.ndarray:
    if naive:
        height, width = img.shape[:2]
        rows = [x for x in range(height + 1) if x % 30 == 0]
        for r in rows:
            cv.line(img, (0, r), (width - 1, r), (0, 0, 0), 1)
    else:
        hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        lower_gray = np.array([0, 0, 0], np.uint8)
        upper_gray = np.array([179, 50, 200], np.uint8)
        mask_gray = cv.inRange(hsv, lower_gray, upper_gray)
        img = cv.bitwise_and(img, img, mask=~mask_gray)
        # black_pixel = np.array([0, 0, 0], np.uint8)
        # for i, row in enumerate(img):
        #     for j, pixel in enumerate(row):
        #         if pixel[0] < 200 and np.all(pixel == pixel[0]):
        #             img[i][j] = black_pixel
    return img


def show_img(fig: np.ndarray) -> Image.Image:
    return Image.fromarray(fig)


def detect_lines_morph(
    img_bin: np.ndarray, scale: int = 50
) -> Tuple[np.ndarray, np.ndarray]:
    rows, cols = img_bin.shape[:2]

    # Create structure element for extracting lines through morphology operations
    ## horizontal lines
    h_kernel = cv.getStructuringElement(cv.MORPH_RECT, (cols // scale, 1))
    horizontal_lines = cv.erode(img_bin, h_kernel, iterations=1)
    horizontal_lines = cv.dilate(horizontal_lines, h_kernel, iterations=1)

    ## vertical lines
    v_kernel = cv.getStructuringElement(cv.MORPH_RECT, (1, rows // scale))
    vertical_lines = cv.erode(img_bin, v_kernel, iterations=1)
    vertical_lines = cv.dilate(vertical_lines, v_kernel, iterations=1)

    return (horizontal_lines, vertical_lines)


def get_intersections(
    img_bin: np.ndarray, hlines: np.ndarray, vlines: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    cross_points = cv.bitwise_and(hlines, vlines)

    h, w = img_bin.shape[:2]
    total_rows = h // 30

    rows = [x for x in range(h) if x % 30 == 0]
    rows.append(h - 1)

    cols = np.sum(cross_points, axis=0)
    cols = np.where(cols > 255 * total_rows // 10)[0]
    cols_diff = np.diff(cols)
    cols = cols[np.where(cols_diff > 10)[0]]
    if w - cols[-1] > 90:
        cols = np.append(cols, w - 1)

    return (rows, cols)


def text_ocr(
    img: np.ndarray,
    x1: int,
    x2: int,
    y1: int,
    y2: int,
    lang: str = "chi_sim+eng",
    oem_psm_config: str = r"--psm 7 --oem 3",
) -> str:
    # psm 7: treat the image as a single text line
    # oem 3: use default OCR engines
    cell = img[y1:y2, x1:x2]
    text = pytesseract.image_to_string(cell, lang=lang, config=oem_psm_config)
    return text
