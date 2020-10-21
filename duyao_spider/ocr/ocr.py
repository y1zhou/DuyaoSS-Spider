from typing import List, Tuple

import cv2 as cv
import numpy as np
import pytesseract
from PIL import Image
from tqdm import tqdm

ROW_HEIGHT = 30  # all rows are approximately 30px
AVG_SPEED_COL_WIDTH = 90  # column width of the AvgSpeed column


def read_img(img_path: str, crop_header: bool = True) -> np.ndarray:
    img = cv.imread(img_path)
    if crop_header:
        img = img[ROW_HEIGHT:]
    return img


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


def enhance_borders(img: np.ndarray, naive: bool = False) -> np.ndarray:
    if naive:
        height, width = img.shape[:2]
        rows = [x for x in range(height + 1) if x % ROW_HEIGHT == 0]
        for r in rows:
            cv.line(img, (0, r), (width - 1, r), (0, 0, 0), 1)
    else:
        hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        lower_gray = np.array([0, 0, 0], np.uint8)
        upper_gray = np.array([179, 50, 200], np.uint8)
        mask_gray = cv.inRange(hsv, lower_gray, upper_gray)
        img = cv.bitwise_and(img, img, mask=~mask_gray)
    return img


def show_img(fig: np.ndarray) -> Image.Image:
    return Image.fromarray(fig)


def detect_lines_morph(
    img_bin: np.ndarray, scale: int = 30
) -> Tuple[np.ndarray, np.ndarray]:
    rows, cols = img_bin.shape[:2]

    # horizontal lines
    horizontal_lines = np.zeros((rows, cols), np.uint8)
    i_hlines = [x for x in range(rows) if x % ROW_HEIGHT == 0]
    if rows - i_hlines[-1] > 20:
        i_hlines.append(rows - 1)
    horizontal_lines[i_hlines] = 255

    # Create structure element for extracting vertical lines through morphology operations
    v_kernel = cv.getStructuringElement(cv.MORPH_RECT, (1, rows // scale))
    vertical_lines = cv.erode(img_bin, v_kernel, iterations=1)
    vertical_lines = cv.dilate(vertical_lines, v_kernel, iterations=1)

    return (horizontal_lines, vertical_lines)


def get_intersections(
    img_bin: np.ndarray, hlines: np.ndarray, vlines: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    cross_points = cv.bitwise_and(hlines, vlines)

    h, w = img_bin.shape[:2]
    total_rows = h // ROW_HEIGHT

    # we can get the rows easily
    rows = np.where(hlines[:, 0] != 0)[0]

    # for columns, only consider the rows where the horizontal lines are.
    # if there's lots of (>20%) white pixels in a column at those rows,
    # then there's probably a vertical line.
    cols = np.sum(cross_points[rows], axis=0)
    cols = np.where(cols > 255 * total_rows // 5)[0]
    cols_diff = np.diff(cols)
    cols = cols[np.where(cols_diff > 10)[0]]  # remove clusters

    # a hack to split the last two columns
    # the colored "AvgSpeed" column is hard to recognize
    if w - cols[-1] > AVG_SPEED_COL_WIDTH:  # missing a column
        if w - cols[-1] > 200:
            cols = np.append(cols, [cols[-1] + AVG_SPEED_COL_WIDTH, w - 1])
        else:
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
    return text.strip()


def img_to_csv(img: np.ndarray) -> Tuple[List[List[str]], List[str]]:
    img = enhance_borders(img)
    img_gray = convert_to_grayscale(img)
    img_bin = convert_to_bw(~img_gray)

    hlines, vlines = detect_lines_morph(img_bin)
    rows, cols = get_intersections(img_bin, hlines, vlines)

    res: List[List[str]] = []
    # Skip last two rows
    for i in tqdm(range(len(rows) - 3)):
        row: List[str] = []
        for j in range(len(cols) - 1):
            x1, x2 = cols[j], cols[j + 1]
            y1, y2 = rows[i], rows[i + 1]
            text = text_ocr(img_gray, x1, x2, y1, y2)
            row.append(text)
        res.append(row)

    # Last two rows
    footer: List[str] = []
    for i in range(len(rows) - 3, len(rows) - 1):
        x1, x2 = 2, img_gray.shape[1] - 3
        y1, y2 = rows[i], rows[i + 1]

        text = text_ocr(img_gray, x1, x2, y1, y2, lang="eng")
        footer.append(text)

    return (res, footer)
