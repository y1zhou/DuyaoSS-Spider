from typing import List, Tuple

import cv2 as cv
import numpy as np
import pytesseract
from PIL import Image
from tqdm import tqdm

ROW_HEIGHT = 30  # all rows are approximately 30px
AVG_SPEED_COL_WIDTH = 90  # column width of the AvgSpeed column
UDPNAT_COL_WIDTH = 200  # column width of the "UDP NAT Type" column
DEFAULT_TESSERACT_CONFIG = r"--psm 7 --oem 3"

TESSEDIT_CHAR_WHITELIST = {
    2: r"0123456789%.",  # Loss
    3: r"0123456789.",  # Ping
    4: r"0123456789.",  # Google Ping
    5: r"0123456789.KMGBNA",  # AvgSpeed (and MaxSpeed)
    -1: r"- ABDFNOPRSTUacdeiklmnoprstuwy",  # UDP NAT Type; see https://github.com/arantonitis/pynat/blob/c5fe553bbbb79deecedcce83c4d4d2974b139355/pynat.py#L51-L59
}


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


def show_img(fig: np.ndarray, cvt_rgb: bool = False) -> Image.Image:
    if cvt_rgb:
        fig = cv.cvtColor(fig, cv.COLOR_BGR2RGB)
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
    while w - cols[-1] > UDPNAT_COL_WIDTH:
        cols = np.append(cols, cols[-1] + AVG_SPEED_COL_WIDTH)

    if w - cols[-1] > AVG_SPEED_COL_WIDTH:
        cols = np.append(cols, w - 1)

    return (rows, cols)


def text_ocr(
    img: np.ndarray,
    lang: str = "chi_sim+eng",
    oem_psm_config: str = DEFAULT_TESSERACT_CONFIG,
) -> str:
    # psm 7: treat the image as a single text line
    # oem 3: use default OCR engines
    text = pytesseract.image_to_string(img, lang=lang, config=oem_psm_config)
    return text.strip()


def crop_image(img: np.ndarray, x1: int, x2: int, y1: int, y2: int) -> np.ndarray:
    return img[y1:y2, x1:x2]


def get_col_ocr_config(j: int) -> List[str]:
    """
    j is the total number of columns. There are three cases:
      - j=6: AvgSpeed column at the end.
      - j=7: AvgSpeed and UDP NAT Type columns at the end.
      - j=8: AvgSpeed, MaxSpeed and UDP NAT Type.
    """
    # first 6 columns are always fixed
    res = []
    for i in range(6):
        config = DEFAULT_TESSERACT_CONFIG
        if i in TESSEDIT_CHAR_WHITELIST:
            config += f" -c tessedit_char_whitelist='{TESSEDIT_CHAR_WHITELIST[i]}'"
        res.append(config)

    udp_nat_type_conf = f"{DEFAULT_TESSERACT_CONFIG} -c tessedit_char_whitelist='{TESSEDIT_CHAR_WHITELIST[-1]}'"
    if j <= 8:
        if j == 7:
            res.append(udp_nat_type_conf)
        elif j == 8:  # extra MaxSpeed column
            res += [res[-1], udp_nat_type_conf]
    else:
        raise ValueError(f"Detected {j} (>8) columns.")

    return res


def img_to_csv(img: np.ndarray) -> Tuple[List[List[str]], List[str]]:
    img = enhance_borders(img)
    img_gray = convert_to_grayscale(img)
    img_bin = convert_to_bw(~img_gray)

    hlines, vlines = detect_lines_morph(img_bin)
    rows, cols = get_intersections(img_bin, hlines, vlines)

    res: List[List[str]] = []
    ocr_configs = get_col_ocr_config(len(cols) - 1)
    # Skip first row and last two rows
    for i in tqdm(range(1, len(rows) - 3)):
        row: List[str] = []

        # skip the first "group" column as we already have the information
        # from scrapy
        for j in range(1, len(cols) - 1):
            x1, x2 = cols[j], cols[j + 1]
            y1, y2 = rows[i], rows[i + 1]
            cell = crop_image(img_gray, x1, x2, y1, y2)

            lang = "chi_sim+eng" if j < 2 else "eng"
            text = text_ocr(cell, lang=lang, oem_psm_config=ocr_configs[j])
            row.append(text)
        res.append(row)

    # Last two rows
    footer: List[str] = []
    for i in range(len(rows) - 3, len(rows) - 1):
        x1, x2 = 2, img_gray.shape[1] - 3  # entire row
        y1, y2 = rows[i], rows[i + 1]
        cell = crop_image(img_gray, x1, x2, y1, y2)
        text = text_ocr(cell, lang="eng")
        footer.append(text)

    return (res, footer)
