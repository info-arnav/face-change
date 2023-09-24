import cv2
from middleware_text import get_grayscale, opening, thresholding, canny
from pytesseract import Output
import pytesseract
import numpy as np

def text_detector(img_path):
    img = cv2.imread(img_path)
    gray = get_grayscale(img)
    opening_img = opening(gray)
    threshold = thresholding(gray)
    canny_img = canny(gray)
    data_opening = pytesseract.image_to_data(opening_img, lang='eng', config='--psm 11 --oem 2', output_type=Output.DICT)
    data_canny = pytesseract.image_to_data(canny_img, lang='eng', config='--psm 11 --oem 2', output_type=Output.DICT)
    data_threshold = pytesseract.image_to_data(threshold, lang='eng', config='--psm 11 --oem 2', output_type=Output.DICT)
    return [data_opening,data_canny,data_threshold]