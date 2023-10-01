import cv2
from middleware_text import get_grayscale, opening, thresholding, canny
from pytesseract import Output
import pytesseract
import numpy as np

def remove_other_regions(img_path,img):
    duplicate = cv2.imread(img_path)
    duplicate[:] = [147, 150, 164]
    duplicate[300:550, 120:650]=img[300:550, 120:650]
    duplicate[150:350, 900:1350]=img[150:350, 900:1350]
    return duplicate

def remove_other_colours(img, array_min, array_max):
    min0 = np.array(array_min[0],np.uint8)
    max0 = np.array(array_max[0],np.uint8)
    min1 = np.array(array_min[1],np.uint8)
    max1 = np.array(array_max[1],np.uint8)
    HSV  = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(HSV, min0, max0)
    mask2 = cv2.inRange(HSV, min1, max1)
    mask = np.ma.mask_or(mask1, mask2)
    img[mask<=0] = [170,170,170]
    return img

def text_erasor(img, array_min, array_max):
    min0 = np.array(array_min[0],np.uint8)
    max0 = np.array(array_max[0],np.uint8)
    min1 = np.array(array_min[1],np.uint8)
    max1 = np.array(array_max[1],np.uint8)
    HSV  = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(HSV, min0, max0)
    mask2 = cv2.inRange(HSV, min1, max1)
    mask = np.ma.mask_or(mask1, mask2)
    img[mask>0] = [147, 150, 164]
    return img

def text_detector(img_path):
    img = cv2.imread(img_path)
    cropped = img
    # cropped = remove_other_regions(img_path,img)
    only_banner_text = remove_other_colours(cropped, [[100, 100, 50],[0, 0, 0]], [[150, 200, 255],[200, 50, 100]])
    gray = get_grayscale(only_banner_text)
    opening_img = opening(gray)
    threshold = thresholding(gray)
    canny_img = canny(gray)
    data_opening = pytesseract.image_to_data(opening_img, lang='eng', config='--psm 11 --oem 2', output_type=Output.DICT)
    data_canny = pytesseract.image_to_data(canny_img, lang='eng', config='--psm 11 --oem 2', output_type=Output.DICT)
    data_threshold = pytesseract.image_to_data(threshold, lang='eng', config='--psm 11 --oem 2', output_type=Output.DICT)
    return [data_opening,data_canny,data_threshold]