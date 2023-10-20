import cv2
from middleware_text import *
from pytesseract import Output
import pytesseract
import numpy as np
from collections import Counter
from ranges import min_range, max_range

def find_most_common_color(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    flattened_image_array = img.reshape(-1, 3)
    color_counts = Counter(tuple(color) for color in flattened_image_array)
    most_common_color_rgb = max(color_counts, key=color_counts.get)
    return list(most_common_color_rgb)

def remove_other_regions():
    return [[0,800,120,650], [0,1000,900,1350]]

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

def text_erasor(img, array_min=min_range, array_max=max_range):
    min0 = np.array(array_min[0],np.uint8)
    max0 = np.array(array_max[0],np.uint8)
    min1 = np.array(array_min[1],np.uint8)
    max1 = np.array(array_max[1],np.uint8)
    HSV  = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    kernel = np.ones((15, 15), np.uint8)
    mask1 = cv2.inRange(HSV, min0, max0)
    mask2 = cv2.inRange(HSV, min1, max1)
    mask1 = cv2.dilate(mask1, kernel, iterations=1)
    mask2 = cv2.dilate(mask2, kernel, iterations=1)
    mask = np.ma.mask_or(mask1, mask2)
    img[mask>0] = rgb_to_hsv(find_most_common_color(img))
    return img

def dict_join(index, position, dict1, dict2):
    keys = ["left","top","width","height", "text", "colour"]
    dict_refer = {
        "left":position[2],
        "top":position[0]
    }
    dict2["colour"] = [str(index)]*len(dict2["left"])
    for x in keys:
        if x in dict_refer.keys():
            temp = dict2[x]
            for y in range(len(dict2[x])):
                temp[y] = dict2[x][y] + dict_refer[x]
            dict2[x] = temp
    for x in keys:
        temp = dict1[x] 
        temp.extend(dict2[x])
        dict1[x] = temp
    return dict1

def rgb_to_hsv(rgb_color):
    bgr_color = np.array([[rgb_color]], np.uint8)
    hsv_color = cv2.cvtColor(bgr_color, cv2.COLOR_RGB2BGR)
    hsv_color = hsv_color[0][0]
    return hsv_color

def text_detector(img_path):
    img = cv2.imread(img_path)
    array = remove_other_regions()
    data_opening = {
        "left":[],
        "top":[],
        "width":[],
        "height":[],
        "text":[],
        "colour":[]
    }
    data_canny = {
        "left":[],
        "top":[],
        "width":[],
        "height":[],
        "text":[],
        "colour":[]
    }
    data_threshold = {
        "left":[],
        "top":[],
        "width":[],
        "height":[],
        "text":[],
        "colour":[]
    }
    for x in array: 
        only_banner_text = remove_other_colours(img[x[0]:x[1], x[2]:x[3]], min_range, max_range)
        noise_removed = remove_noise(only_banner_text)
        eroded = erode(noise_removed)
        dilated = dilate(eroded)
        gray = get_grayscale(eroded)
        opening_img = opening(gray)
        threshold = thresholding(gray)
        canny_img = canny(gray)
        data_opening = dict_join(array.index(x), x, data_opening, pytesseract.image_to_data(opening_img, lang='eng', config='--psm 12 -c tessedit_char_whitelist=ROHIT', output_type=Output.DICT))
        data_canny = dict_join(array.index(x), x, data_canny, pytesseract.image_to_data(canny_img, lang='eng', config='--psm 12 -c tessedit_char_whitelist=ROHIT', output_type=Output.DICT))
        data_threshold = dict_join(array.index(x), x, data_threshold, pytesseract.image_to_data(threshold, lang='eng', config='--psm 12 -c tessedit_char_whitelist=ROHIT', output_type=Output.DICT))
    return [data_opening,data_canny,data_threshold]

