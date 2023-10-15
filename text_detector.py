import cv2
from middleware_text import get_grayscale, opening, thresholding, canny
from pytesseract import Output
import pytesseract
import numpy as np
from ranges import min_range, max_range

def remove_other_regions():
    return [[300,550,120,650], [150,350,900,1350]]

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

def text_erasor(img, colour, array_min=min_range, array_max=max_range):
    min0 = np.array(array_min[0],np.uint8)
    max0 = np.array(array_max[0],np.uint8)
    min1 = np.array(array_min[1],np.uint8)
    max1 = np.array(array_max[1],np.uint8)
    HSV  = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(HSV, min0, max0)
    mask2 = cv2.inRange(HSV, min1, max1)
    mask = np.ma.mask_or(mask1, mask2)
    img[mask>0] = colour
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
        cv2.imwrite(f"delete/{x}{img_path[-6:]}", only_banner_text)
        gray = get_grayscale(only_banner_text)
        opening_img = opening(gray)
        threshold = thresholding(gray)
        canny_img = canny(gray)
        data_opening = dict_join(array.index(x), x, data_opening, pytesseract.image_to_data(opening_img, lang='eng', config='--psm 11 --oem 2', output_type=Output.DICT))
        data_canny = dict_join(array.index(x), x, data_canny, pytesseract.image_to_data(canny_img, lang='eng', config='--psm 11 --oem 2', output_type=Output.DICT))
        data_threshold = dict_join(array.index(x), x, data_threshold, pytesseract.image_to_data(threshold, lang='eng', config='--psm 11 --oem 2', output_type=Output.DICT))
    return [data_opening,data_canny,data_threshold]

