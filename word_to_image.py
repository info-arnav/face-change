import cv2
import numpy as np
from pytesseract import Output
import pytesseract

def remove_other_colours(img, bgcolour, colour, min_value=[0,0,200], max_value=[255,255,255]):
    minv = np.array(min_value,np.uint8)
    maxv = np.array(max_value,np.uint8)
    HSV  = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(HSV, minv, maxv)
    img[mask>0] = bgcolour
    img[mask<=0] = colour
    return img

def hconcat_resize(img_list, interpolation = cv2.INTER_CUBIC): 
    h_min = min(img.shape[0] for img in img_list) 
    im_list_resize = [cv2.resize(img,(int(img.shape[1] * h_min / img.shape[0]),h_min), interpolation = interpolation) for img in img_list] 
    return cv2.hconcat(im_list_resize) 

def get_image(text, path, filename, bgcolour=[255,255,255], colour=[0,0,0]):
    image = cv2.imread(path)
    hImg,wImg,_ = image.shape
    boxes = pytesseract.image_to_boxes(image)
    dictionary = {}
    img_new = [None]
    for b in boxes.splitlines():
        b = b.split(' ')
        [x,y,w,h] = [int(b[1]),hImg-int(b[4]),int(b[3]),hImg-int(b[2])]
        dictionary[b[0].upper()] = [x,y,w,h]
    for x in text:
        if x.upper() in dictionary.keys():
            [x,y,w,h] = dictionary[x.upper()]
            if None in img_new:
                img_new = image[y:h,x:w]
            else:        
                img_new = hconcat_resize([img_new, image[y:h,x:w]])
    img_new = remove_other_colours(img_new, bgcolour, colour, [0,0,200], [255,255,255])
    cv2.imwrite(f"temp/{filename}", img_new)
    return img_new
