from face_swap import swap, cv2, DeepFace
from text_detector_test import text_detector, text_erasor, rgb_to_hsv, find_most_common_color
import os,shutil
import face_recognition
import numpy as np
from moviepy.editor import *
from PIL import Image
from error_scope import original_name_errors
from east_text_model import find_text_boxes
from word_to_image import *
from ranges import black, blue
import math

import keras_ocr

pipeline = keras_ocr.pipeline.Pipeline()

executable = True

tolerance = 0.5 # 0.5 -> Very Accurate | >0.5 -> Decent

old_face = "static/orignal.png"
new_face = "static/demo.png"
video = "static/sample.mp4"
new_name = "Arnav"
original_name = "rohit"

banner_blue_loc_array = []
banner_blue_dim_array = [1,1]
banner_black_loc_array = []
banner_black_dim_array = [1,1]

def video_frames(path):
    array = []
    shutil.rmtree("frames")
    os.mkdir("frames")
    shutil.rmtree("text")
    os.mkdir("text")
    shutil.rmtree("text-box")
    os.mkdir("text-box")
    shutil.rmtree("detected-text")
    os.mkdir("detected-text")
    capture = cv2.VideoCapture(path)
    video = VideoFileClip(path)
    audio = video.audio
    audio.write_audiofile("temp/output-audio.mp3")
    fps = capture.get(cv2.CAP_PROP_FPS)
    frameNr = 0
    while (True):
        success, frame = capture.read()
        if success:
            cv2.imwrite(f'frames/frame_{frameNr}.jpg', frame)
            array.append(f'frames/frame_{frameNr}.jpg')
        else:
            break
        frameNr = frameNr+1
    capture.release()
    return [array, fps]

def video_play(path):
    cap = cv2.VideoCapture(path)
    if (cap.isOpened()== False):
        print("Error opening video file")
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            cv2.imshow('Frame', frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()

def video_join(frames, fps):
    height, width, layers = cv2.imread(frames[0]).shape
    video = cv2.VideoWriter("temp/output-video.mp4", cv2.VideoWriter_fourcc(*'DIVX'), fps, (width,height))
    for image in frames:
        video.write(cv2.imread(image))
    cv2.destroyAllWindows()
    video.release()
    video_clip = VideoFileClip("temp/output-video.mp4")
    audio_clip = AudioFileClip("temp/output-audio.mp3")
    final_clip = video_clip.set_audio(audio_clip)
    final_clip.write_videofile("output/output.mp4")
    print("Video Saved")

def match_face(face_array, face):
    result = face_recognition.compare_faces(face_array, face, tolerance=tolerance) #tolerance=0.5 removed
    if True in result:
        return [True, result.index(True)]
    else:
        return [False, len(face_array)]

def action(encodings, dictionary, face, frame, img,x,y,w,h):
    file = f'middleware/{len(encodings)}.jpg'
    cv2.imwrite(file, face)
    loaded = cv2.imread(file)
    if face_recognition.face_encodings(loaded, model="large") != []:
        encoding = face_recognition.face_encodings(loaded, model="large")[0]
        result = match_face(encodings, encoding)
        matched = result[1]
        print(matched, result)
        if matched in dictionary.keys():
            new_image = swap(dictionary[matched],face)
            cv2.imwrite("temp/temp.jpg", new_image)
            img[y:y+h, x:x+w] = cv2.imread("temp/temp.jpg")
            cv2.imwrite(frame, img)

encodings = [ face_recognition.face_encodings(cv2.imread(old_face),model="large")[0]]
dictionary = {0:cv2.imread(new_face)} 

def detect_faces_and_swap(frame):
    img = cv2.imread(frame)
    cv2.imwrite(frame, img)
    i = 0
    obj = []
    try:
        obj = DeepFace.analyze(img, actions = ["gender"])
    except:
        print("No face detected")
    for temp_img in obj:
        i = i + 1
        x = max(temp_img["region"]["x"] - int(temp_img["region"]["w"]/2), 0)
        y = max(temp_img["region"]["y"] - int(temp_img["region"]["h"]/2), 0)
        w = temp_img["region"]["w"] + int(temp_img["region"]["w"]/1.5)
        h = temp_img["region"]["h"] + int(temp_img["region"]["h"]/1.5)
        face = img[y:y+h, x:x+w]
        action(encodings, dictionary, face, frame, img, x,y,w,h)

def overlay_images(background_img_path, data, x, y):
    w = int(math.sqrt(math.pow(data[1][2][0] - data[1][3][0], 2) + math.pow(data[1][2][1] - data[1][3][1], 2)))
    h = int(math.sqrt(math.pow(data[1][1][0] - data[1][2][0], 2) + math.pow(data[1][1][1] - data[1][2][1], 2)))
    angle = math.degrees(math.atan((data[1][2][1] - data[1][3][1])/(data[1][3][0] - data[1][2][0])))
    background = Image.open(background_img_path)
    overlay = Image.open("temp/banner.png")
    if overlay.mode != 'RGBA':
        overlay = overlay.convert('RGBA')
    overlay = overlay.resize((w, h))
    overlay = overlay.rotate(angle, expand=True, resample=Image.BICUBIC)
    mask = overlay.split()[3] if len(overlay.split()) == 4 else None
    background.paste(overlay, (x, y), mask=mask)
    background.save(background_img_path)

def extract_text_and_erase(image_path):
    global banner_blue_loc_array
    global banner_black_loc_array
    image = keras_ocr.tools.read(image_path)
    prediction_groups = pipeline.recognize([image])
    for data in prediction_groups[0]:
        coordinates = data[1]
        coordinates = coordinates.astype(np.int32)
        coordinates = coordinates.reshape((-1, 1, 2))
        coordinates = coordinates.reshape((4, 2))
        coordinates = coordinates.astype(int)
        if data[0] == original_name:
            image = cv2.imread(image_path)
            x, y, w, h = cv2.boundingRect(coordinates)
            x = max(0, x)
            y = max(0, y)
            w = min(w, image.shape[1] - x)
            h = min(h, image.shape[0] - y)
            color = find_most_common_color(image[y:y+h, x:x+w])
            if x < 800:
                banner = remove_other_colours(cv2.imread("temp/new_banner_man.png"), rgb_to_hsv(color), rgb_to_hsv(blue)) 
            if x > 800:
                banner = remove_other_colours(cv2.imread("temp/new_banner_women.png"), rgb_to_hsv(color), rgb_to_hsv(black)) 
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask, [coordinates], 255)
            image[mask == 255] = rgb_to_hsv(color)
            cv2.imwrite(image_path, image)
            cv2.imwrite("temp/banner.png", banner)
            overlay_images(image_path, data, x, y)

def change_video(path):
    array = video_frames(path)
    frames = array[0]
    fps = array[1]
    get_image(new_name, "static/alphabets-v2.jpg", "new_banner_man.png")
    get_image(new_name, "static/alphabets.png", "new_banner_women.png")
    for x in frames:
        print(x)
        extract_text_and_erase(x)
        detect_faces_and_swap(x)
    video_join(frames, fps)

if executable:
    change_video(video)
