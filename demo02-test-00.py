
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

def should_replace(text):
    return True

def resize_bbox_to_minimum(x,y,w,h):
    min_rect = cv2.minAreaRect(np.array([(x, y), (x + w, y), (x, y + h), (x + w, y + h)]))
    min_rect_points = cv2.boxPoints(min_rect).astype(int)
    min_x, min_y = np.min(min_rect_points, axis=0)
    max_x, max_y = np.max(min_rect_points, axis=0)
    return (min_x, min_y, max_x - min_x, max_y - min_y)
    
def create_boxes(n_boxes, data, path, type):
    img = cv2.imread(f"frames/{path}.jpg")
    for i in range(n_boxes):
        (x, y, w, h) = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        img = cv2.putText(img, data["text"][i].lower(), (x, y - 5),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1)
    cv2.imwrite(f"text-box/{path}-{type}.jpg", img)

def extract_text_and_erase(image_path, execution_number, frame_number):
    global banner_blue_loc_array
    global banner_black_loc_array
    global banner_blue_dim_array
    global banner_black_dim_array
    img = cv2.imread(image_path)
    path = image_path.split("/")[1].split(".")[0]
    all_boxes = find_text_boxes(img, path, "0")
    values = []
    banner_blue_loc = [0,0,1,1]
    banner_black_loc = [0,0,1,1]
    most_common_blue_color = ""
    most_common_black_color = ""
    for value in all_boxes:
        xt = value[0]
        wwt = value[2]
        wt = wwt-xt
        yt = value[1]
        wht = min(value[3], 800)
        ht = wht-yt
        if xt > 0 and yt > 0 and wt > 0 and ht > 0:
            values.append([xt-10,yt-10,wt+35,ht+35])
    for value in values:
        if (banner_blue_loc == [0,0,1,1] or banner_blue_loc[1] > value[1]) and value[0] < 800:
            banner_blue_loc = value
            most_common_blue_color = rgb_to_hsv(find_most_common_color(img[value[1]:value[1]+value[3], value[0]:value[0]+value[2]]))
        elif (banner_black_loc == [0,0,1,1] or banner_black_loc[1] > value[1]) and value[0] > 800:
            banner_black_loc = value
            most_common_black_color = rgb_to_hsv(find_most_common_color(img[value[1]:value[1]+value[3], value[0]:value[0]+value[2]]))
        img[value[1]:value[1]+value[3], value[0]:value[0]+value[2]] = text_erasor(value[0], img[value[1]:value[1]+value[3], value[0]:value[0]+value[2]])
    if execution_number == 0:
        if banner_blue_loc != [0,0,1,1] and banner_blue_dim_array == [1,1]:
            banner_blue_dim_array = [banner_blue_loc[2], banner_blue_loc[3]]
        if banner_black_loc != [0,0,1,1] and banner_black_dim_array == [1,1]:
            banner_black_dim_array = [banner_black_loc[2], banner_black_loc[3]]
        banner_blue_loc_array.append([banner_blue_loc, most_common_blue_color])
        banner_black_loc_array.append([banner_black_loc, most_common_black_color])
    if execution_number == 1:
        if banner_blue_loc_array[frame_number][1] != "":
            banner_blue = remove_other_colours(cv2.imread("temp/new_banner_man.png"), banner_blue_loc_array[frame_number][1],rgb_to_hsv(blue)) 
            new_banner_blue_resized = cv2.resize(banner_blue, (banner_blue_dim_array[0],banner_blue_dim_array[1])) 
            img[banner_blue_loc_array[frame_number][0][1]:banner_blue_loc_array[frame_number][0][1]+banner_blue_dim_array[1], banner_blue_loc_array[frame_number][0][0]:banner_blue_loc_array[frame_number][0][0]+banner_blue_dim_array[0]] = new_banner_blue_resized 
        if banner_black_loc_array[frame_number][1] != "":
            banner_black = remove_other_colours(cv2.imread("temp/new_banner_man.png"), banner_black_loc_array[frame_number][1],rgb_to_hsv(black)) 
            new_banner_black_resized = cv2.resize(banner_black, (banner_black_dim_array[0],banner_black_dim_array[1])) 
            img[banner_black_loc_array[frame_number][0][1]:banner_black_loc_array[frame_number][0][1]+banner_black_dim_array[1], banner_black_loc_array[frame_number][0][0]:banner_black_loc_array[frame_number][0][0]+banner_black_dim_array[0]] = new_banner_black_resized 
    cv2.imwrite(image_path, img)



def change_video(path):
    array = video_frames(path)
    frames = array[0]
    fps = array[1]
    get_image(new_name, "static/alphabets-v2.jpg", "new_banner_man.png")
    get_image(new_name, "static/alphabets.png", "new_banner_women.png")
    for x in frames:
        print(x)
        for y in range(2):
            extract_text_and_erase(x, y, frames.index(x))
        # detect_faces_and_swap(x)
    video_join(frames, fps)

if executable:
    change_video(video)


