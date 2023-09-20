
from face_swap import swap, replace_face, plt, cv2, DeepFace
import os,shutil
import face_recognition
from moviepy.editor import *
import pytesseract
import numpy as np
from PIL import Image

tolerance = 0.5 # 0.5 -> Very Accurate | >0.5 -> Decent

old_face = "static/orignal.png"
new_face = "static/demo.png"
video = "static/sample.mp4"
new_name = "Arnav"

def video_frames(path):
    array = []
    shutil.rmtree("frames")
    os.mkdir("frames")
    capture = cv2.VideoCapture(path)
    video = VideoFileClip(path)
    audio = video.audio
    audio.write_audiofile("output-audio.mp3")
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
    video = cv2.VideoWriter("output-video.mp4", cv2.VideoWriter_fourcc(*'DIVX'), fps, (width,height))
    for image in frames:
        video.write(cv2.imread(image))
    cv2.destroyAllWindows()
    video.release()
    video_clip = VideoFileClip("output-video.mp4")
    audio_clip = AudioFileClip("output-audio.mp3")
    final_clip = video_clip.set_audio(audio_clip)
    final_clip.write_videofile("output.mp4")
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
            cv2.imwrite("temp.jpg", new_image)
            img[y:y+h, x:x+w] = cv2.imread("temp.jpg")
            cv2.imwrite(frame, img)

encodings = [ face_recognition.face_encodings(cv2.imread(old_face),model="large")[0]]
dictionary = {0:cv2.imread(new_face)} 

def detect_faces_and_swap(frame):
    img = cv2.imread(frame)
    cv2.imwrite(frame, img)
    i = 0
    obj = DeepFace.analyze(img, actions = ["gender"])
    for temp_img in obj:
        i = i + 1
        x = max(temp_img["region"]["x"] - int(temp_img["region"]["w"]/2), 0)
        y = max(temp_img["region"]["y"] - int(temp_img["region"]["h"]/2), 0)
        w = temp_img["region"]["w"] + int(temp_img["region"]["w"]/1.5)
        h = temp_img["region"]["h"] + int(temp_img["region"]["h"]/1.5)
        face = img[y:y+h, x:x+w]
        action(encodings, dictionary, face, frame, img, x,y,w,h)

def extract_text_and_replace(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    th = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    black = np.zeros([img.shape[0] + 2, img.shape[1] + 2], np.uint8)
    mask = cv2.floodFill(th.copy(), black, (0,0), 0, 0, 0, flags=8)[1]
    kernel_length = 30
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length, 1))
    dilate = cv2.dilate(mask, horizontal_kernel, iterations=1)
    contours = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        cv2.imwrite("temp-text.png", img[y:y+h, x:x+w])
        text_img = Image.open("temp-text.png")
        text_img.resize((text_img.width * 3, text_img.height * 3))
        text_img = text_img.convert('L')
        extracted_text = pytesseract.image_to_string(text_img,config='--psm 6', lang='eng')
        if  "rohit" in extracted_text.lower() or "rohii" in extracted_text.lower() or "ohi" in extracted_text.lower() or ("r" in extracted_text.lower() and "m" not in extracted_text.lower()):
            empty = cv2.imread("static/white.png")
            resized_empty = cv2.resize(empty, (w,h))
            img[y:y+h, x:x+w] = resized_empty
            cv2.imwrite(image_path, img)

def change_video(path):
    array = video_frames(path)
    frames = array[0]
    fps = array[1]
    for x in frames:
        print(x)
        extract_text_and_replace(x)
        detect_faces_and_swap(x)
    video_join(frames, fps)

change_video(video)


