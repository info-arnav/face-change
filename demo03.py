
from face_swap import swap, replace_face, plt, cv2, DeepFace
import os,shutil
import face_recognition
from moviepy.editor import *

tolerance = 0.6 # 0.5 -> Very Accurate | >0.5 -> Decent

def video_frames(path):
    array = []
    shutil.rmtree("frames")
    os.mkdir("frames")
    capture = cv2.VideoCapture(path)
    video = VideoFileClip(path)
    audio = video.audio
    audio.write_audiofile("output.mp3")
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
    video = cv2.VideoWriter("output.mp4", cv2.VideoWriter_fourcc(*'DIVX'), fps, (width,height))
    for image in frames:
        video.write(cv2.imread(image))
    cv2.destroyAllWindows()
    video.release()
    video_clip = VideoFileClip("output.mp4")
    audio_clip = AudioFileClip("output.mp3")
    final_clip = video_clip.set_audio(audio_clip)
    final_clip.write_videofile("output.mp4")
    video_play("output.mp4")

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
    encoding = face_recognition.face_encodings(loaded, model="large")[0]
    result = match_face(encodings, encoding)
    exists = result[0]
    matched = result[1]
    print(matched, result)
    if matched in dictionary.keys():
        new_image = swap(dictionary[matched],face)
        cv2.imwrite("temp.jpg", new_image)
        img[y:y+h, x:x+w] = cv2.imread("temp.jpg")
        cv2.imwrite(frame, img)

encodings = [ face_recognition.face_encodings(cv2.imread("static/test01.png"),model="large")[0]]
dictionary = {0:cv2.imread("static/pic2.jpg")} 
def detect_faces_and_swap(frame):
    img = cv2.imread(frame)
    i = 0
    try:
        obj = DeepFace.analyze(img, actions = ["gender"])
        for temp_img in obj:
            i = i + 1
            x = max(temp_img["region"]["x"] - 50, 0)
            y = max(temp_img["region"]["y"] - 50, 0)
            w = temp_img["region"]["w"] + 100
            h = temp_img["region"]["h"] + 100
            face = img[y:y+h, x:x+w]
            face_test = img
            action(encodings, dictionary, face, frame, img, x,y,w,h)
    except:
        print(f'No face in frame')

def change_video(path):
    array = video_frames(path)
    frames = array[0]
    fps = array[1]
    for x in range(650,750):
        frames.append(f'frames/frame_{x}.jpg')
    for x in frames:
        print(x)
        detect_faces_and_swap(x)
    video_join(frames, fps)

change_video("static/sample3.mp4")


