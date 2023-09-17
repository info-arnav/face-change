import insightface
import os
import onnxruntime
import onnx
import cv2
import threading
import cv2
import numpy
import numpy as np
from deepface import DeepFace
import matplotlib.pyplot as plt
from gfpgan.utils import GFPGANer
import threading

available = onnxruntime.get_available_providers()

provider = [available[0]]

print("Using", provider)

reference_face_position = 0
similar_face_distance = 0.85

swapper = insightface.model_zoo.get_model("inswapper_128.onnx", providers=provider)
enhancer = GFPGANer(model_path="GFPGANv1.4.pth", upscale=1, device="cpu")
lock = threading.Lock()
face_annalyser = None

def annalyser():
    global face_annalyser
    with lock:
        if face_annalyser is None:
            face_annalyser = insightface.app.FaceAnalysis(name='buffalo_l', providers=provider)
            face_annalyser.prepare(ctx_id=0)
    return face_annalyser

def get_many_faces(frame):
    try:
        return annalyser().get(frame)
    except ValueError:
        return None

def get_one_face(frame, position = 0):
    many_faces = get_many_faces(frame)
    if many_faces:
        try:
            return many_faces[position]
        except IndexError:
            return many_faces[-1]
    return None

def find_similar_face(frame, reference_face):
    many_faces = get_many_faces(frame)
    if many_faces:
        for face in many_faces:
            if hasattr(face, 'normed_embedding') and hasattr(reference_face, 'normed_embedding'):
                distance = numpy.sum(numpy.square(face.normed_embedding - reference_face.normed_embedding))
                if distance < similar_face_distance:
                    return face
    return None

def enhance_face(target_face, temp_frame):
    start_x, start_y, end_x, end_y = map(int, target_face['bbox'])
    padding_x = int((end_x - start_x) * 0.5)
    padding_y = int((end_y - start_y) * 0.5)
    start_x = max(0, start_x - padding_x)
    start_y = max(0, start_y - padding_y)
    end_x = max(0, end_x + padding_x)
    end_y = max(0, end_y + padding_y)
    temp_face = temp_frame[start_y:end_y, start_x:end_x]
    if temp_face.size:
        with threading.Semaphore():
            _, _, temp_face = enhancer.enhance(
                temp_face,
                paste_back=True
            )
        temp_frame[start_y:end_y, start_x:end_x] = temp_face
    return temp_frame

def process_frame(source_face, reference_face, temp_frame):
    target_face = find_similar_face(temp_frame, reference_face)
    if target_face:
        try:
            temp_frame = swapper.get(temp_frame, target_face, source_face, paste_back=True)
            temp_frame = enhance_face(source_face, temp_frame)
        except:
            print("Face not found in the image provided")
#             Exception handinling for wrong face
    return temp_frame


def swap(new_face,image): 
    source_face = get_one_face(new_face)
    target_frame = image
    reference_face = get_one_face(target_frame, reference_face_position)
    result = process_frame(source_face, reference_face, target_frame)
    return result

def face_image_resize(img, w,h):
    detected_faces = DeepFace.analyze(img, actions = ["gender"], enforce_detection=False)
    detected = False
    detected_face = img
    if len(detected_faces) > 0:
        for temp_img in detected_faces[:1]:
            x = max(temp_img["region"]["x"] - 50, 0)
            y = max(temp_img["region"]["y"] - 50, 0)
            w = temp_img["region"]["w"] + 100
            h = temp_img["region"]["h"] + 100
            detected_face = img[y:y+h, x:x+w]
            detected = True
    new_image = cv2.resize(detected_face, (w, h))
    return new_image

def replace_face(path, images=[]):
    img = cv2.imread(path)
    obj = DeepFace.analyze(img, actions = ["gender"])
    temp_obj = []
    i = 0
    test = ""
    face = ""
    for temp_img in obj:
        sub_path = ""
        if len(images) > i:
            sub_path = images[i]
        elif len(images) == i:
            i = 0
            sub_path = f'static/{i}.jpg'
        else:
            sub_path = f'static/{i}.jpg'
        xo = temp_img["region"]["x"]
        yo = temp_img["region"]["y"]
        wo = temp_img["region"]["w"]
        ho = temp_img["region"]["h"]
        x = max(temp_img["region"]["x"] - 50, 0)
        y = max(temp_img["region"]["y"] - 50, 0)
        w = temp_img["region"]["w"] + 100
        h = temp_img["region"]["h"] + 100
        print("Gender : " + temp_img["dominant_gender"])
        print("Face : ")
        test = img[y:y+h, x:x+w]
        plt.imshow(img[y:y+h, x:x+w][:,:,::-1])
        plt.show()
        if bool(int(input("Change face 0/1? "))):
            # Can use temp_img object gender key to change images with same gender
            print(f'using {sub_path}')
            face = face_image_resize(cv2.imread(sub_path), w,h)
            new_face_image = swap(face, test)
            img[y:y+h, x:x+w] = new_face_image
            i = i + 1
        print("")
    cv2.imwrite("output.jpg", img)
    print("image saved")
    plt.imshow(img[:,:,::-1])
    plt.show()




