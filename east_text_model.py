import cv2
import numpy as np

def calculate_intersection_area(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    x_intersection = max(x1, x2)
    y_intersection = max(y1, y2)
    w_intersection = min(x1 + w1, x2 + w2) - x_intersection
    h_intersection = min(y1 + h1, y2 + h2) - y_intersection

    if w_intersection > 0 and h_intersection > 0:
        intersection_area = w_intersection * h_intersection
        return intersection_area
    else:
        return 0

def find_text_boxes(image):
    east_model_path = 'models/frozen_east_text_detection.pb'
    net = cv2.dnn.readNet(east_model_path)

    net_shape = (320, 320)
    height, width = image.shape[:2]
    rW = width / float(net_shape[0])
    rH = height / float(net_shape[1])
    blob = cv2.dnn.blobFromImage(image, scalefactor=1.0, size=net_shape, mean=(123.68, 116.78, 103.94), swapRB=True, crop=False)

    net.setInput(blob)
    (scores, geometry) = net.forward(["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"])

    min_confidence = 0.5
    boxes = []
    confidences = []

    for y in range(0, scores.shape[2]):
        scores_data = scores[0, 0, y]
        x0 = geometry[0, 0, y]
        x1 = geometry[0, 1, y]
        x2 = geometry[0, 2, y]
        x3 = geometry[0, 3, y]
        angles = geometry[0, 4, y]

        for x in range(0, scores.shape[3]):
            if scores_data[x] < min_confidence:
                continue

            offset_x = x * 4.0
            offset_y = y * 4.0

            angle = angles[x]
            cos = np.cos(angle)
            sin = np.sin(angle)

            h = x0[x] + x2[x]
            w = x1[x] + x3[x]

            end_x = int(offset_x + (cos * x1[x]) + (sin * x2[x]))
            end_y = int(offset_y - (sin * x1[x]) + (cos * x2[x]))
            start_x = int(end_x - w)
            start_y = int(end_y - h)

            boxes.append((start_x, start_y, end_x, end_y))
            confidences.append(float(scores_data[x]))

    indices = cv2.dnn.NMSBoxes(boxes, confidences, min_confidence, 0.4)

    array = []

    if isinstance(indices, np.ndarray) and len(indices) > 0:
        for i in range(len(indices)):
            index = int(indices[i])
            start_x, start_y, end_x, end_y = boxes[index]
            intersection_area = 0
            for existing_box in array:
                intersection_area += calculate_intersection_area((start_x, start_y, end_x - start_x, end_y - start_y),(existing_box[0], existing_box[1], existing_box[2] - existing_box[0], existing_box[3] - existing_box[1]))
            box_area = (end_x - start_x) * (end_y - start_y)
            adjusted_area = max(0, box_area - intersection_area)
            array.append([int(start_x * rW), int(start_y * rH), int(end_x * rW), int(end_y * rH), adjusted_area])

    return array
