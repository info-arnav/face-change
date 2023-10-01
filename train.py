import cv2
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from pre_process import load_and_preprocess_data
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import tensorflow as tf
import os
from gpu_allocation import set_allocation

set_allocation()

def preprocess_data(images, annotations, input_shape):
    processed_images = []
    processed_labels = []

    for image_path, annotation in zip(images, annotations):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 

        image = cv2.resize(image, (input_shape[1], input_shape[0]))

        image = image / 255.0

        class_name = annotation['class_name']
        bbox = annotation['bbox']

        processed_images.append(image)
        processed_labels.append({
            'class_name': class_name,
            'bbox': bbox,
        })

    processed_images = np.array(processed_images)
    processed_labels = np.array(processed_labels)

    return processed_images, processed_labels

def create_model(input_shape, num_classes):
    model = Sequential([
        Conv2D(1, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    return model

input_shape = (1080, 1920, 3)

train_images, train_annotations = load_and_preprocess_data("train_data")
val_images, val_annotations = load_and_preprocess_data("val_data")
processed_train_images, processed_train_labels = preprocess_data(train_images, train_annotations, input_shape)
processed_val_images, processed_val_labels = preprocess_data(val_images, val_annotations, input_shape)

class_to_label = {'banner': 0, 'self': 1}

train_labels_numeric = np.array([class_to_label[entry['class_name']] for entry in processed_train_labels])
val_labels_numeric = np.array([class_to_label[entry['class_name']] for entry in processed_val_labels])

num_classes = len(class_to_label)

model = create_model(input_shape, num_classes)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

processed_train_labels_one_hot = to_categorical(train_labels_numeric, num_classes=num_classes)
processed_val_labels_one_hot = to_categorical(val_labels_numeric, num_classes=num_classes)


train_data = (processed_train_images, processed_train_labels)
val_data = (processed_val_images, processed_val_labels)  

assert train_data[0].shape[1:] == input_shape, "Mismatched input shape for training data"
assert val_data[0].shape[1:] == input_shape, "Mismatched input shape for validation data"

model.fit(train_data[0], train_data[1], epochs=5, validation_data=val_data)

model.save('model/banners.h5')

print("Model saved successfully")
