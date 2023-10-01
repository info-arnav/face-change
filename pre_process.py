from parse_xml import parse_xml_annotation
import os

def load_and_preprocess_data(data_dir):
    images = []
    annotations = []

    for filename in os.listdir(data_dir):
        if filename.endswith('.jpg'):
            image_path = os.path.join(data_dir, filename)
            annotation_file = os.path.join(data_dir, filename.replace('.jpg', '.xml'))

            # Parse the annotation file
            annotation_info = parse_xml_annotation(annotation_file)

            images.append(image_path)
            annotations.append(annotation_info)

    return images, annotations