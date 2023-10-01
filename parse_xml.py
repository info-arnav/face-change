import os
import xml.etree.ElementTree as ET

def parse_xml_annotation(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Extract class and bounding box information
    class_name = root.find('object/name').text
    xmin = int(root.find('object/bndbox/xmin').text)
    ymin = int(root.find('object/bndbox/ymin').text)
    xmax = int(root.find('object/bndbox/xmax').text)
    ymax = int(root.find('object/bndbox/ymax').text)

    return {
        'class_name': class_name,
        'bbox': [xmin, ymin, xmax, ymax]
    }