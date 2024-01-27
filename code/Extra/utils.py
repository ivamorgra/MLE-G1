import os
import shutil

import xml.etree.ElementTree as ET

def clear_directory(directory_path):
    if os.path.exists(directory_path):
        shutil.rmtree(directory_path)
    os.makedirs(directory_path)

def parse_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    annotations = []

    for object_elem in root.findall(".//object"):
        name_elem = object_elem.find("name")
        bbox_elem = object_elem.find("bndbox")

        if name_elem is not None and bbox_elem is not None:
            name = name_elem.text
            xmin = int(bbox_elem.find("xmin").text)
            ymin = int(bbox_elem.find("ymin").text)
            xmax = int(bbox_elem.find("xmax").text)
            ymax = int(bbox_elem.find("ymax").text)

            annotation = {
                'name': name,
                'xmin': xmin,
                'ymin': ymin,
                'xmax': xmax,
                'ymax': ymax
            }

            annotations.append(annotation)

    return annotations

def translate_coco_to_images_net(searched_label):
    label = ""
    if (searched_label in ["n02105855", "n02114548", "n02089973", "n02097209", "n02111277", "n02112018", "n02093859"]) :
        label = "dog"
    elif (searched_label in ["n02930766", "n03930630"]) :
        label = "car"
    elif (searched_label in ["n07742313"]) :
        label = "apple"
    return label

def find_xml_file(path, jpeg_filename):
    # Get the base filename without extension
    base_filename = os.path.splitext(os.path.basename(jpeg_filename))[0]

    # Construct the XML filename
    xml_filename = os.path.join(path, f"{base_filename}.xml")

    # Check if the XML file exists
    if os.path.exists(xml_filename):
        return xml_filename
    else:
        return None
    
def calculate_iou(box1, box2):
    # Calculate the intersection coordinates
    inter_xmin = max(box1[0], box2[0])
    inter_ymin = max(box1[1], box2[1])
    inter_xmax = min(box1[2], box2[2])
    inter_ymax = min(box1[3], box2[3])

    # Calculate the area of intersection
    inter_area = max(0, inter_xmax - inter_xmin + 1) * max(0, inter_ymax - inter_ymin + 1)

    # Calculate the area of both boxes
    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    # Calculate IoU
    iou = inter_area / float(box1_area + box2_area - inter_area)
    return iou

def calculate_absolute_difference(box1, box2):
    # Calculate absolute differences for each coordinate
    diff_xmin = abs(box1[0] - box2[0])
    diff_ymin = abs(box1[1] - box2[1])
    diff_xmax = abs(box1[2] - box2[2])
    diff_ymax = abs(box1[3] - box2[3])

    # You can sum up the absolute differences or use any other measure
    total_absolute_difference = diff_xmin + diff_ymin + diff_xmax + diff_ymax
    return total_absolute_difference