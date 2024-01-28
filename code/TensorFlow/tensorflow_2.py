
import json
from matplotlib import pyplot as plt
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import cv2
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from time import time
#from Extra.utils import clear_directory,calculate_iou, parse_xml, find_xml_file
import os
import shutil
import math
from statistics import mean
import xml.etree.ElementTree as ET

##############extra####################
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

def calculate_iou(box1, box2):
    # Convertir las coordenadas [x, y, w, h] a [x_min, y_min, x_max, y_max]
    box1_converted = [box1[0], box1[1], box1[2] + box1[0], box1[3] + box1[1]]
    box2_converted = [box2[0], box2[1], box2[2] + box2[0], box2[3] + box2[1]]

    # Calcular las coordenadas de la intersección
    inter_xmin = np.maximum(box1_converted[0], box2_converted[0])
    inter_ymin = np.maximum(box1_converted[1], box2_converted[1])
    inter_xmax = np.minimum(box1_converted[2], box2_converted[2])
    inter_ymax = np.minimum(box1_converted[3], box2_converted[3])

    # Calcular el área de la intersección
    inter_area = np.maximum(0, inter_xmax - inter_xmin) * np.maximum(0, inter_ymax - inter_ymin)

    # Calcular el área de ambas cajas
    box1_area = (box1_converted[2] - box1_converted[0]) * (box1_converted[3] - box1_converted[1])
    box2_area = (box2_converted[2] - box2_converted[0]) * (box2_converted[3] - box2_converted[1])

    # Calcular IoU
    iou = inter_area / (box1_area + box2_area - inter_area + 1e-10)  # Añadir pequeño valor para evitar división por cero
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

def get_scores(results, threshold):
    TP = 0
    FP = 0
    FN = 0
    # R2 is non-significant because all actual values are 1
    mse = 0
    mae = 0
    mlse = 0

    n = len(results)
    for result in results:
        if result == 0:
            FN += 1
        elif result < threshold:
            FP += 1
        else:
            TP += 1

        mse += (1 - result) ** 2
        mae += abs(1 - result)
        mlse += (math.log(1 + 1) - math.log(1 + result))**2
    mse /= n
    mae /= n
    mlse /= n

    precision = TP/(TP + FP)
    recall = TP/(TP + FN)

    f1 = 2 * (precision * recall) / (precision + recall)

    return {"average_iou" : mean(results),
            "precision" : precision,
            "recall" : recall,
            "f1" : f1,
            "mse" : mse, 
            "mae" : mae, 
            "mlse" : mlse, }
# URL del modelo preentrenado en COCO

model_url = "https://tfhub.dev/tensorflow/efficientdet/d7/1"

# Cargar el modelo desde TensorFlow Hub
#model = hub.load(model_url)
model = hub.load("https://www.kaggle.com/models/tensorflow/efficientdet/frameworks/TensorFlow2/variations/d7/versions/1")

# Lista para almacenar las cajas delimitadoras predichas
predicted_boxes_list = []
inicio = time()

folder_path = "../../testImagesNet/Data/"
annotations_directory = "../../testImagesNet/Annotations/"
output_directory = "../../output/images/tensorflow_pretrained/"
clear_directory(output_directory)

image_list = []
image_path_list = []

# List all files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith(('.jpg', '.png', '.jpeg', '.JPEG')):  # You can add more extensions if needed
        # Construct the full file path
        file_path = os.path.join(folder_path, filename)
        
        # Read the image and append it to the list
        img = cv2.imread(file_path)
        image_list.append(img)
        image_path_list.append(file_path)
    


true_boxes_list = []

# Recorrer las imágenes y obtener las anotaciones de ground truth

resources_directory = "../../resources/"

# Load COCO class labels
class_labels = []
with open(resources_directory + "yolo_coco.names", "rt") as f:
    class_labels = f.read().rstrip('\n').split('\n')

# Lista para almacenar las cajas delimitadoras predichas
predicted_boxes_list_tf = []
results = {}
# Realizar inferencias en cada imagen
for image_path in image_path_list:

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Realizar inferencias en la imagen
    outputs = model([image])

    # Obtener las detecciones
    boxes = outputs["detection_boxes"].numpy()
    scores = outputs["detection_scores"].numpy()
    classes = outputs["detection_classes"].numpy().astype(np.int32)

    # (Opcional) Filtrar detecciones por confianza
    confidence_threshold = 0.5
    filtered_boxes = boxes[scores > confidence_threshold]
    predicted_boxes_list_tf.append(filtered_boxes)

    xml_data = parse_xml(find_xml_file(annotations_directory, image_path))
    
    for target in xml_data:
        iou = 0

        searched_label = translate_coco_to_images_net(target["name"])

        image_iou_dict = {}
        if searched_label in results:
            image_iou_dict = results[searched_label]
        if image_path not in image_iou_dict:
            image_iou_dict[image_path] = []

        for predicted_boxes in predicted_boxes_list_tf:
            for box in predicted_boxes:

                # Obtener el tamaño de la imagen
                image_size = image.shape[:2]

                # Convertir las coordenadas normalizadas a píxeles reales
                box_minmax = [
                    box[0] * image_size[1],  # x_min
                    box[1] * image_size[0],  # y_min
                    (box[2] + box[0]) * image_size[1],  # x_max
                    (box[3] + box[1]) * image_size[0]   # y_max
                ]

                # Calcular IoU
                new_iou = calculate_iou(box_minmax, [target["xmin"], target["ymin"], target["xmax"], target["ymax"]])

                if np.any(new_iou > iou):
                    iou = new_iou

        # Añadir el resultado al diccionario
        image_iou_dict[image_path].append(iou)
        results[searched_label] = image_iou_dict

duracion = time() - inicio

# Visualización de resultados en gráficos y exportación a CSV
score_export_folder_path = "../../output/scores/"

data = results["dog"]

merged_array = np.concatenate(list(data.values()))

# Create bar chart
plt.bar(range(len(merged_array)), merged_array)

# Set the y-axis limits to ensure the range is between 0 and 1
plt.ylim(0, 1)

# Add labels and title
plt.xlabel('Detecciones')
plt.ylabel('Intersection over Union')
plt.title('Tensorflow entrenado con COCO: detección de perros')

# Show the plot
plt.show()


#Métricas de error

data = results["dog"]
threshold = 0.5
merged_array = np.concatenate(list(data.values()))

scores = get_scores(merged_array, threshold)

print("Average IoU: " + str(scores["average_iou"]))
print("Precision: " + str(scores["precision"]))
print("Recall: " + str(scores["recall"]))
print("F1-score: " + str(scores["f1"]))
print("MSE: " + str(scores["mse"]))
print("MAE: " + str(scores["mae"]))
print("MLSE: " + str(scores["mlse"]))

scores["time"] = duracion

score_export_file_path = "tensorflow_opencv_scores.json"
# Export the dictionary to a JSON file
with open(score_export_folder_path + score_export_file_path, 'w') as file:
    json.dump(scores, file)

data = results["car"]

merged_array = np.concatenate(list(data.values()))

scores = get_scores(merged_array, threshold)

print("Average IoU: " + str(scores["average_iou"]))
print("Precision: " + str(scores["precision"]))
print("Recall: " + str(scores["recall"]))
print("F1-score: " + str(scores["f1"]))
print("MSE: " + str(scores["mse"]))
print("MAE: " + str(scores["mae"]))
print("MLSE: " + str(scores["mlse"]))

scores["time"] = duracion

score_export_file_path = "yolo_car_opencv_scores.json"
# Export the dictionary to a JSON file
with open(score_export_folder_path + score_export_file_path, 'w') as file:
    json.dump(scores, file)