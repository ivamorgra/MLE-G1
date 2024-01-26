import xml.etree.ElementTree as ET    

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