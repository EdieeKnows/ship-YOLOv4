"""
XUYIZHI 11/13/23
./datasets/__init__.py

Datasets package: Providing functions & custom classes to find, create, and forward data to training pipeline.
"""

import os
from xml.etree import ElementTree as ET
from torchvision import tv_tensors

def read_annotation(path) -> dict:
    tree = ET.parse(path)
    return tree

def get_bounding_boxes(xml_elementTree):
    annotation = xml_elementTree
    root = annotation.getroot()
    bounding_boxes = []
    for child in root:
        if child.tag == "size":
            height = int(child[1].text)
            width = int(child[0].text)
        elif child.tag == "object":
            bounding_box = [int(element.text) for element in child[3]]
            bounding_boxes.append(bounding_box)
    return tv_tensors.BoundingBoxes(bounding_boxes, format="XYXY", canvas_size=(height, width))

def get_label_index(label : str):
    enum_labels = enumerate(["bulk cargo carrier",
    "container ship",
    "fishing boat",
    "general cargo ship",
    "ore carrier",
    "passenger ship"])
    for i, value in enum_labels:
        if value == label:
            return i

def get_labels(xml_elementTree):
    
    annotation = xml_elementTree
    root = annotation.getroot()
    labels = []
    for child in root:
        if child.tag == "object":
            labels.append(child[0].text)
    return labels
