"""
XUYIZHI 11/13/23
Dataset package > __init__.py

Containing functions to find, create, and forward data to training pipeline.
"""
import os
from xml.etree import ElementTree as ET

def read_annotation(path) -> dict:
    tree = ET.parse(path)
    return tree

def get_bounding_boxes(xml_elementTree):
    annotation = xml_elementTree
    root = annotation.getroot()
    labels = []
    bounding_boxes = []
    for child in root:
        if child.tag == "object":
            labels.append(child[0].text)
            bounding_box = [int(element.text) for element in child[3]]
            bounding_boxes.append(bounding_box)
    return labels, bounding_boxes
