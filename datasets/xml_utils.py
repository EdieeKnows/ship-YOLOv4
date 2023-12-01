import os
from xml.etree import ElementTree as ET
from torchvision import tv_tensors 
import json

def parse_xml_directory(path):
    """
    Parse all XML files within a specified directory and return an element tree.

    This function iterates through the given directory, identifies XML files,
    and uses xml.etree to parse them into an element tree structure. It is particularly
    useful for extracting structured data from XML annotations in object detection tasks.

    Parameters:
    dir_path : str
        The path to the directory containing XML files.

    Returns:
    ElementTree
        An element tree structure representing the parsed XML data.

    Note:
    The function assumes that all files in the specified directory are XML files
    relevant to the annotations. Non-XML files in the directory may cause errors.
    """
    tree = ET.parse(path)
    return tree

def extract_bounding_boxes_from_tree(element_tree):
    """
    Extract bounding box coordinates from an element tree and return a tv_tensors.BoundingBoxes object.

    This function navigates through the provided element tree, searches for child elements
    with a specific tag (e.g., 'boundingbox'), extracts their coordinate data, and 
    compiles this data into a tv_tensors.BoundingBoxes object. This is particularly useful 
    for object detection tasks where precise bounding box coordinates are required from XML annotations.

    Parameters:
    element_tree : ElementTree
        The element tree from which bounding box data is to be extracted.

    Returns:
    tv_tensors.BoundingBoxes
        An object containing the bounding box coordinates extracted from the element tree.

    Note:
    The function assumes that the element tree structure contains child elements with
    bounding box information. Absence of such elements or any deviation in the expected
    structure may result in errors or incorrect data extraction.
    """
    annotation = element_tree
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

def extract_labels_from_tree(xml_elementTree):
    """
    Extract label information from a tree object and return a list of labels.

    This function traverses the provided tree object, identifying child elements
    that contain label information. It extracts these labels and compiles them into a list.
    This is especially useful in object detection tasks where each bounding box 
    is associated with a specific label from XML annotations.

    Parameters:
    tree : ElementTree
        The tree object from which label data is to be extracted.

    Returns:
    list
        A list of extracted labels from the tree.

    Note:
    The function expects the tree structure to have child elements containing
    label information. If such elements are missing or structured differently,
    it may lead to incorrect results or errors.
    """
    annotation = xml_elementTree
    root = annotation.getroot()
    labels = []
    for child in root:
        if child.tag == "object":
            labels.append(child[0].text)
    return labels
        
def create_classes_json(annotations_dir: str, save_path: str):
    """
    Create a classes.json file from XML annotations in a specified directory.

    This function iterates over all XML files in the given directory, extracts class
    labels from them, and then creates a JSON file that maps each class label to a unique
    identifier (integer index). This JSON file is useful for object detection tasks
    where each class needs to be uniquely identified.

    Parameters:
    annotations_dir : str
        The directory path that contains the XML annotation files.
    save_path : str
        The path where the classes.json file will be saved.

    Note:
    The function assumes that each XML file in the directory is an annotation file
    that follows a specific format with class labels. Files not matching the expected
    format may lead to incorrect results or errors.
    """
    # Helper function to parse XML and extract classes
    def parse_xml(xml_file: str):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        return [label[0].text for label in root.findall('object')]

    # Collect all unique class labels from all XML files
    class_labels = set()
    for xml_filename in os.listdir(annotations_dir):
        if xml_filename.endswith('.xml'):
            xml_path = os.path.join(annotations_dir, xml_filename)
            class_labels.update(parse_xml(xml_path))
    print(class_labels)

    # Create an enumeration of class labels
    class_enum = {label: idx for idx, label in enumerate(sorted(class_labels))}
    print(class_enum)

    # Save the enumeration to a JSON file
    save_path = os.path.join(save_path, 'classes.json')
    with open(save_path, 'w') as f:
        json.dump(class_enum, f, indent=4)

def get_label_index(label : str, class_enum):
    """
    Retrieve the index of a given label from a class enumeration.

    This function uses direct dictionary access to find the corresponding index
    for a given label in the class enumeration dictionary. If the label is not
    found, it raises a ValueError.

    Parameters:
    label : str
        The label whose index is to be found.
    class_enum : dict
        A dictionary mapping labels to their corresponding indices.

    Returns:
    int
        The index corresponding to the label.

    Raises:
    ValueError
        If the label is not found in the class enumeration.
    """
    if label in class_enum:
        return class_enum[label]
    else:
        raise ValueError(f"Label '{label}' not found in class enumeration.")