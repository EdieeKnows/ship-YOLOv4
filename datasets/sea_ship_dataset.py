"""
CETC: YOLOv4 Ship Detection
XUYIZHI 11/13/23

./datasets/sea_ship_dataset.py: Custom dataset for sea ship 7000.
"""
import os
from typing import Iterable, List, Optional, Union
import cv2
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader, Sampler, _collate_fn_t, _worker_init_fn_t
import torch
import datasets

class SeaShipDataset(Dataset):
    def __init__(self, 
                 annotations_file : str, 
                 image_file : str, 
                 transforms = None) -> None:
        self.annotations_dir = annotations_file
        self.img_dir = image_file
        self.transforms = transforms
        self.imgs = list(sorted(os.listdir(self.img_dir)))
        self.annotations = list(sorted(os.listdir(self.annotations_dir)))

    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, self.imgs[index])
        annotation_path = os.path.join(self.annotations_dir, self.annotations[index])
        img = read_image(img_path)
        annotation = datasets.read_annotation(annotation_path)
        labels = datasets.get_labels(annotation)
        bounding_boxes = datasets.get_bounding_boxes(annotation)
        target = {}
        target['boxes'] = bounding_boxes
        target["labels"]  = labels
        target["image_id"] = index

        if self.transforms is not None:
            img, target['boxes'] = self.transforms(img, target['boxes'])

        return img, target
    
    def __len__(self):
        return len(self.imgs)