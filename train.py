"""
XUYIZHI 11/13/23
.train.py

train.py : Call to start training the model.

USAGE: 
"""

import torch
import numpy as np
import datasets, losses, optimizer

from models.yolov4 import Yolov4
from datasets.sea_ship_dataset import SeaShipDataset
from torch.utils.data import DataLoader

def training_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train() # set model to training mode
    for batch, (Image, targets) in enumerate(dataloader):
        pred = model(Image)
        loss = loss_fn(pred, targets)

        # Backward propagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # print batch log information
        
def validation_loop(dataloader, model, loss_fn):
    model.eval() # set model to evaluation mode
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for Image, targets in dataloader:
            pred = model(Image)
            # calculate loss function
            test_loss += loss_fn(pred, targets).item()
            # calculate correct answers

if __name__ == "__main__":
    somewhere = None # read config file settings
    
    # prepare data & dataloader
    training_data = SeaShipDataset(
        annotations_file=somewhere, 
        image_file=somewhere, 
        transforms=None
        )
    test_data = SeaShipDataset(
        annotations_file=somewhere,
        image_file=somewhere,
        transforms=None
    )

    train_dataloader = DataLoader(
        dataset=training_data,
        batch_size=64,
        shuffle=True)
    test_dataloader = DataLoader(
        dataset=test_data,
        batch_size=64,
        shuffle=True)
    # instantiate optimizer
    # instantiate loss function
    # instantiate training model
    model = Yolov4()


