from abc import ABC, abstractmethod
import torch
import torchvision.transforms.functional as TF
from torchvision.utils import _log_api_usage_once

class CustomTransforms(ABC):
    """
    An abstract base class for all custom transformations that operate on an image and its corresponding target.

    This class serves as a template for defining transformations that are applied
    to both the image and its target, such as scaling, cropping, or normalization.
    Derived classes need to implement the __call__ method to specify the transformation.
    """
    def __init__(self) -> None:
        super().__init__()
    
    @abstractmethod
    def __call__(self, img, target):
        """
        Apply a transformation to an image and its target.

        Parameters:
        img : Any
            The image to be transformed.
        target : Any
            The corresponding target data to be transformed.

        Returns:
        Any
            The transformed image and target.
        """
        pass

class MyCompose:
    """
    A custom composition of transformations to be applied to an image and its target.

    This class allows for the composition of multiple transformations, both custom
    (defined by CustomTransforms) and standard torchvision transforms. The transforms
    are applied sequentially to the image and, if applicable, to the target.

    Attributes:
    transforms (list): A list of transformations to be applied.
    """

    def __init__(self, transforms):
        """
        Initialize the myCompose object with a list of transformations.

        Parameters:
        transforms (list): A list of transformations. Each transformation can be
                           an instance of CustomTransforms or a standard torchvision transform.
        """
        if not torch.jit.is_scripting() and not torch.jit.is_tracing():
            _log_api_usage_once(self)
        self.transforms = transforms

    def __call__(self, img, target):
        """
        Apply the sequence of transformations to an image and its target.

        This method sequentially applies each transformation in the list to the image.
        If a transformation is an instance of CustomTransforms, it is also applied to the target.
        Otherwise, it's applied only to the image.

        Parameters:
        img (torch.Tensor): The image to which the transformations are applied.
        target (dict): The corresponding target data (e.g., bounding boxes, labels).

        Returns:
        tuple: A tuple containing the transformed image and target.
        """
        for t in self.transforms:
            if isinstance(t, CustomTransforms):
                img, target = t(img, target)
            else:
                img = t(img)
        return img, target

    def __repr__(self) -> str:
        """
        Generate a formatted string representation of the myCompose object.

        Returns:
        str: A string representation of the myCompose object, including the list of transformations.
        """
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += f"    {t}"
        format_string += "\n)"
        return format_string

class ResizeTransform(CustomTransforms):
    """
    A transform class to resize an image and adjust its bounding box coordinates.

    This class inherits from CustomTransforms and implements the __call__ method
    to resize an input image to a new size while also adjusting the bounding box
    coordinates in the target to match the new image dimensions.

    Attributes:
    new_size (tuple): The target size for resizing the image (width, height).
    """
    def __init__(self, new_size):
        self.new_size = new_size

    def __call__(self, image, target):
        """
        Apply the resizing transformation to an image and its bounding boxes.

        The method resizes the input image and adjusts the bounding box coordinates
        within the target dictionary to match the new image dimensions.

        Parameters:
        image (torch.Tensor): The input image to be resized.
        target (dict): A dictionary containing the bounding box coordinates under 'boxes' key.

        Returns:
        tuple: A tuple containing the resized image and the updated target dictionary.
        """
        boxes = target['boxes']
        
        resized_image = TF.resize(image, self.new_size)

        # 假设图像是 (C, H, W)
        old_dims = torch.FloatTensor([image.shape[2], image.shape[1], image.shape[2], image.shape[1]]).unsqueeze(0)
        new_dims = torch.FloatTensor([self.new_size[1], self.new_size[0], self.new_size[1], self.new_size[0]]).unsqueeze(0)
        # 调整边界框坐标
        resized_boxes = boxes / old_dims * new_dims

        target['boxes'] = resized_boxes

        return resized_image, target

