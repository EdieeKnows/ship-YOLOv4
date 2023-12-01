from torch.utils.data import DataLoader

# 每个dataLoader的灵魂，返回一个batch的数据
def custom_collate_fn(batch):
    """
    Custom collate function for DataLoader.

    This function unpacks each item in a batch and combines them by corresponding elements.
    It is used to preprocess batches of data before they are fed into a model.

    Parameters:
    batch: list
        A list of data items to be combined into a batch.

    Returns:
    iterator
        An iterator of combined elements from the batch.
    """
    res = zip(*batch)
    return res

class SeaShipDataLoader(DataLoader):
    """
    A custom DataLoader for sea ship data with a specific collate function.

    This DataLoader is tailored for handling sea ship data, and uses a custom collate function
    to preprocess the batches of data.

    Inherits from PyTorch's DataLoader.
    """
    def __init__(self, *args, **kwargs):
        """
        Initialize the SeaShipDataLoader with custom collate function.

        Parameters:
        *args: Variable length argument list for DataLoader.
        **kwargs: Arbitrary keyword arguments for DataLoader.
        """
        super(SeaShipDataLoader, self).__init__(*args, collate_fn=custom_collate_fn, **kwargs)