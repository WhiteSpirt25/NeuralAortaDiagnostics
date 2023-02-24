import os
import pickle

import cv2
import numpy as np
from torch.utils import data
import torchvision.transforms as T
import torchvision.transforms.functional as TF



def load_sparce_npz(path: str):
    """Loads npy array as sparce pickled scipy matrix."""

    with open(path, "rb") as file:
        s = pickle.load(file)

    # convert to numpy array
    s = s.todense()
    if len(s.shape) == 2:
        return s
    return np.transpose(s, [1, 2, 0])


class SparseDatasetNpz(data.Dataset):
    """Custom PyTorch dataset created for working with pickled sparse images.

    Masks are images stored as pickled sparse matrixes from "sparse" library.
    Expected data structure:
    folder
    └───images
    │   │ [filename 1].jpg
    │   │ [filename 2].jpg
    │   │ ...
    └───masks
    │   │   [filename 1].pickle
    │   │   [filename 2].pickle
    │   │   ...
    """

    # initialise function of class
    def __init__(
        self,
        root,
        augmentations=None,
        image_only_aug=None,
        mask_only_aug=None,
        preprocessing=None,
    ):
        # the data directory
        self.root = root
        # the list of filename
        self.filenames = os.listdir(os.path.join(root, "images"))
        # transforms
        self.augmentation = augmentations
        self.image_only_aug = image_only_aug
        self.mask_only_aug = mask_only_aug
        self.preprocessing = preprocessing

    
    def __getitem__(self, index):
        # obtain filenames from list
        image_filename = self.filenames[index]

        # Load data
        image = cv2.imread(os.path.join(self.root, "images", image_filename))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Load label
        pre, ext = os.path.splitext(image_filename)
        mask_filename = pre + ".pickle"
        mask = load_sparce_npz(os.path.join(self.root, "masks", mask_filename))

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample["image"], sample["mask"]

        if self.image_only_aug:
            sample = self.image_only_aug(image=image)
            image = sample["image"]

        if self.mask_only_aug:
            sample = self.mask_only_aug(mask=mask)
            mask = sample["mask"]

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample["image"], sample["mask"]

        image = image.float()
        
        return image, mask

    # the total number of samples (optional)
    def __len__(self):
        return len(self.filenames)