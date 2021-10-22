


import os
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset
import random
from collections import defaultdict
from cropRectangle import ImageRactangle# getLagestRectangle
from emnist import extract_test_samples


class EmnistDatasetTest(Dataset):
    def __init__(self, data_transforms):
        images, labels = extract_test_samples('byclass')
        self.images = images
        self.labels = labels
        self.transform = data_transforms

    def __len__(self):
        return len(self.images)



    def __getitem__(self, index):
        img = self.images[index]
        label = self.labels[index]
        img = Image.fromarray(img)
        img = self.transform(img)
        return img, label


