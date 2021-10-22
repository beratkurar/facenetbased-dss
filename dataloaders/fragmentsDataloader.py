"""This module was imported from liorshk's 'facenet_pytorch' github repository:
        https://github.com/liorshk/facenet_pytorch/blob/master/LFWDataset.py

    It was modified to support lfw .png files for loading by using the code here:
        https://github.com/davidsandberg/facenet/blob/master/src/lfw.py#L46
"""

"""MIT License

Copyright (c) 2017 

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE."""

import torchvision.datasets as datasets
import os
import numpy as np
from tqdm import tqdm
import random

class FragmentsDataset(datasets.ImageFolder):
    def __init__(self, dir, transform=None, numPairs =1000):

        super(FragmentsDataset, self).__init__(dir, transform)
        self.numPairs = numPairs
        # LFW dir contains 2 folders: faces and lists
        self.validation_images = self.get_lfw_paths(dir)


    def read_lfw_pairs(self, imagesFolder):
        imDict = {}
        dirs = os.listdir(imagesFolder)
        for d in dirs:
            dirPath = os.path.join(imagesFolder,d)
            for im in os.listdir( dirPath):
                image = os.path.join(dirPath,im)
                imDict[image] = d
        return imDict

    def get_lfw_paths(self, imageFolder):

        imDict = self.read_lfw_pairs(imageFolder)
        allImages = list(imDict.keys())
        path_list = []
        issame_list = []
        progress_bar = tqdm(range(self.numPairs))

        for _ in progress_bar:
            firstImage = random.choice(allImages)
            seconfImage = random.choice(allImages)
            while firstImage == seconfImage:
                seconfImage = random.choice(allImages)
            issame = imDict[firstImage] == imDict[seconfImage]
            path_list.append((firstImage, seconfImage, issame))
            issame_list.append(issame)

        return path_list

    def __getitem__(self, index):
        """
        Args:
            index: Index of the triplet or the matches - not of a single image
        Returns:
        """

        def transform(img_path):
            """Convert image into numpy array and apply transformation
               Doing this so that it is consistent with all other datasets
               to return a PIL Image.
            """

            img = self.loader(img_path)
            return self.transform(img)

        (path_1, path_2, issame) = self.validation_images[index]
        img1, img2 = transform(path_1), transform(path_2)
        return img1, img2, issame

    def __len__(self):
        return len(self.validation_images)
