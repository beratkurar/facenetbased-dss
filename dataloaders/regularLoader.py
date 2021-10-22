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
import pandas as pd

from collections import defaultdict
import random

from PIL import Image

class LettersDataset(datasets.ImageFolder):
    def __init__(self, dir, manuToPeriodMapPath = r"/home/olya/Documents/scrollPeriods/scroll2Period.csv",  transform=None):

        super(LettersDataset, self).__init__(dir, transform)
        self.imageToClass = self.get_imageToPeriod(dir, manuToPeriodMapPath)

        self.allImages = list(self.imageToClass.keys())


    def read_lfw_pairs(self, imagesFolder):
        imDict = {}
        dirs = os.listdir(imagesFolder)
        for d in dirs:
            dirPath = os.path.join(imagesFolder,d)
            for im in os.listdir( dirPath):
                image = os.path.join(dirPath,im)
                imDict[image] = d
        return imDict

    def get_imageToPeriod(self, imageFolder, manuToPeriodMapPath):

        imDict = self.read_lfw_pairs(imageFolder)
        #map to final class
        manuToPeriodMap = pd.read_csv(manuToPeriodMapPath, index_col="manuscript").dropna().T.to_dict()
        manuToPeriodDict = defaultdict()
        for manu, val in manuToPeriodMap.items():
            period = val["period"]
            if period != "roman":
                manuToPeriodDict[manu] = period

        imDict = { image : manuToPeriodDict[manu] for image, manu in imDict.items() if manu in manuToPeriodDict}
        return imDict


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
            return self.transform(img), img

        imagePath = self.allImages[index]
        img, raw_img = transform(imagePath)
        raw_img = np.array(raw_img.resize((28,28)))
        label  =  self.imageToClass[imagePath]
        return img, raw_img, label

    def __len__(self):
        return len(self.allImages)



from dataloaders.triplet_loss_dataloader import getBinarisedImage

from cropRectangle import ImageRactangle#getLagestRectangle
from PIL import Image


def getRectangle(imgName, img,  addBinary, rec, orentation, img_bin = None, isBinary = False):

    if imgName.split("/")[-1].startswith("P662-Fg007-R-C01-R01-D120"):
        print("")
    center, side, allRec = rec.getLagestRectangle(imgName)

    # imgName = imgName.split("/")[-1]
    # if imgName in orentation:
    #     orent = orentation[imgName]
    #     if orent == 'c':
    #         # print("right")
    #         img = img.rotate(angle=240)
    #         img_bin = img_bin.rotate(angle=240) if img_bin is not None else img_bin
    #     elif orent == "v":
    #         img = img.rotate(angle=90)
    #         img_bin = img_bin.rotate(angle=90) if img_bin is not None else img_bin
    #         # print("left")
    #     elif orent == "x":
    #         # print("upside")
    #         img = img.rotate(angle=180)
    #         img_bin = img_bin.rotate(angle=180) if img_bin is not None else img_bin
    #     else:
    #         pass
    #         # print("orent {} regular".format(orent))

    w,h = center
    show= False
    if img_bin:
        img_bin = img_bin.resize(img.size)
    if isBinary:
        img = img.resize( (img.size[0]//2, img.size[1]//2))
    im = img.crop((h-side, w-side, h+side, w+side))
    if show:
        im.show()


    if addBinary:
        img_bin = img_bin.crop((h-side, w-side, h+side, w+side))
        img_bin = np.array(img_bin)
        img = np.append(np.array(im), img_bin.reshape((img_bin.shape[0], img_bin.shape[1], 1)), axis=2)
        im = Image.fromarray(img)
    return im, side

class FragmentsDataset(datasets.ImageFolder):
    def __init__(self, dir,  transform=None, isTest = False, addBinary = True):

        super(FragmentsDataset, self).__init__(dir, transform)
        self.imageToClass = self.read_lfw_pairs(dir)

        self.allImages = list(self.imageToClass.keys())
        self._isTest  =isTest
        self.addBinary = addBinary
        self.rec = ImageRactangle(isTest)



    def read_lfw_pairs(self, imagesFolder):
        imDict = {}
        dirs = os.listdir(imagesFolder)
        for d in dirs:
            dirPath = os.path.join(imagesFolder,d)
            for im in os.listdir( dirPath):
                if im.endswith("npy"):
                    continue
                if im.startswith("P662-Fg007-R-C01-R01-D120"):
                    print("")
                image = os.path.join(dirPath,im)

                if os.path.isfile(image) :
                    imDict[image] = d

        man2im=defaultdict(int)

        for i,man in imDict.items():
            man2im[man]+=1
        allJ= len(list(imDict.keys()))*(len(list(imDict.keys()))-1)
        posJ=0
        for m, cnt in man2im.items():
            posJ+=cnt*(cnt-1)

        #posPerc = posJ/allJ
        return imDict

    def __len__(self):
        return len(self.allImages)



    def __getitem__(self, index):
        """
        Args:
            index: Index of the triplet or the matches - not of a single image
        Returns:
        """


        def transform(img):
            """Convert image into numpy array and apply transformation
               Doing this so that it is consistent with all other datasets
               to return a PIL Image.
            """


            return self.transform(img), img

        imagePath = self.allImages[index]
        if self.addBinary:
            img_bin_path  = None
            while img_bin_path is None:
                imagePath = self.allImages[index]
                #img = self.loader(imagePath)
                img_bin_path = getBinarisedImage(imagePath)
                if img_bin_path is None:
                    index += 1
            img_bin = Image.open(img_bin_path)
        else:
            img_bin=None



        # Modified to open as PIL image in the first place
        img = Image.open(imagePath).convert("RGB")
        img, side = getRectangle(imagePath, img, self.addBinary, img_bin=img_bin, rec = self.rec, orentation= self.rec.orentation)


        # if self.addBinary:
        #     img_bin = np.array(img_bin.resize(img.size))
        #     img = np.append(np.array(img), img_bin.reshape((img_bin.shape[0], img_bin.shape[1], 1)), axis=2)
        #     img = Image.fromarray(img)

        img, raw_img = transform(img)
        raw_img = np.array(raw_img.resize((28,28)))
        label  =  self.imageToClass[imagePath]

        if self._isTest:
            label += ":"+imagePath
        addSide = False
        if addSide:
            return img, side, raw_img, label
        else:
            return img,  raw_img, label


    def __len__(self):
        return len(self.allImages)

def isValidBinRectangle(image):
    import numpy as np
    recFile = image[:-4]+".npy"

    d = np.load(recFile, allow_pickle='TRUE').item()
    center = d['center']
    side = d['side']
    w, h = center
    image = Image.open(image)
    image = image.resize((image.size[0] // 2, image.size[1] // 2))
    im = image.crop((h - side, w - side, h + side, w + side))

    arr = np.array(im)
    arr[arr < 100] = 1
    arr[arr > 100] = 0
    h, w = arr.shape
    black = arr.sum() / (h * w)

    if black > 0.1 and black < 0.35:
        return True
    return False


class FragmentsDatasetBinary(datasets.ImageFolder):
    def __init__(self, dir,  transform=None, isTest = False):

        super(FragmentsDatasetBinary, self).__init__(dir, transform)
        self.imageToClass = self.read_lfw_pairs(dir)

        self.allImages = list(self.imageToClass.keys())
        self._isTest  =isTest
        self.rec = ImageRactangle(isTest)


    def read_lfw_pairs(self, imagesFolder):
        imDict = {}
        dirs = os.listdir(imagesFolder)
        for d in dirs:
            if not d.endswith("_bin2"):
                continue

            dirPath = os.path.join(imagesFolder,d)
            for im in os.listdir( dirPath):
                if im.endswith(".npy"):
                    continue
                image = os.path.join(dirPath,im)

                if os.path.isfile(image) and isValidBinRectangle(image):
                    imDict[image] = d
        return imDict

    def __len__(self):
        return len(self.allImages)


    def __getitem__(self, index):
        """
        Args:
            index: Index of the triplet or the matches - not of a single image
        Returns:
        """

        def transform(img):
            """Convert image into numpy array and apply transformation
               Doing this so that it is consistent with all other datasets
               to return a PIL Image.
            """

            return self.transform(img), img


        imagePath = self.allImages[index]
        # Modified to open as PIL image in the first place
        img = Image.open(imagePath)
        img, side = getRectangle(imagePath, img, addBinary=False, rec=self.rec,
                                 orentation=self.rec.orentation, isBinary=True)


        if imagePath == r'/home/olya/Documents/fragmentsData/DSS_Joins_Test_bw/4Q262_bin2/105-Fg003-R-C01-R01-D03012012-T135030-LR924_012_F.jpg':
            print("")
        img, raw_img = transform(img)

        raw_img = np.array(raw_img.resize((28, 28)))
        label = self.imageToClass[imagePath]
        label = label.replace("_bin2", "")
        if self._isTest:
            label += ":" + imagePath
        return img, raw_img, label

    def __len__(self):
        return len(self.allImages)




class FragmentsDatasetBinaryLetters(datasets.ImageFolder):
    def __init__(self, dir,  transform=None, isTest = False):

        super(FragmentsDatasetBinaryLetters, self).__init__(dir, transform)
        self.imageToClass = self.read_lfw_pairs(dir)

        self.allImages = list(self.imageToClass.keys())
        self._isTest  =isTest



    def read_lfw_pairs(self, imagesFolder):
        imDict = {}
        dirs = os.listdir(imagesFolder)
        for d in dirs:
            if not d.endswith("_bin"):
                continue

            dirPath = os.path.join(imagesFolder,d)
            for im in os.listdir( dirPath):
                if im.endswith(".npy"):
                    continue
                image = os.path.join(dirPath,im)

                #if os.path.isfile(image) and isValidBinRectangle(image):
                imDict[image] = d
        return imDict

    def __len__(self):
        return len(self.allImages)


    def __getitem__(self, index):
        """
        Args:
            index: Index of the triplet or the matches - not of a single image
        Returns:
        """

        def transform(img):
            """Convert image into numpy array and apply transformation
               Doing this so that it is consistent with all other datasets
               to return a PIL Image.
            """

            return self.transform(img), img


        imagePath = self.allImages[index]
        # Modified to open as PIL image in the first place
        img = Image.open(imagePath)
        img, raw_img = transform(img)

        raw_img = np.array(raw_img.resize((28, 28)))
        label = self.imageToClass[imagePath]
        label = label.replace("_bin", "")
        if self._isTest:
            label += ":" + imagePath
        return img, raw_img, label

    def __len__(self):
        return len(self.allImages)
