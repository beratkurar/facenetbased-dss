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

from cropRectangle import ImageRactangle#getLagestRectangle
from PIL import Image


def getRectangle(imgName, img,  addBinary,  rec, img_bin = None):

    center, side = rec.getLagestRectangle(imgName)

    w,h = center

    if img_bin:
        img_bin = img_bin.resize(img.size)
    im = img.crop((h-side, w-side, h+side, w+side))


    if addBinary:
        img_bin = img_bin.crop((h-side, w-side, h+side, w+side))
        img_bin = np.array(img_bin)
        img = np.append(np.array(im), img_bin.reshape((img_bin.shape[0], img_bin.shape[1], 1)), axis=2)
        im = Image.fromarray(img)
    return im


def getBinarisedImage(imagePath):
    import os
    from PIL import Image
    imName = imagePath.split("/")[-1]
    manu = imagePath.split("/")[-2]
    imsplit = imName.split("-")
    plate = imsplit[0]
    fragment = imsplit[1]
    binFolder = imagePath.split("/"+manu)[0]+"_bw/{}".format(manu)+'_bin'
    if not os.path.exists(binFolder):
        return None
    for f in os.listdir(binFolder):
        otherSpliyted = f.split("-")
        otherPlate = otherSpliyted[0]
        otherFragment = otherSpliyted[1]
        if otherPlate == plate and otherFragment==fragment:
            return os.path.join(binFolder, f)

    return None


def getSameImage(folderToImage):
    manu = random.choice(list(folderToImage.keys()))
    while len(folderToImage[manu]) < 2:
        manu = random.choice(list(folderToImage.keys()))
    firstImage = random.choice(folderToImage[manu])
    seconfImage = random.choice(folderToImage[manu])
    while firstImage == seconfImage:
        seconfImage = random.choice(folderToImage[manu])
    return sorted([firstImage, seconfImage])




def getSameImages(folderToImage):
    allImages = []
    for manu, images in folderToImage.items():

        for i, first in enumerate(images):
            for second in images[i+1:]:
                allImages.append((first, second, True))

    return allImages







def pairPresent(path_list, firstImage, secondImage):
    for f,s,_ in path_list:
        if f==firstImage and s==secondImage:
            return True
    return False

def getDiffImages(allImages, imDict):
    firstImage = random.choice(allImages)
    seconfImage = random.choice(allImages)
    issame = imDict[firstImage] == imDict[seconfImage]
    while firstImage == seconfImage or issame:
        seconfImage = random.choice(allImages)
        issame = imDict[firstImage] == imDict[seconfImage]
    return sorted([firstImage, seconfImage])

class LFWDataset(datasets.ImageFolder):
    def __init__(self, dir, transform=None, numPairs =4500, addBinary =True):

        super(LFWDataset, self).__init__(dir, transform)
        self.numPairs = numPairs
        # LFW dir contains 2 folders: faces and lists
        self.validation_images = self.get_lfw_paths(dir)
        self.addBinary = addBinary
        self.rec = ImageRactangle()


    def read_lfw_pairs(self, imagesFolder):
        from collections import defaultdict
        imDict = {}
        folderToImage = defaultdict(list)
        dirs = os.listdir(imagesFolder)
        for d in dirs:
            dirPath = os.path.join(imagesFolder,d)
            for im in os.listdir( dirPath):
                if im.endswith(".npy"):
                    continue
                image = os.path.join(dirPath,im)
                if  getBinarisedImage(image) is None:
                    continue
                imDict[image] = d
                folderToImage[d].append(image)
        return imDict, folderToImage

    def get_lfw_paths(self, imageFolder):
        """

        :param imageFolder:
        :return:
        """
        sameImagesNum = self.numPairs//3
        diffImagesNum = self.numPairs - sameImagesNum
        print("test possitiveSamples: {}, testNegativeSamples: {}".format(sameImagesNum, diffImagesNum))

        imDict, folderToImage = self.read_lfw_pairs(imageFolder)
        allImages = list(imDict.keys())
        path_list = []

        allSameImages = getSameImages(folderToImage)
        path_list += allSameImages
        progress_bar = tqdm(range(self.numPairs - len(allSameImages)))



        for _ in progress_bar:
            # if sameImagesNum >0:
            #     firstImage, secondImage = getSameImage(folderToImage)
            #
            #     while( pairPresent(path_list, firstImage, secondImage) ):
            #         firstImage, secondImage = getSameImage(folderToImage)
            #
            #     # manu = random.choice(list(folderToImage.keys()))
            #     # while len(folderToImage[manu])<2:
            #     #     manu = random.choice(list(folderToImage.keys()))
            #     # firstImage = random.choice(folderToImage[manu])
            #     # seconfImage = random.choice(folderToImage[manu])
            #     # while firstImage == seconfImage:
            #     #     seconfImage = random.choice(folderToImage[manu])
            #     sameImagesNum -=1
            # else:
                firstImage, secondImage = getDiffImages(allImages, imDict)
                while pairPresent(path_list, firstImage, secondImage):
                    firstImage, secondImage = getDiffImages(allImages, imDict)

                issame = imDict[firstImage] == imDict[secondImage]
                path_list.append((firstImage, secondImage, issame))


        random.shuffle(path_list)
        return path_list

    def __getitem__(self, index):
        """
        Args:
            index: Index of the triplet or the matches - not of a single image
        Returns:
        """
        from PIL import Image


        def transform(img):
            """Convert image into numpy array and apply transformation
               Doing this so that it is consistent with all other datasets
               to return a PIL Image.
            """

            #img = self.loader(img_path)
            return self.transform(img)

        (path_1, path_2, issame) = self.validation_images[index]


        img_bin_path1 = getBinarisedImage(path_1)
        img_bin_path2 = getBinarisedImage(path_2)


        img_bin1 = Image.open(img_bin_path1)
        img_bin2 = Image.open(img_bin_path2)





        # Modified to open as PIL image in the first place
        img1 = Image.open(path_1).convert("RGB")
        img2 = Image.open(path_2).convert("RGB")
        # if self.addBinary:
        #     img_bin1 = np.array(img_bin1.resize(img1.size))
        #     img1 = np.append(np.array(img1), img_bin1.reshape((img_bin1.shape[0], img_bin1.shape[1], 1)), axis=2)
        #     img1 = Image.fromarray(img1)
        #
        #     img_bin2 = np.array(img_bin2.resize(img2.size))
        #     img2 = np.append(np.array(img2), img_bin2.reshape((img_bin2.shape[0], img_bin2.shape[1], 1)), axis=2)
        #     img2 = Image.fromarray(img2)

        img1 = getRectangle(path_1, img1, self.addBinary, img_bin=img_bin1, rec= self.rec)
        img2 = getRectangle(path_2, img2, self.addBinary, img_bin=img_bin2, rec= self.rec)


        img1, img2 = transform(img1), transform(img2)
        return img1, img2, issame

    def __len__(self):
        return len(self.validation_images)
