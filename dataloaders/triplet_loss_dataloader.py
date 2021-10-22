"""This code was imported from tbmoon's 'facenet' repository:
    https://github.com/tbmoon/facenet/blob/master/data_loader.py

    The code was modified to support .png and .jpg files.
"""


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

# class TripletFaceDataset(Dataset):
#     # Modified to add 'training_triplets_path' parameter
#     def __init__(self, root_dir, csv_name, num_triplets, training_triplets_path=None, transform=None):
#
#         # Modified here to set the data types of the dataframe columns to be suitable for other datasets other than the
#         #   VggFace2 dataset (Casia-WebFace in this case because of the identities starting with numbers automatically
#         #   forcing the 'name' column as being of type 'int' instead of type 'object')
#         self.df = pd.read_csv(csv_name, dtype={'id': object, 'name': object, 'class': int})
#         self.root_dir = root_dir
#         self.num_triplets = num_triplets
#         self.transform = transform
#
#         # Modified here
#         if training_triplets_path is None:
#             self.training_triplets = self.generate_triplets(self.df, self.num_triplets)
#         else:
#             self.training_triplets = np.load(training_triplets_path)
#
#     @staticmethod
#     def generate_triplets(df, num_triplets):
#
#         def make_dictionary_for_face_class(df):
#             """
#               - face_classes = {'class0': [class0_id0, ...], 'class1': [class1_id0, ...], ...}
#             """
#             face_classes = dict()
#             for idx, label in enumerate(df['class']):
#                 if label not in face_classes:
#                     face_classes[label] = []
#                 face_classes[label].append(df.iloc[idx, 0])
#
#             return face_classes
#
#         triplets = []
#         classes = df['class'].unique()
#         face_classes = make_dictionary_for_face_class(df)
#
#         # Modified here to add a print statement
#         print("\nGenerating {} triplets...".format(num_triplets))
#
#         progress_bar = tqdm(range(num_triplets))
#         for _ in progress_bar:
#
#             """
#               - randomly choose anchor, positive and negative images for triplet loss
#               - anchor and positive images in pos_class
#               - negative image in neg_class
#               - at least, two images needed for anchor and positive images in pos_class
#               - negative image should have different class as anchor and positive images by definition
#             """
#
#             pos_class = np.random.choice(classes)
#             neg_class = np.random.choice(classes)
#
#             while len(face_classes[pos_class]) < 2:
#                 pos_class = np.random.choice(classes)
#
#             while pos_class == neg_class:
#                 neg_class = np.random.choice(classes)
#
#             pos_name = df.loc[df['class'] == pos_class, 'name'].values[0]
#             neg_name = df.loc[df['class'] == neg_class, 'name'].values[0]
#
#             if len(face_classes[pos_class]) == 2:
#                 ianc, ipos = np.random.choice(2, size=2, replace=False)
#
#             else:
#                 ianc = np.random.randint(0, len(face_classes[pos_class]))
#                 ipos = np.random.randint(0, len(face_classes[pos_class]))
#
#                 while ianc == ipos:
#                     ipos = np.random.randint(0, len(face_classes[pos_class]))
#
#             ineg = np.random.randint(0, len(face_classes[neg_class]))
#
#             triplets.append(
#                 [
#                     face_classes[pos_class][ianc],
#                     face_classes[pos_class][ipos],
#                     face_classes[neg_class][ineg],
#                     pos_class,
#                     neg_class,
#                     pos_name,
#                     neg_name
#                 ]
#             )
#
#         # Modified here to save the training triplets as a numpy file to not have to redo this process every
#         #   training execution from scratch
#         print("Saving training triplets list in datasets/ directory ...")
#         np.save('datasets/training_triplets_{}.npy'.format(num_triplets), triplets)
#         print("Training triplets' list Saved!\n")
#
#         return triplets
#
#     def __getitem__(self, idx):
#
#         anc_id, pos_id, neg_id, pos_class, neg_class, pos_name, neg_name = self.training_triplets[idx]
#         usePatch = True
#         if anc_id in self.cache:
#             if usePatch:
#                 anc_img, anc_side = random.choice(self.cache[anc_id][2])
#             else:
#                 anc_img, anc_side = self.cache[anc_id]
#         else:
#             anc_imgName = os.path.join(self.root_dir, str(anc_id))
#             anc_img = Image.open(anc_imgName)
#             anc_img, anc_side, allRec = getRectangle(anc_imgName, anc_img, addBinary=False, img_bin=None,
#                                                      rec=self.rec, orentation=self.orentation, isBinary=True)
#             self.cache[anc_id] = [anc_img, anc_side, self.turnToImages(allRec, anc_id)]
#
#         if pos_id in self.cache:
#             if usePatch:
#                 pos_img, pos_side = random.choice(self.cache[pos_id][2])
#             else:
#                 pos_img, pos_side = self.cache[pos_id]
#         else:
#             pos_imgName = os.path.join(self.root_dir, str(pos_id))
#             pos_img = Image.open(pos_imgName)
#             pos_img, pos_side, allRec = getRectangle(pos_imgName, pos_img, addBinary=False, img_bin=None,
#                                                      rec=self.rec, orentation=self.orentation, isBinary=True)
#             self.cache[pos_id] = [pos_img, pos_side, self.turnToImages(allRec, pos_id)]
#
#
#
#         # anc_img = self.add_extension(os.path.join(self.root_dir, str(pos_name), str(anc_id)))
#         # pos_img = self.add_extension(os.path.join(self.root_dir, str(pos_name), str(pos_id)))
#         # neg_img = self.add_extension(os.path.join(self.root_dir, str(neg_name), str(neg_id)))
#         #
#         # # Modified to open as PIL image in the first place
#         # anc_img = Image.open(anc_img)
#         # pos_img = Image.open(pos_img)
#         # neg_img = Image.open(neg_img)
#
#         pos_class = torch.from_numpy(np.array([pos_class]).astype('long'))
#         neg_class = torch.from_numpy(np.array([neg_class]).astype('long'))
#
#         sample = {
#             'anc_img': anc_img,
#             'pos_img': pos_img,
#             'neg_img': neg_img,
#             'pos_class': pos_class,
#             'neg_class': neg_class
#         }
#
#         if self.transform:
#             sample['anc_img'] = self.transform(sample['anc_img'])
#             sample['pos_img'] = self.transform(sample['pos_img'])
#             sample['neg_img'] = self.transform(sample['neg_img'])
#
#         return sample
#
#     def __len__(self):
#         return len(self.training_triplets)
#
#     # Added this method to allow .jpg and .png image support
#     def add_extension(self, path):
#         if os.path.exists(path + '.jpg'):
#             return path + '.jpg'
#         elif os.path.exists(path + '.png'):
#             return path + '.png'
#         else:
#             raise RuntimeError('No file "%s" with extension png or jpg.' % path)
#
#
#


def getLettersClasses():
    pass


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



#
#
# class TripletLettersDataset(Dataset):
#     # Modified to add 'training_triplets_path' parameter
#     def __init__(self, root_dir, num_triplets, training_triplets_path="./training_triplets_1100000.npy",
#                  transform=None, manuToPeriodMapPath=r"/home/olya/Documents/scrollPeriods/scroll2Period.csv"):
#
#         # Modified here to set the data types of the dataframe columns to be suitable for other datasets other than the
#         #   VggFace2 dataset (Casia-WebFace in this case because of the identities starting with numbers automatically
#         #   forcing the 'name' column as being of type 'int' instead of type 'object')
#         self.root_dir = root_dir
#         self.num_triplets = num_triplets
#         self.transform = transform
#         manuToPeriodMap = pd.read_csv(manuToPeriodMapPath, index_col = "manuscript").dropna().T.to_dict()
#         self.periodToManuMap = defaultdict(list)
#         for manu, val in  manuToPeriodMap.items():
#             period = val["period"]
#             if period != "roman":
#                 self.periodToManuMap[period].append(manu)
#
#         # Modified here
#         if training_triplets_path is None:
#             self.training_triplets = self.generate_triplets(self.num_triplets, self.root_dir, self.periodToManuMap)
#         else:
#             self.training_triplets = np.load(training_triplets_path)
#
#         self.periodToClass = { k:i for i, k in enumerate(list(self.periodToManuMap.keys())) }
#
#     @staticmethod
#     def generate_triplets(num_triplets, root_dir, periodToManuMap):
#
#         def make_dictionary_for_letter_class(root_dir):
#             """
#               - face_classes = {'class0': [class0_id0, ...], 'class1': [class1_id0, ...], ...}
#             """
#             letters_classes = dict()
#             periodToLetter = defaultdict(list)
#             for p in os.listdir(root_dir):
#                 currDir = os.path.join(root_dir, p)
#                 if not os.path.isdir(currDir):
#                     continue
#
#                 allLetters = os.listdir(currDir)
#                 letters_classes[p] = [ os.path.join(p, l) for l in allLetters]
#             for period, manus in periodToManuMap.items():
#                 for manu in manus:
#                     periodToLetter[period] += letters_classes[manu] if manu in letters_classes else []
#             return periodToLetter
#
#         triplets = []
#
#         letters_classes = make_dictionary_for_letter_class(root_dir)
#         classes = list(letters_classes.keys())
#         # Modified here to add a print statement
#         print("\nGenerating {} triplets...".format(num_triplets))
#
#         progress_bar = tqdm(range(num_triplets))
#         for _ in progress_bar:
#
#             """
#               - randomly choose anchor, positive and negative images for triplet loss
#               - anchor and positive images in pos_class
#               - negative image in neg_class
#               - at least, two images needed for anchor and positive images in pos_class
#               - negative image should have different class as anchor and positive images by definition
#             """
#
#             pos_class = random.choice(classes)
#             neg_class = random.choice(classes)
#
#             while len(letters_classes[pos_class]) < 2:
#                 pos_class = np.random.choice(classes)
#
#             while pos_class == neg_class or len(letters_classes[neg_class]) < 1:
#                 neg_class = np.random.choice(classes)
#
#             pos_name = pos_class#df.loc[df['class'] == pos_class, 'name'].values[0]
#             neg_name = neg_class#df.loc[df['class'] == neg_class, 'name'].values[0]
#
#             if len(letters_classes[pos_class]) == 2:
#                 ianc, ipos = np.random.choice(2, size=2, replace=False)
#
#             else:
#                 ianc = np.random.randint(0, len(letters_classes[pos_class]))
#                 ipos = np.random.randint(0, len(letters_classes[pos_class]))
#
#                 while ianc == ipos:
#                     ipos = np.random.randint(0, len(letters_classes[pos_class]))
#
#             ineg = np.random.randint(0, len(letters_classes[neg_class]))
#
#             triplets.append(
#                 [
#                     letters_classes[pos_class][ianc],
#                     letters_classes[pos_class][ipos],
#                     letters_classes[neg_class][ineg],
#                     pos_class,
#                     neg_class,
#                     pos_name,
#                     neg_name
#                 ]
#             )
#
#         # Modified here to save the training triplets as a numpy file to not have to redo this process every
#         #   training execution from scratch
#         print("Saving training triplets list in datasets/ directory ...")
#         np.save('datasets/training_triplets_{}.npy'.format(num_triplets), triplets)
#         print("Training triplets' list Saved!\n")
#
#         return triplets
#
#     def __getitem__(self, idx):
#
#         anc_id, pos_id, neg_id, pos_class, neg_class, pos_name, neg_name = self.training_triplets[idx]
#
#         usePatch = True
#
#         if anc_id in self.cache:
#             if usePatch:
#                 anc_img, anc_side = random.choice(self.cache[anc_id][2])
#             else:
#                 anc_img, anc_side = self.cache[anc_id]
#         else:
#             anc_imgName = os.path.join(self.root_dir, str(anc_id))
#             anc_img_bin = Image.open(getBinarisedImage(anc_imgName))  if self.addBinary else None
#             anc_img = Image.open(anc_imgName)
#             anc_img , anc_side, allRec= getRectangle(anc_imgName, anc_img, self.addBinary, rec=self.rec, orentation=self.orentation, isBinary = True)
#             self.cache[anc_id] = [anc_img, anc_side, self.turnToImages(allRec, anc_id) ]
#
#
#         anc_img = os.path.join(self.root_dir, str(anc_id))
#
#         pos_img = os.path.join(self.root_dir,  str(pos_id))
#         neg_img = os.path.join(self.root_dir,  str(neg_id))
#
#         # Modified to open as PIL image in the first place
#         anc_img = Image.open(anc_img)
#         pos_img = Image.open(pos_img)
#         neg_img = Image.open(neg_img)
#
#         pos_class = torch.from_numpy(np.array([self.periodToClass[ pos_class]]))
#         neg_class = torch.from_numpy(np.array([self.periodToClass[neg_class]]))
#
#         sample = {
#             'anc_img': anc_img,
#             'pos_img': pos_img,
#             'neg_img': neg_img,
#             'pos_class': pos_class,
#             'neg_class': neg_class
#         }
#
#         if self.transform:
#             sample['anc_img'] = self.transform(sample['anc_img'])
#             sample['pos_img'] = self.transform(sample['pos_img'])
#             sample['neg_img'] = self.transform(sample['neg_img'])
#
#         return sample
#
#     def __len__(self):
#         return len(self.training_triplets)



FOLDERS2IGNORE = ["XHev", "5"]


def isValidImage(pth, root, addBinary):

    if pth.endswith("npy"):
        return False
    fullP = os.path.join(root, pth)
    if not os.path.isfile(fullP):
        print("{} is not a file".format(fullP))
        return False

    if addBinary:
        try:
            Image.open(fullP).convert("RGB")
            bin = getBinarisedImage(fullP)
            if bin is None:
                return False
        except:
            print("cannot open image {}".format(fullP))
            return False
    return True



def validFile(binName):
    if binName.endswith(".npy"):
        return False

    npyName = binName[:-4]+".npy"


    d = np.load(npyName, allow_pickle='TRUE').item()
    return len(d['rand_sample'])>0



class TripletFragmentsDatasetBinary(Dataset):
    # Modified to add 'training_triplets_path' parameter
    def __init__(self, root_dir, num_triplets, training_triplets_path="./training_fragment_triplets__500000_bin.npy",
                 transform=None):

        # Modified here to set the data types of the dataframe columns to be suitable for other datasets other than the
        #   VggFace2 dataset (Casia-WebFace in this case because of the identities starting with numbers automatically
        #   forcing the 'name' column as being of type 'int' instead of type 'object')
        self.root_dir = root_dir
        self.num_triplets = num_triplets
        self.transform = transform
        allManus = os.listdir(root_dir)
        #allManus=[i for i in allManus if i not in FOLDERS2IGNORE]
        self.manuToClass = {k: i for i, k in enumerate(allManus)}
        # Modified here
        if training_triplets_path is None:
            self.training_triplets = self.generate_triplets(self.num_triplets, self.root_dir)
        else:
            self.training_triplets = np.load(training_triplets_path)
        self.rec = ImageRactangle(False)
        self.cache = {}

    def getImage(self, id, center, side):
        imgName = os.path.join(self.root_dir, str(id))
        img = Image.open(imgName)
        img = img.resize((img.size[0], img.size[1]))
        w, h = center
        img = img.crop((h - side, w - side, h + side, w + side))
        return img

    def turnToImages(self, allRec, id):
        allImages = []
        for center, side in allRec:
            allImages.append((self.getImage(id, center, side), side))
        return allImages


    @staticmethod
    def generate_triplets(num_triplets, root_dir):

        def make_dictionary_for_letter_class(root_dir):
            """
              - face_classes = {'class0': [class0_id0, ...], 'class1': [class1_id0, ...], ...}
            """
            fragment_classes = dict()
            for p in os.listdir(root_dir):
                if not p.endswith("_bin3"):
                    continue

                currDir = os.path.join(root_dir, p)
                if not os.path.isdir(currDir):
                    continue
                print(p)


                fullPaths = os.listdir(currDir)
                #fullPaths = [ os.path.join(currDir,pa) for pa in fullPaths if validFile(os.path.join( currDir, pa ))]

                if fullPaths:
                    fragment_classes[p.replace("_bin3", "")] = fullPaths


            return fragment_classes

        triplets = []

        fragment_classes = make_dictionary_for_letter_class(root_dir)
        classes = list(fragment_classes.keys())
        # Modified here to add a print statement
        print("\nGenerating {} triplets...".format(num_triplets))

        progress_bar = tqdm(range(num_triplets))
        for _ in progress_bar:

            """
              - randomly choose anchor, positive and negative images for triplet loss
              - anchor and positive images in pos_class
              - negative image in neg_class
              - at least, two images needed for anchor and positive images in pos_class
              - negative image should have different class as anchor and positive images by definition
            """

            pos_class = random.choice(classes)
            neg_class = random.choice(classes)

            while len(fragment_classes[pos_class]) < 2:
                pos_class = np.random.choice(classes)

            while pos_class == neg_class or len(fragment_classes[neg_class]) < 1:
                neg_class = np.random.choice(classes)

            pos_name = pos_class#df.loc[df['class'] == pos_class, 'name'].values[0]
            neg_name = neg_class#df.loc[df['class'] == neg_class, 'name'].values[0]

            if len(fragment_classes[pos_class]) == 2:
                ianc, ipos = np.random.choice(2, size=2, replace=False)

            else:
                ianc = np.random.randint(0, len(fragment_classes[pos_class]))
                ipos = np.random.randint(0, len(fragment_classes[pos_class]))

                while ianc == ipos:
                    ipos = np.random.randint(0, len(fragment_classes[pos_class]))

            ineg = np.random.randint(0, len(fragment_classes[neg_class]))

            triplets.append(
                [
                    fragment_classes[pos_class][ianc],
                    fragment_classes[pos_class][ipos],
                    fragment_classes[neg_class][ineg],
                    pos_class,
                    neg_class,
                    pos_name,
                    neg_name
                ]
            )

        # Modified here to save the training triplets as a numpy file to not have to redo this process every
        #   training execution from scratch
        print("Saving training triplets list in datasets/ directory ...")
        np.save('datasets/training_fragment_triplets_{}_bin.npy'.format(num_triplets), triplets)
        print("Training triplets' list Saved!\n")

        return triplets

    def __getitem__(self, idx):

        anc_id, pos_id, neg_id, pos_class, neg_class, pos_name, neg_name = self.training_triplets[idx]

        anc_imgName = os.path.join(self.root_dir, pos_class+"_bin3", str(anc_id))
        anc = Image.open(anc_imgName)

        neg_imgName = os.path.join(self.root_dir, neg_class+"_bin3", str(neg_id))
        neg = Image.open(neg_imgName)

        pos_imgName = os.path.join(self.root_dir,pos_class+"_bin3", str(pos_id))
        pos = Image.open(pos_imgName)

        usePatch = True
        # if anc_id in self.cache:
        #     if usePatch:
        #         anc, anc_side = random.choice(self.cache[anc_id][2])
        #     else:
        #         anc, anc_side = self.cache[anc_id]
        # else:
        #     anc_imgName = os.path.join(self.root_dir, str(anc_id))
        #     anc_img = Image.open(anc_imgName)
        #     anc, anc_side, allRec = getRectangle(anc_imgName, anc_img, addBinary=False, img_bin=None,
        #                                              rec=self.rec, orentation=None, isBinary=True)
        #     self.cache[anc_id] = [anc, anc_side, self.turnToImages(allRec, anc_id)]
        #
        # if pos_id in self.cache:
        #     if usePatch:
        #         pos, pos_side = random.choice(self.cache[pos_id][2])
        #     else:
        #         pos, pos_side = self.cache[pos_id]
        # else:
        #     pos_imgName = os.path.join(self.root_dir, str(pos_id))
        #     pos_img = Image.open(pos_imgName)
        #     pos, pos_side, allRec = getRectangle(pos_imgName, pos_img, addBinary=False, img_bin=None,
        #                                              rec=self.rec, orentation=None, isBinary=True)
        #     self.cache[pos_id] = [pos, pos_side, self.turnToImages(allRec, pos_id)]
        #
        #
        # if neg_id in self.cache:
        #     if usePatch:
        #         neg, neg_side = random.choice(self.cache[neg_id][2])
        #     else:
        #         neg, neg_side = self.cache[neg_id]
        # else:
        #     neg_imgName = os.path.join(self.root_dir, str(neg_id))
        #     neg_img = Image.open(neg_imgName)
        #     neg, neg_side, allRec = getRectangle(neg_imgName, neg_img, addBinary=False, img_bin=None,
        #                                              rec=self.rec, orentation=None, isBinary=True)
        #     self.cache[neg_id] = [neg, neg_side, self.turnToImages(allRec, neg_id)]




        pos_class = torch.from_numpy(np.array([self.manuToClass[ pos_class]]))
        neg_class = torch.from_numpy(np.array([self.manuToClass[neg_class]]))

        sample = {
            'anc_img': anc,
            'pos_img': pos,
            'neg_img': neg,
            'pos_class': pos_class,
            'neg_class': neg_class
        }

        if self.transform:
            sample['anc_img'] = self.transform(sample['anc_img'])
            sample['pos_img'] = self.transform(sample['pos_img'])
            sample['neg_img'] = self.transform(sample['neg_img'])

        return sample

    def __len__(self):
        return len(self.training_triplets)



class TripletFragmentsDatasetBinaryLetters(Dataset):
    # Modified to add 'training_triplets_path' parameter
    def __init__(self, root_dir, num_triplets, training_triplets_path="./training_fragment_triplets__letters_500000_bin.npy",
                 transform=None):

        # Modified here to set the data types of the dataframe columns to be suitable for other datasets other than the
        #   VggFace2 dataset (Casia-WebFace in this case because of the identities starting with numbers automatically
        #   forcing the 'name' column as being of type 'int' instead of type 'object')
        self.root_dir = root_dir
        self.num_triplets = num_triplets
        self.transform = transform
        allManus = os.listdir(root_dir)
        #allManus=[i for i in allManus if i not in FOLDERS2IGNORE]
        self.manuToClass = {k: i for i, k in enumerate(allManus)}
        # Modified here
        if training_triplets_path is None:
            self.training_triplets = self.generate_triplets(self.num_triplets, self.root_dir)
        else:
            self.training_triplets = np.load(training_triplets_path)
        self.cache = {}

    def getImage(self, id):
        imgName = os.path.join(self.root_dir, str(id))
        img = Image.open(imgName)
        return img



    @staticmethod
    def generate_triplets(num_triplets, root_dir):

        def make_dictionary_for_letter_class(root_dir):
            """
              - face_classes = {'class0': [class0_id0, ...], 'class1': [class1_id0, ...], ...}
            """
            fragment_classes = dict()
            for p in os.listdir(root_dir):
                if not p.endswith("_bin"):
                    continue

                currDir = os.path.join(root_dir, p)
                if not os.path.isdir(currDir):
                    continue
                print(p)


                fullPaths = os.listdir(currDir)
                #fullPaths = [ os.path.join(currDir,pa) for pa in fullPaths if validFile(os.path.join( currDir, pa ))]

                if fullPaths:
                    fragment_classes[p.replace("_bin", "")] = fullPaths


            return fragment_classes

        triplets = []

        fragment_classes = make_dictionary_for_letter_class(root_dir)
        classes = list(fragment_classes.keys())
        # Modified here to add a print statement
        print("\nGenerating {} triplets...".format(num_triplets))

        progress_bar = tqdm(range(num_triplets))
        for _ in progress_bar:

            """
              - randomly choose anchor, positive and negative images for triplet loss
              - anchor and positive images in pos_class
              - negative image in neg_class
              - at least, two images needed for anchor and positive images in pos_class
              - negative image should have different class as anchor and positive images by definition
            """

            pos_class = random.choice(classes)
            neg_class = random.choice(classes)

            while len(fragment_classes[pos_class]) < 2:
                pos_class = np.random.choice(classes)

            while pos_class == neg_class or len(fragment_classes[neg_class]) < 1:
                neg_class = np.random.choice(classes)

            pos_name = pos_class#df.loc[df['class'] == pos_class, 'name'].values[0]
            neg_name = neg_class#df.loc[df['class'] == neg_class, 'name'].values[0]

            if len(fragment_classes[pos_class]) == 2:
                ianc, ipos = np.random.choice(2, size=2, replace=False)

            else:
                ianc = np.random.randint(0, len(fragment_classes[pos_class]))
                ipos = np.random.randint(0, len(fragment_classes[pos_class]))

                while ianc == ipos:
                    ipos = np.random.randint(0, len(fragment_classes[pos_class]))

            ineg = np.random.randint(0, len(fragment_classes[neg_class]))

            triplets.append(
                [
                    fragment_classes[pos_class][ianc],
                    fragment_classes[pos_class][ipos],
                    fragment_classes[neg_class][ineg],
                    pos_class,
                    neg_class,
                    pos_name,
                    neg_name
                ]
            )

        # Modified here to save the training triplets as a numpy file to not have to redo this process every
        #   training execution from scratch
        print("Saving training triplets list in datasets/ directory ...")
        np.save('datasets/training_fragment_triplets_letters{}_bin.npy'.format(num_triplets), triplets)
        print("Training triplets' list Saved!\n")

        return triplets

    def __getitem__(self, idx):

        anc_id, pos_id, neg_id, pos_class, neg_class, pos_name, neg_name = self.training_triplets[idx]

        anc_imgName = os.path.join(self.root_dir, pos_class+"_bin", str(anc_id))
        anc = Image.open(anc_imgName)

        neg_imgName = os.path.join(self.root_dir, neg_class+"_bin", str(neg_id))
        neg = Image.open(neg_imgName)

        pos_imgName = os.path.join(self.root_dir,pos_class+"_bin", str(pos_id))
        pos = Image.open(pos_imgName)



        pos_class = torch.from_numpy(np.array([self.manuToClass[ pos_class]]))
        neg_class = torch.from_numpy(np.array([self.manuToClass[neg_class]]))

        sample = {
            'anc_img': anc,
            'pos_img': pos,
            'neg_img': neg,
            'pos_class': pos_class,
            'neg_class': neg_class
        }

        if self.transform:
            sample['anc_img'] = self.transform(sample['anc_img'])
            sample['pos_img'] = self.transform(sample['pos_img'])
            sample['neg_img'] = self.transform(sample['neg_img'])

        return sample

    def __len__(self):
        return len(self.training_triplets)


def getRectangle(imgName, img,  addBinary, rec , orentation, img_bin = None, isBinary = False):
    import random
    center, side, allRec = rec.getLagestRectangle(imgName)
    # imgName = imgName.split("/")[-1]
    # if imgName in orentation:
    #     orent = orentation[imgName]
    #     if orent == 'c':
    #         #print("right")
    #         img= img.rotate(angle=240)
    #         img_bin=img_bin.rotate(angle=240) if img_bin is not None else img_bin
    #     elif orent == "v":
    #         img = img.rotate(angle=90)
    #         img_bin = img_bin.rotate(angle=90) if img_bin is not None else img_bin
    #         #print("left")
    #     elif orent == "x":
    #         #print("upside")
    #         img = img.rotate(angle=180)
    #         img_bin = img_bin.rotate(angle=180) if img_bin is not None else img_bin
    #     else:
    #         pass
    #         #print("orent {} regular".format(orent))

    center, side = random.choice(allRec)
    w,h = center

    if img_bin:
        img_bin = img_bin.resize(img.size)
    if isBinary:
        img = img.resize( (img.size[0]//2, img.size[1]//2))
    im = img.crop((h-side, w-side, h+side, w+side))
    #im.show()
    return im, side, allRec

def getRectangle_(imgName, img,  addBinary, rec , orentation, img_bin = None):

    center, side = rec.getLagestRectangle(imgName)
    imgName = imgName.split("/")[-1]
    if imgName in orentation:
        orent = orentation[imgName]
        if orent == 'c':
            #print("right")
            img= img.rotate(angle=240)
            img_bin=img_bin.rotate(angle=240) if img_bin is not None else img_bin
        elif orent == "v":
            img = img.rotate(angle=90)
            img_bin = img_bin.rotate(angle=90) if img_bin is not None else img_bin
            #print("left")
        elif orent == "x":
            #print("upside")
            img = img.rotate(angle=180)
            img_bin = img_bin.rotate(angle=180) if img_bin is not None else img_bin
        else:
            pass
            #print("orent {} regular".format(orent))


    w,h = center

    if img_bin:
        img_bin = img_bin.resize(img.size)
    im = img.crop((h-side, w-side, h+side, w+side))
    #im.show()


    if addBinary:
        img_bin = img_bin.crop((h-side, w-side, h+side, w+side))
        img_bin = np.array(img_bin)
        img = np.append(np.array(im), img_bin.reshape((img_bin.shape[0], img_bin.shape[1], 1)), axis=2)
        im = Image.fromarray(img)

    return im, side


class TripletFragmentsDataset(Dataset):
    # Modified to add 'training_triplets_path' parameter
    def __init__(self, root_dir, num_triplets, training_triplets_path="./training_fragment_triplets__1100000.npy",
                 transform=None, addBinary = True):

        # Modified here to set the data types of the dataframe columns to be suitable for other datasets other than the
        #   VggFace2 dataset (Casia-WebFace in this case because of the identities starting with numbers automatically
        #   forcing the 'name' column as being of type 'int' instead of type 'object')
        self.root_dir = root_dir
        self.num_triplets = num_triplets
        self.transform = transform
        self.addBinary = addBinary
        allManus = os.listdir(root_dir)
        allManus=[i for i in allManus if i not in FOLDERS2IGNORE]
        self.manuToClass = {k: i for i, k in enumerate(allManus)}
        # Modified here
        if training_triplets_path is None:
            self.training_triplets = self.generate_triplets(self.num_triplets, self.root_dir, self.addBinary)
        else:
            self.training_triplets = np.load(training_triplets_path)

        self.cache={}

        isTest = "test" in root_dir.lower()
        orentDFPath = "/home/olya/Documents/facenetbased-dss/misc/fragmentOrentTest.csv" if isTest else "/home/olya/Documents/facenetbased-dss/misc/fragmentOrentTrain.csv"
        orent = {}
        orent1 = pd.read_csv(orentDFPath).transpose().to_dict()
        for k, v in orent1.items():
            orent[v["Unnamed: 0"]] = v["orentation"]
        self.orentation = orent
        self.rec = ImageRactangle(isTest)



    @staticmethod
    def generate_triplets(num_triplets, root_dir, addBinary):

        def make_dictionary_for_letter_class(root_dir):
            """
              - face_classes = {'class0': [class0_id0, ...], 'class1': [class1_id0, ...], ...}
            """
            fragment_classes = dict()
            for p in os.listdir(root_dir):
                if p in FOLDERS2IGNORE:
                    continue
                currDir = os.path.join(root_dir, p)
                if not os.path.isdir(currDir):
                    continue
                print(p)
                allFragments = os.listdir(currDir)
                fullPaths = [ os.path.join(p, l) for l in allFragments]
                fullPaths =[ pth for pth in fullPaths if isValidImage(pth,root_dir, addBinary)]
                if fullPaths:
                    fragment_classes[p] = fullPaths


            return fragment_classes

        triplets = []

        fragment_classes = make_dictionary_for_letter_class(root_dir)
        classes = list(fragment_classes.keys())
        # Modified here to add a print statement
        print("\nGenerating {} triplets...".format(num_triplets))

        progress_bar = tqdm(range(num_triplets))
        for _ in progress_bar:

            """
              - randomly choose anchor, positive and negative images for triplet loss
              - anchor and positive images in pos_class
              - negative image in neg_class
              - at least, two images needed for anchor and positive images in pos_class
              - negative image should have different class as anchor and positive images by definition
            """

            pos_class = random.choice(classes)
            neg_class = random.choice(classes)

            while len(fragment_classes[pos_class]) < 2:
                pos_class = np.random.choice(classes)

            while pos_class == neg_class or len(fragment_classes[neg_class]) < 1:
                neg_class = np.random.choice(classes)

            pos_name = pos_class#df.loc[df['class'] == pos_class, 'name'].values[0]
            neg_name = neg_class#df.loc[df['class'] == neg_class, 'name'].values[0]

            if len(fragment_classes[pos_class]) == 2:
                ianc, ipos = np.random.choice(2, size=2, replace=False)

            else:
                ianc = np.random.randint(0, len(fragment_classes[pos_class]))
                ipos = np.random.randint(0, len(fragment_classes[pos_class]))

                while ianc == ipos:
                    ipos = np.random.randint(0, len(fragment_classes[pos_class]))

            ineg = np.random.randint(0, len(fragment_classes[neg_class]))

            triplets.append(
                [
                    fragment_classes[pos_class][ianc],
                    fragment_classes[pos_class][ipos],
                    fragment_classes[neg_class][ineg],
                    pos_class,
                    neg_class,
                    pos_name,
                    neg_name
                ]
            )

        # Modified here to save the training triplets as a numpy file to not have to redo this process every
        #   training execution from scratch
        print("Saving training triplets list in datasets/ directory ...")
        np.save('datasets/training_fragment_triplets_{}.npy'.format(num_triplets), triplets)
        print("Training triplets' list Saved!\n")

        return triplets


    def getImage(self, id, center, side):
        imgName = os.path.join(self.root_dir, str(id))
        img = Image.open(imgName).convert("RGB")
        w, h = center
        img = img.crop((h - side, w - side, h + side, w + side))
        return img


    def turnToImages(self, allRec, id):
        allImages = []
        for center, side in allRec:
            allImages.append( ( self.getImage(id,center,side), side) )
        return allImages



    def __getitem__(self, idx):
        usePatch = True

        anc_id, pos_id, neg_id, pos_class, neg_class, pos_name, neg_name = self.training_triplets[idx]

        if anc_id in self.cache:
            if usePatch:
                anc_img, anc_side = random.choice(self.cache[anc_id][2])
            else:
                anc_img , anc_side= self.cache[anc_id]
            #anc_img = Image.fromarray(anc_img)
        else:
            anc_imgName = os.path.join(self.root_dir, str(anc_id))
            anc_img_bin = Image.open(getBinarisedImage(anc_imgName))  if self.addBinary else None
            anc_img = Image.open(anc_imgName).convert("RGB")
            anc_img , anc_side, allRec= getRectangle(anc_imgName, anc_img, self.addBinary, img_bin=anc_img_bin, rec=self.rec, orentation=self.orentation)
            self.cache[anc_id] = [anc_img, anc_side, self.turnToImages(allRec, anc_id) ]


        if pos_id in self.cache:
            if usePatch:
                pos_img, pos_side = random.choice(self.cache[pos_id][2])

            else:
                pos_img, pos_side = self.cache[pos_id]
            #pos_img = Image.fromarray(pos_img)
        else:
            pos_imgName = os.path.join(self.root_dir,  str(pos_id))
            pos_img_bin = Image.open(getBinarisedImage(pos_imgName)) if self.addBinary else None
            pos_img = Image.open(pos_imgName).convert("RGB")
            pos_img, pos_side, allRec = getRectangle(pos_imgName, pos_img, self.addBinary, img_bin=pos_img_bin, rec=self.rec, orentation=self.orentation)
            self.cache[pos_id] = [pos_img, pos_side, self.turnToImages(allRec, pos_id)]


        if neg_id in self.cache:
            if usePatch:
                neg_img, neg_side = random.choice(self.cache[neg_id][2])

            else:
                neg_img , neg_side= self.cache[neg_id]
            #neg_img = Image.fromarray(neg_img)
        else:
            neg_imgName = os.path.join(self.root_dir,  str(neg_id))
            neg_img_bin = Image.open(getBinarisedImage(neg_imgName)) if self.addBinary else None
            neg_img = Image.open(neg_imgName).convert("RGB")
            neg_img, neg_side, allRec = getRectangle(neg_imgName, neg_img, self.addBinary, img_bin=neg_img_bin, rec=self.rec, orentation=self.orentation)
            self.cache[neg_id] = [neg_img, neg_side, self.turnToImages(allRec, neg_id)]


        pos_class = torch.from_numpy(np.array([self.manuToClass[ pos_class]]))
        neg_class = torch.from_numpy(np.array([self.manuToClass[neg_class]]))

        sample = {
            'anc_img': [anc_img, anc_side],
            'pos_img': [pos_img, pos_side],
            'neg_img': [neg_img, neg_side],
            'pos_class': pos_class,
            'neg_class': neg_class
        }

        if self.transform:
            sample['anc_img'][0] = self.transform(sample['anc_img'][0])
            sample['pos_img'][0] = self.transform(sample['pos_img'][0])
            sample['neg_img'][0] = self.transform(sample['neg_img'][0])

        return sample

    def __len__(self):
        return len(self.training_triplets)

