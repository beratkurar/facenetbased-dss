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
from emnist import extract_training_samples


class TripletEmnistDataset(Dataset):

    # Modified to add 'training_triplets_path' parameter
    def __init__(self, num_triplets, data_transforms):
        images, labels = extract_training_samples('byclass')
        self.images = images
        self.labels = labels
        self.training_triplets = self.generate_triplets(num_triplets)
        self.transform = data_transforms



    def generate_triplets(self, num_triplets):

        def make_dictionary_for_letter_class(labels):
            """
              - face_classes = {'class0': [class0_id0, ...], 'class1': [class1_id0, ...], ...}
            """
            classes = dict()
            for l in np.unique( labels):
                classes[l] = np.where(labels == l)[0]
            return classes



        triplets = []

        fragment_classes = make_dictionary_for_letter_class(self.labels)
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




    def __getitem__(self, idx):

        anc_id, pos_id, neg_id, pos_class, neg_class, pos_name, neg_name = self.training_triplets[idx]

        #.convert("RGB")
        anc_img = Image.fromarray(self.images[anc_id])
        pos_img = Image.fromarray(self.images[pos_id])
        neg_img = Image.fromarray(self.images[neg_id])
        sample = {
            'anc_img': [anc_img, 1],
            'pos_img': [pos_img, 1],
            'neg_img': [neg_img, 1],
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



if __name__ == "__main__":
    m = TripletEmnistDataset()
