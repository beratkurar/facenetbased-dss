import numpy as np
import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from heapq import  nsmallest
import torchvision.transforms as transforms
from torch.nn.modules.distance import PairwiseDistance
from dataloaders.LFWDataset import LFWDataset
from losses.triplet_loss import TripletLoss
from dataloaders.triplet_loss_dataloader import  TripletFragmentsDataset, TripletFragmentsDatasetBinary
from dataloaders.regularLoader import   LettersDataset, FragmentsDataset, FragmentsDatasetBinary
from validate_on_LFW import evaluate_lfw
from plot import plot_roc_lfw, plot_accuracy_lfw, plot_triplet_losses
from tqdm import tqdm
from models.resnet import Resnet18Triplet
from models.resnet import Resnet34Triplet
from models.resnet import Resnet50Triplet
from models.resnet import Resnet101Triplet
from models.resnet import Resnet152Triplet
from models.inceptionresnetv2 import InceptionResnetV2Triplet
from visualizeEmbeddings import visualizeEmbeddings

from torch.utils.tensorboard import SummaryWriter
import datetime


name ='bin_patches'
writer = SummaryWriter('runs/traint_triplet/{}'.format(name))#datetime.datetime.now()))


parser = argparse.ArgumentParser(description="Training a FaceNet facial recognition model using Triplet Loss.")
# Dataset
parser.add_argument('--dataroot', '-d', type=str, required=False,
                    help="(REQUIRED) Absolute path to the dataset folder"
                    )
# LFW
parser.add_argument('--lfw', type=str, required=False,
                    help="(REQUIRED) Absolute path to the labeled faces in the wild dataset folder"
                    )
parser.add_argument('--dataset_csv', type=str, default='datasets/vggface2_full.csv',
                    help="Path to the csv file containing the image paths of the training dataset."
                    )
parser.add_argument('--lfw_batch_size', default=64, type=int,
                    help="Batch size for LFW dataset (default: 64)"
                    )
parser.add_argument('--lfw_validation_epoch_interval', default=1, type=int,
                    help="Perform LFW validation every n epoch interval (default: every 1 epoch)"
                    )
# Training settings
parser.add_argument('--model', type=str, default="resnet34", choices=["resnet18", "resnet34", "resnet50", "resnet101", "resnet152", "inceptionresnetv2"],
    help="The required model architecture for training: ('resnet18','resnet34', 'resnet50', 'resnet101', 'resnet152', 'inceptionresnetv2'), (default: 'resnet34')"
                    )
parser.add_argument('--epochs', default=3, type=int,
                    help="Required training epochs (default: 30)"
                    )
parser.add_argument('--training_triplets_path', default=None, type=str,
    help="Path to training triplets numpy file in 'datasets/' folder to skip training triplet generation step."
                    )
parser.add_argument('--num_triplets_train', default=500000, type=int,
                    help="Number of triplets for training (default: 1100000)"
                    )
parser.add_argument('--resume_path', default =  "/home/olya/Documents/facenetbased-dss/Model_training_checkpoints/model_resnet34color_patches_triplet_allepoch_1.pt",# 'Model_training_checkpoints/model_resnet34_triplet_allepoch_1.pt',
                    type=str,
    help='path to latest model checkpoint: (Model_training_checkpoints/model_resnet34_epoch_0.pt file) (default: None)'
                    )
parser.add_argument('--batch_size', default=16, type=int,
                    help="Batch size (default: 16)"
                    )
parser.add_argument('--num_workers', default=8, type=int,
                    help="Number of workers for data loaders (default: 8)"
                    )
parser.add_argument('--embedding_dim', default=128, type=int,
                    help="Dimension of the embedding vector (default: 128)"
                    )
parser.add_argument('--pretrained', default=False, type=bool,
                    help="Download a model pretrained on the ImageNet dataset (Default: False)"
                    )
parser.add_argument('--optimizer', type=str, default="sgd", choices=["sgd", "adagrad", "rmsprop", "adam"],
    help="Required optimizer for training the model: ('sgd','adagrad','rmsprop','adam'), (default: 'sgd')"
                    )
parser.add_argument('--lr', default=0.001, type=float,
                    help="Learning rate for the optimizer (default: 0.01)"
                    )
parser.add_argument('--margin', default=0.5, type=float,
                    help='margin for triplet loss (default: 0.5)'
                    )
args = parser.parse_args()



TRAIN_ONLY_HARD_NEGATIVE = False

def set_model_architecture(model_architecture, pretrained, embedding_dimension):
    if model_architecture == "resnet18":
        model = Resnet18Triplet(
            embedding_dimension=embedding_dimension,
            pretrained=pretrained
        )
    elif model_architecture == "resnet34":
        model = Resnet34Triplet(
            embedding_dimension=embedding_dimension,
            pretrained=pretrained
        )
    elif model_architecture == "resnet50":
        model = Resnet50Triplet(
            embedding_dimension=embedding_dimension,
            pretrained=pretrained
        )
    elif model_architecture == "resnet101":
        model = Resnet101Triplet(
            embedding_dimension=embedding_dimension,
            pretrained=pretrained
        )
    elif model_architecture == "resnet152":
        model = Resnet152Triplet(
            embedding_dimension=embedding_dimension,
            pretrained=pretrained
        )
    elif model_architecture == "inceptionresnetv2":
        model = InceptionResnetV2Triplet(
            embedding_dimension=embedding_dimension,
            pretrained=pretrained
        )
    print("Using {} model architecture.".format(model_architecture))

    return model


def set_model_gpu_mode(model):
    flag_train_gpu = torch.cuda.is_available()
    flag_train_multi_gpu = False

    if flag_train_gpu and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        model.cuda()
        flag_train_multi_gpu = True
        print('Using multi-gpu training.')

    elif flag_train_gpu and torch.cuda.device_count() == 1:
        model.cuda()
        print('Using single-gpu training.')

    return model, flag_train_multi_gpu




def howMinLettersEffectEmbedding(model, test_dataloader):
    from collections import defaultdict
    import matplotlib.pyplot as plt
    plt.style.use('seaborn-whitegrid')
    import cv2
    import numpy as np

    model.eval()
    class_labels = []
    features = []
    progress_bar = enumerate(tqdm(test_dataloader))
    manu2Vector = defaultdict(list)
    with torch.no_grad():
        for batch_index, (data, raw_image, label) in progress_bar:
            data = data.cuda()
            output = model(data)
            class_labels += label
            currFeatures = output.data.cpu().numpy().tolist()
            features += currFeatures
            for i, l in enumerate(label):
                manu = l.split(":")[0]
                location = l.split(":")[1]
                manu2Vector[manu].append( (currFeatures[i], location ))


    allDist = []
    allMinLetters = []
    allMaxLetters = []
    allAvgLetters = []
    for manu , allVecLoc in manu2Vector.items():
        for i, data in enumerate(allVecLoc):
            vec, loc = data
            for j, otherData in enumerate(allVecLoc[i+1:]) :
                othervec, otherloc = otherData
                dist = np.linalg.norm(np.array(vec)-np.array(othervec))
                minLettersLoc = getNumberOfLetters(loc)
                minLettersLocOther = getNumberOfLetters(otherloc)
                if minLettersLoc !=-1 and minLettersLocOther != -1:
                    minLetters =  min(minLettersLoc, minLettersLocOther)
                    maxLetters = max(minLettersLoc, minLettersLocOther)
                    avgLetters = (minLettersLoc+ minLettersLocOther)/2
                    allDist.append(dist)
                    allMinLetters.append(minLetters)
                    allMaxLetters.append(maxLetters)
                    allAvgLetters.append(avgLetters)

    plt.plot(np.array(allDist), np.array(allMinLetters), '.')
    plt.show()

    plt.plot(np.array(allDist), np.array(allMaxLetters), '.')
    plt.show()

    plt.plot(np.array(allDist), np.array(allAvgLetters), '.')
    plt.show()
    cv2.waitKey(0)



def copySmallestDist(class_labels, distPairs, pairToCopy = 30, toPath="./closestPairs"):
    import shutil
    import random

    if not os.path.exists(toPath):
        os.mkdir(toPath)
    for v  in distPairs[:pairToCopy]:
        first = class_labels[v[0]].split(":")[1]
        second = class_labels[v[1]].split(":")[1]
        dist = v[2]
        isSame = v[3]
        dir = "{}_{}".format(dist, isSame)
        dir = os.path.join(toPath, dir)
        os.mkdir(dir)
        shutil.copy(first,dir)
        shutil.copy(second, dir)



def calcAccuracy(model, test_dataloader, idxWriter=0):
    import sys
    model.eval()
    class_labels=[]
    features=[]
    progress_bar = enumerate(tqdm(test_dataloader))
    with torch.no_grad():
        for batch_index, (data, raw_image, label) in progress_bar:
            data = data.cuda()
            output = model(data, None)

            class_labels += label
            features += output.data.cpu().numpy().tolist()


    alldistances= [ [np.inf for _ in range(len(features))] for _ in range(len(features))]

    AllDistancePairs = []
    mindist = np.Inf

    for i in range(len(features)):
        for j in range(i, len(features)):
            v1 = features[i]
            v2 = features[j]
            dist = np.linalg.norm(np.array(v1)-np.array(v2))
            alldistances[i][j] = dist
            alldistances[j][i] = dist
            if i != j:
                AllDistancePairs.append( (i,j,dist))
                if dist  < 10**-4:
                    print("")
                if dist < mindist:
                    mindist = dist

    #mindist += 10^-3
    copySmallest = False
    if copySmallest:
        toCopy = [(i, j, d , class_labels[i].split(":")[0] == class_labels[j].split(":")[0]) for
                            i, j, d in AllDistancePairs]
        toCopy = sorted(toCopy, key=lambda x: x[2], reverse=False)
        copySmallestDist(class_labels, toCopy, pairToCopy=50, toPath="./closestPairsBin")
        toCopy = sorted(toCopy, key=lambda x: x[2], reverse=True)
        copySmallestDist(class_labels, toCopy, pairToCopy=50, toPath="./farthestPairsBin")





    AllDistancePairs = [ (i, j, 1/(d/mindist), class_labels[i].split(":")[0] == class_labels[j].split(":")[0]) for i,j,d in AllDistancePairs]
    AllDistancePairs = sorted(AllDistancePairs, key  = lambda x: x[2], reverse=True)
    thresh = 0.13764085668100007

    testy = [ 1 if i else 0 for _,_,_,i in AllDistancePairs]
    lr_probs = [ prob for _,_, prob,_ in AllDistancePairs]

    from sklearn.metrics import roc_curve
    from sklearn.metrics import roc_auc_score
    from matplotlib import pyplot

    lr_auc = roc_auc_score(testy, lr_probs)
    print("lr_auc {}".format(lr_auc))
    showAUC = False

    # # summarize scores
    #
    if showAUC:
        print('Logistic: ROC AUC=%.3f' % (lr_auc))
        # calculate roc curves

        lr_fpr, lr_tpr, _ = roc_curve(testy, lr_probs)
        # plot the roc curve for the model
        pyplot.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')
        # axis labels
        pyplot.xlabel('False Positive Rate')
        pyplot.ylabel('True Positive Rate')
        # show the legend
        pyplot.legend()
        # show the plot
        pyplot.show()




    truePoss = [0 for i in range(10)]
    mindistPos = []
    mindistNeg = []

    invRanks = []
    for i in range(len(features)):
        allIDist = []
        for j in range( len(features)):
            if i==j:
                continue
            dist = alldistances[i][j]
            allIDist.append( (dist, class_labels[j]))
        closestVals = sorted(nsmallest(10, allIDist))
        idx = 1
        found = False
        otherManu = class_labels[i].split(":")[0]
        otherPath = class_labels[i].split(":")[1]
        for v,l in closestVals:
            lManu = l.split(":")[0]
            lPath = l.split(":")[1]

            minDist = min(getNumberOfLetters(lPath), getNumberOfLetters((otherPath)))
            if minDist != -1:
                if lManu == otherManu:
                    mindistPos.append(minDist)
                else:
                    mindistNeg.append(minDist)

            if lManu == otherManu and not found:
                truePoss[idx-1]+=1
                found = True
                invRanks.append( 1/idx)

            idx+=1
        if not found:
            invRanks.append(0)

    print("final accuracy: {}".format(truePoss[0]/len(features) ))
    print("final Mean reciprocal rank: {}".format(sum(invRanks) / len(features)))

    meanLetterNumbersPos = np.array(mindistPos).mean()
    meanLetterNumbersNeg = np.array(mindistNeg).mean()
    medianLetterNumbersPos = np.median( np.array(mindistPos))
    medianLetterNumbersNeg = np.median(np.array(mindistNeg))
    print(" mean letter number in positive samples from 10 closest {}".format(meanLetterNumbersPos))
    print(" mean letter number in negative samples from 10 closest {}".format(meanLetterNumbersNeg))

    writer.add_scalar('Train/MRR', sum(invRanks) / len(features), idxWriter)
    writer.add_scalar('Train/Accuracy', truePoss[0] / len(features), idxWriter)
    writer.add_scalar('Train/auc', lr_auc, idxWriter)
    writer.add_scalar('Train/AccuracyTop5', sum(truePoss[0:5]) / len(features), idxWriter)
    writer.add_scalar('Train/AccuracyTop10', sum(truePoss[0:10]) / len(features), idxWriter)
    #writer.add_scalar('Train/meanLetterNumbersPos', meanLetterNumbersPos, idxWriter)
    #writer.add_scalar('Train/meanLetterNumbersNeg', meanLetterNumbersNeg, idxWriter)
    #writer.add_scalar('Train/medianLetterNumbersPos', medianLetterNumbersPos, idxWriter)
    #writer.add_scalar('Train/medianLetterNumbersNeg', medianLetterNumbersNeg, idxWriter)
    model.train()





def getNumberOfLetters(filename):
    filename = filename.split("/")[-1]
    plate = filename.split("-")[0]

    textFolder = r'/home/olya/Documents/fragmentsData/texts'
    folderPath = os.path.join(textFolder, plate)
    if not os.path.exists(folderPath):
        return -1
    allImages = os.listdir(folderPath)

    maxLen = 0
    imageToGet = ""
    for n in allImages:
        currLen = len(os.path.commonprefix([n, filename]))
        if currLen >= maxLen:
            imageToGet = n
            maxLen = currLen

    with open(os.path.join( folderPath, imageToGet), 'r') as f:
        allData = f.read()
    return len(allData)



def testModel(model, test_dataloader):
    #computeKMeans(model, test_dataloader)
    calcAccuracy(model, test_dataloader)
    visualizeEmbeddings(model, test_dataloader, writer)




def calculateDistances(known, unknown):
    import pandas as pd
    import shutil
    allManus = []
    allP = []
    for vec, p, _ in unknown:
        distances = []
        df =pd.DataFrame()
        for vecn, pn, manu in known:
            dist = np.linalg.norm(np.array(vec) - np.array(vecn))
            distances.append( (dist, pn, manu) )

        sortedDist = sorted(distances, key=lambda x:x[0])
        closest = sortedDist[:20]
        manus = []
        for d, path, manu in closest:
            manus.append(manu)
            toPath = p.rsplit("/",1)[0]
            toPath = os.path.join(toPath, p.rsplit("/",1)[1][:-4], 'closest')
            if not os.path.exists(toPath):
                os.makedirs(toPath)
            shutil.copy(path, os.path.join(toPath, "{}_{}_{}".format( path.rsplit("/",1)[1], manu, str(d) )))
        allManus.append(manus)
        allP.append(p.rsplit("/",1)[1][:-4])

    df['image'] = allP
    df['proposed manus'] = allManus
    df.to_csv("output.csv")




def calculateVectors(dataloader, model, known = True):
    class_labels = []
    features = []
    progress_bar = enumerate(tqdm(dataloader))
    with torch.no_grad():
        for batch_index, (data, raw_image, label) in progress_bar:
            data = data.cuda()
            output = model(data, None)

            class_labels += label
            features += output.data.cpu().numpy().tolist()

    ret = []
    for  vec, label in zip(features, class_labels):
        manu, img = label.split(":")
        if known:
            ret.append(( vec, img, manu ))
        else:
            ret.append((vec, img, None))

    return ret











def main():

    model_architecture = args.model
    resume_path = args.resume_path
    batch_size = args.batch_size
    num_workers = args.num_workers
    embedding_dimension = args.embedding_dim
    pretrained = args.pretrained




    # Size 160x160 RGB image
    lfw_transforms = transforms.Compose([
        transforms.Resize(size=(160, 160)),
        # transforms.Lambda(myTransform),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5] ,
            std=[0.5, 0.5, 0.5]
        )
    ])



    # Instantiate model
    model = set_model_architecture(
        model_architecture=model_architecture,
        pretrained=pretrained,
        embedding_dimension=embedding_dimension
    )

    # Load model to GPU or multiple GPUs if available
    model, flag_train_multi_gpu = set_model_gpu_mode(model)



    # Resume from a model checkpoint
    if resume_path:
        if os.path.isfile(resume_path):
            print("Loading checkpoint {} ...".format(resume_path))
            checkpoint = torch.load(resume_path)
            # In order to load state dict for optimizers correctly, model has to be loaded to gpu first
            if flag_train_multi_gpu:
                model.module.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint['model_state_dict'])



    known_dataloader = torch.utils.data.DataLoader(
        dataset=FragmentsDataset(
            dir=r'/home/olya/Documents/fragmentsData/DSS_Joins',
            transform=lfw_transforms,
            isTest=True,
            addBinary=False
        ),
        batch_size=16,
        num_workers=0,
        shuffle=False
    )

    unknown_dataloader = torch.utils.data.DataLoader(
        dataset=FragmentsDataset(
            dir=r'/home/olya/Documents/fragmentsData/unknownBase/DSS_Fragments/fragments_nojp1' ,
            transform=lfw_transforms,
            isTest= True,
            addBinary=False
        ),
        batch_size=16,
        num_workers=0,
        shuffle=False
    )

    knownVectors = calculateVectors(known_dataloader, model)
    unknownVrctors = calculateVectors(unknown_dataloader, model, known=False)
    calculateDistances(knownVectors, unknownVrctors)
    #calcAccuracy(model, test_dataloader)








if __name__ == '__main__':
    main()

