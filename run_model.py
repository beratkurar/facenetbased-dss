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
from dataloaders.triplet_loss_dataloader import  TripletFragmentsDataset, TripletFragmentsDatasetBinary, TripletFragmentsDatasetBinaryLetters
from dataloaders.regularLoader import   LettersDataset, FragmentsDataset, FragmentsDatasetBinary, FragmentsDatasetBinaryLetters

from torch.utils.tensorboard import SummaryWriter
import datetime


name ='bin_patches'
writer = SummaryWriter('runs/traint_triplet/{}'.format(name))#datetime.datetime.now()))





def set_model_architecture(model_architecture, pretrained, embedding_dimension, binary = False):
    if model_architecture == "resnet18":
        model = Resnet18Triplet(
            embedding_dimension=embedding_dimension,
            pretrained=pretrained
        )
    elif model_architecture == "resnet34":
        model = Resnet34Triplet(
            embedding_dimension=embedding_dimension,
            pretrained=pretrained,
            binary = binary
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



def calcAccuracy(model, test_dataloader, idxWriter=0):
    import sys
    model.eval()
    class_labels=[]
    features=[]
    progress_bar = enumerate(tqdm(test_dataloader))
    with torch.no_grad():
        #for batch_index, (data,  label) in progress_bar:
        for batch_index, (data, raw_image, label) in progress_bar:
            data = data.cuda()
            output = model(data, None)

            class_labels += list(label)# list(label.numpy()) #label
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
        #otherPath = class_labels[i].split(":")[1]
        for v,l in closestVals:
            lManu = l.split(":")[0]

            if lManu == otherManu and not found:
                truePoss[idx-1]+=1
                found = True
                invRanks.append( 1/idx)

            idx+=1
        if not found:
            invRanks.append(0)

    print("final accuracy: {}".format(truePoss[0]/len(features) ))
    print("final Mean reciprocal rank: {}".format(sum(invRanks) / len(features)))


    writer.add_scalar('Train/MRR', sum(invRanks) / len(features), idxWriter)
    writer.add_scalar('Train/Accuracy', truePoss[0] / len(features), idxWriter)
    writer.add_scalar('Train/auc', lr_auc, idxWriter)
    writer.add_scalar('Train/AccuracyTop5', sum(truePoss[0:5]) / len(features), idxWriter)
    writer.add_scalar('Train/AccuracyTop10', sum(truePoss[0:10]) / len(features), idxWriter)



def myResize(im):
        from PIL import Image
        desired_size = 40

        old_size = im.size  # old_size[0] is in (width, height) format

        ratio = float(desired_size) / max(old_size)
        new_size = tuple([int(x * ratio) for x in old_size])

        # thumbnail is a in-place operation)

        im = im.resize(new_size, Image.ANTIALIAS)
        # create a new image and paste the resized on it

        new_im = Image.new("L", (desired_size, desired_size), color =255)
        new_im.paste(im, ((desired_size - new_size[0]) // 2,
                          (desired_size - new_size[1]) // 2))
        return new_im

def main():

    model_architecture = "resnet34"
    resume_path = r'/home/olya/Documents/facenetbased-dss/Model_training_checkpoints/trainedModels/model_resnet34BinaryLettersAlef_triplet_allepoch_3.pt'
    batch_size = 16
    num_workers = 0
    embedding_dimension = 128

    binary_dir = r'/home/olya/Documents/fragmentsData/Letters/LettersAlefTest'
    dir =r'/home/olya/Documents/fragmentsData/DSS_Joins_Test'
    RUN_ON_BINARY = True


    if RUN_ON_BINARY:
        lfw_transforms = transforms.Compose([
            transforms.Lambda(myResize),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5] ,
                std=[0.5]
            )
        ])
    else:
        lfw_transforms = transforms.Compose([
            transforms.Resize(size=(40, 40)),
            # transforms.Lambda(myTransform),
            transforms.ToTensor(),
            transforms.Normalize(
                mean= [0.5, 0.5, 0.5],
                std= [0.5, 0.5, 0.5]
            )
        ])


    # Instantiate model
    model = set_model_architecture(
        model_architecture=model_architecture,
        pretrained=None,
        embedding_dimension=embedding_dimension,
        binary = RUN_ON_BINARY
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



    if RUN_ON_BINARY:
        test_dataloader = torch.utils.data.DataLoader(
            dataset=FragmentsDatasetBinaryLetters(
                dir=binary_dir,
                transform=lfw_transforms,
                isTest=False
            ),
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False
        )
    else:
        test_dataloader = torch.utils.data.DataLoader(
            dataset=FragmentsDataset(
                dir=dir,
                transform=lfw_transforms,
                isTest=True,
                addBinary=False
            ),
            batch_size=batch_size,
            num_workers=0,
            shuffle=False
        )



    calcAccuracy(model, test_dataloader)




if __name__ == '__main__':
    main()

