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
from dataloaders.triplet_loss_dataloader import  TripletFragmentsDataset, TripletFragmentsDatasetBinary, TripletFragmentsDatasetBinaryLetters
from dataloaders.regularLoader import   LettersDataset, FragmentsDataset, FragmentsDatasetBinary, FragmentsDatasetBinaryLetters
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
from  dataloaders.triplet_mnist_dataloader import TripletEmnistDataset

RUN_ON_LETTERS = True


name ='BinaryLettersAlefCleanSmallRes'
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
parser.add_argument('--resume_path', default =  '/home/olya/Documents/facenetbased-dss/Model_training_checkpoints/model_resnet34BinaryLettersAlefCleanSmallRes_triplet_allepoch_1.pt', # 'Model_training_checkpoints/model_resnet34_triplet_allepoch_1.pt',
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


def set_optimizer(optimizer, model, learning_rate):
    if optimizer == "sgd":
        optimizer_model = torch.optim.SGD(model.parameters(), lr=learning_rate)

    elif optimizer == "adagrad":
        optimizer_model = torch.optim.Adagrad(model.parameters(), lr=learning_rate)

    elif optimizer == "rmsprop":
        optimizer_model = torch.optim.RMSprop(model.parameters(), lr=learning_rate)

    elif optimizer == "adam":
        optimizer_model = torch.optim.Adam(model.parameters(), lr=learning_rate)

    return optimizer_model


def validate_lfw(model, lfw_dataloader, model_architecture, epoch, epochs):
    model.eval()
    with torch.no_grad():
        l2_distance = PairwiseDistance(2).cuda()
        distances, labels = [], []

        print("Validating on LFW! ...")
        progress_bar = enumerate(tqdm(lfw_dataloader))

        for batch_index, (data_a, data_b, label) in progress_bar:
            data_a, data_b, label = data_a.cuda(), data_b.cuda(), label.cuda()

            output_a, output_b = model(data_a), model(data_b)
            distance = l2_distance.forward(output_a, output_b)  # Euclidean distance

            distances.append(distance.cpu().detach().numpy())
            labels.append(label.cpu().detach().numpy())

        labels = np.array([sublabel for label in labels for sublabel in label])
        distances = np.array([subdist for distance in distances for subdist in distance])

        true_positive_rate, false_positive_rate, precision, recall, accuracy, roc_auc, best_distances, \
        tar, far = evaluate_lfw(
            distances=distances,
            labels=labels,
            far_target=1e-2
        )
        # Print statistics and add to log
        print("Accuracy on LFW: {:.4f}+-{:.4f}\tPrecision {:.4f}+-{:.4f}\tRecall {:.4f}+-{:.4f}\t"
              "ROC Area Under Curve: {:.4f}\tBest distance threshold: {:.2f}+-{:.2f}\t"
              "TAR: {:.4f}+-{:.4f} @ FAR: {:.4f}".format(
                np.mean(accuracy),
                np.std(accuracy),
                np.mean(precision),
                np.std(precision),
                np.mean(recall),
                np.std(recall),
                roc_auc,
                np.mean(best_distances),
                np.std(best_distances),
                np.mean(tar),
                np.std(tar),
                np.mean(far)
            )
        )
        with open('logs/lfw_{}_log_triplet.txt'.format(model_architecture), 'a') as f:
            val_list = [
                epoch + 1,
                np.mean(accuracy),
                np.std(accuracy),
                np.mean(precision),
                np.std(precision),
                np.mean(recall),
                np.std(recall),
                roc_auc,
                np.mean(best_distances),
                np.std(best_distances),
                np.mean(tar)
            ]
            log = '\t'.join(str(value) for value in val_list)
            f.writelines(log + '\n')

    try:
        # Plot ROC curve
        plot_roc_lfw(
            false_positive_rate=false_positive_rate,
            true_positive_rate=true_positive_rate,
            figure_name="plots/roc_plots/roc_{}_epoch_{}_triplet.png".format(model_architecture, epoch + 1)
        )
        # Plot LFW accuracies plot
        plot_accuracy_lfw(
            log_dir="logs/lfw_{}_log_triplet.txt".format(model_architecture),
            epochs=epochs,
            figure_name="plots/lfw_accuracies_{}_triplet.png".format(model_architecture)
        )
    except Exception as e:
        print(e)

    return best_distances


def train_triplet(start_epoch, end_epoch, epochs, train_dataloader, lfw_dataloader, lfw_validation_epoch_interval,
                  model, model_architecture, optimizer_model, embedding_dimension, batch_size, margin,
                  flag_train_multi_gpu, regular_dataloader, test_loader):

    #visualizeEmbeddings(model, regular_dataloader, writer)
    binaryTrain = True
    idx =0
    addside = False
    for epoch in range(start_epoch, end_epoch):
        flag_validate_lfw = False #(epoch + 1) % lfw_validation_epoch_interval == 0 or (epoch + 1) % epochs == 0
        triplet_loss_sum = 0
        num_valid_training_triplets = 0
        l2_distance = PairwiseDistance(2).cuda()

        # Training pass
        model.train()
        progress_bar = enumerate(tqdm(train_dataloader))

        for batch_idx, (batch_sample) in progress_bar:
            try:
                idx+=1
                if addside:
                    anc_img, anc_side = batch_sample['anc_img'][0].cuda(), batch_sample['anc_img'][1]
                    pos_img, pos_side = batch_sample['pos_img'][0].cuda(), batch_sample['pos_img'][1]
                    neg_img, neg_side = batch_sample['neg_img'][0].cuda(), batch_sample['neg_img'][1]
                else:
                    anc_img  = batch_sample['anc_img'][0].cuda()
                    pos_img  = batch_sample['pos_img'][0].cuda()
                    neg_img  = batch_sample['neg_img'][0].cuda()
                    anc_side=None
                    pos_side=None
                    neg_side=None

                if binaryTrain:
                    anc_img  = anc_img.unsqueeze(0)
                    pos_img = pos_img.unsqueeze(0)
                    neg_img = neg_img.unsqueeze(0)

                # Forward pass - compute embeddings
                anc_embedding, pos_embedding, neg_embedding = model(anc_img, anc_side), model(pos_img, pos_side), model(neg_img, neg_side)



                if TRAIN_ONLY_HARD_NEGATIVE:
                    # Forward pass - choose hard negatives only for training
                    pos_dist = l2_distance.forward(anc_embedding, pos_embedding)
                    neg_dist = l2_distance.forward(anc_embedding, neg_embedding)



                    all = (neg_dist - pos_dist < margin).cpu().numpy().flatten()


                    #if TRAIN_ONLY_HARD_NEGATIVE:
                    hard_triplets = np.where(all == 1)
                    if len(hard_triplets[0]) == 0:
                        continue

                    anc_hard_embedding = anc_embedding[hard_triplets].cuda()
                    pos_hard_embedding = pos_embedding[hard_triplets].cuda()
                    neg_hard_embedding = neg_embedding[hard_triplets].cuda()

                    # Calculate triplet loss
                    triplet_loss = TripletLoss(margin=margin).forward(
                        anchor=anc_hard_embedding,
                        positive=pos_hard_embedding,
                        negative=neg_hard_embedding
                    ).cuda()
                    num_valid_training_triplets += len(anc_hard_embedding)
                else:
                    triplet_loss = TripletLoss(margin=margin).forward(
                        anchor=anc_embedding,
                        positive=pos_embedding,
                        negative=neg_embedding
                    ).cuda()
                    num_valid_training_triplets += len(anc_embedding)


                # Calculating loss
                triplet_loss_sum += triplet_loss.item()

                print("loss: {}".format(triplet_loss.item()))

                # Backward pass
                optimizer_model.zero_grad()
                triplet_loss.backward()
                optimizer_model.step()

                #torch.nn.utils.clip_grad_norm_(model.parameters(), 5)

                if batch_idx%(len(train_dataloader)//100)==0 and batch_idx != 0:
                    calcAccuracy(model, test_loader, idx)
            except Exception as e:
                print("error for image {}, {} ".format(batch_idx, batch_sample))
                raise e



        # Model only trains on hard negative triplets
        avg_triplet_loss = 0 if (num_valid_training_triplets == 0) else triplet_loss_sum / num_valid_training_triplets

        # Print training statistics and add to log
        print('Epoch {}:\tAverage Triplet Loss: {:.4f}\tNumber of valid training triplets in epoch: {}'.format(
                epoch + 1,
                avg_triplet_loss,
                num_valid_training_triplets
            )
        )
        with open('logs/{}_log_triplet.txt'.format(model_architecture), 'a') as f:
            val_list = [
                epoch + 1,
                avg_triplet_loss,
                num_valid_training_triplets
            ]
            log = '\t'.join(str(value) for value in val_list)
            f.writelines(log + '\n')

        try:
            # Plot Triplet losses plot
            plot_triplet_losses(
                log_dir="logs/{}_log_triplet.txt".format(model_architecture),
                epochs=epochs,
                figure_name="plots/triplet_losses_{}.png".format(model_architecture)
            )
        except Exception as e:
            print(e)



        # Save model checkpoint
        state = {
            'epoch': epoch + 1,
            'embedding_dimension': embedding_dimension,
            'batch_size_training': batch_size,
            'model_state_dict': model.state_dict(),
            'model_architecture': model_architecture,
            'optimizer_model_state_dict': optimizer_model.state_dict()
        }

        # For storing data parallel model's state dictionary without 'module' parameter
        if flag_train_multi_gpu:
            state['model_state_dict'] = model.module.state_dict()



        # Save model checkpoint
        torch.save(state, 'Model_training_checkpoints/model_{}_triplet_allepoch_{}.pt'.format(
                model_architecture+name,
                epoch + 1
            )
        )

        # Evaluation pass on LFW dataset
        if flag_validate_lfw:
            best_distances = validate_lfw(
                model=model,
                lfw_dataloader=lfw_dataloader,
                model_architecture=model_architecture,
                epoch=epoch,
                epochs=epochs
            )
        # For storing best euclidean distance threshold during LFW validation
        if flag_validate_lfw:
                state['best_distance_threshold'] = np.mean(best_distances)

        #visualizeEmbeddings(model, regular_dataloader, writer)




def calcStatus(dataloader):
    from collections import defaultdict
    progress_bar = enumerate(dataloader)
    allLabels = []
    with torch.no_grad():
        for batch_index, (data, raw_image, label) in progress_bar:
            allLabels += label
    d= defaultdict(int)
    for l in allLabels:
        d[l] += 1
    print("number of manuscripts {} ".format(len(list(d.keys()))))
    allVals = np.array(list(d.values()))
    print("min fragments in manu {}".format(allVals.min()))
    print("max fragments in manu {}".format(allVals.max()))
    print("mean fragments in manu {}".format(allVals.mean()))
    print("done")




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
        #otherPath = class_labels[i].split(":")[1]
        for v,l in closestVals:
            lManu = l.split(":")[0]
            #lPath = l.split(":")[1]

            # minDist = min(getNumberOfLetters(lPath), getNumberOfLetters((otherPath)))
            # if minDist != -1:
            #     if lManu == otherManu:
            #         mindistPos.append(minDist)
            #     else:
            #         mindistNeg.append(minDist)

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



def computeKMeans(model, dataloader):
    from sklearn.cluster import KMeans
    from collections import defaultdict
    class_labels=[]
    features=[]
    progress_bar = enumerate(tqdm(dataloader))
    with torch.no_grad():
        for batch_index, (data, raw_image, label) in progress_bar:
            data = data.cuda()
            output = model(data)
            class_labels += label
            features += output.data.cpu().numpy().tolist()

    X = np.array(features)
    kmeans = KMeans(n_clusters=73, random_state=0).fit(X)
    print(kmeans.labels_)

    from sklearn import metrics
    class_labelss = list(set(class_labels))
    labelToNum = {class_labelss[i] : i  for i in range(len(class_labelss)) }
    labels_true = [ labelToNum[l ] for l  in class_labels ]

    val = metrics.adjusted_mutual_info_score(labels_true, kmeans.labels_)
    print("k means metric: {}".format(val))
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import seaborn as sns
    import pandas as pd
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(features)
    df = pd.DataFrame()
    df['pca-one'] = pca_result[:, 0]
    df['pca-two'] = pca_result[:, 1]
    df['pca-three'] = pca_result[:, 2]
    df['y'] = class_labels
    print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))
    plt.figure(figsize=(16, 10))
    sns.scatterplot(
        x="pca-one", y="pca-two",
        hue="y",
        palette=sns.color_palette("hls", len(list(set(class_labels)))),
        data=df,
        legend="full",
        alpha=0.3
    )
    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(features)
    df = pd.DataFrame()
    df['tsne-2d-one'] = tsne_results[:, 0]
    df['tsne-2d-two'] = tsne_results[:, 1]
    df['y'] = class_labels
    plt.figure(figsize=(16, 10))
    sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        hue="y",
        palette=sns.color_palette("hls", len(list(set(class_labels)))),
        data=df,
        legend="full",
        alpha=0.3
    )



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




def getVector(fullName, model, transform):
    from PIL import Image
    im = Image.open(fullName)
    im = transform(im)

    vec = model(im.unsqueeze(0).cuda(), None)
    vec = vec.cpu().data.numpy()
    return vec



def compare2Manus(model, transform):
    from sklearn.cluster import KMeans
    root = '/home/olya/Documents/fragmentsData/LettersAlefTest'
    first = '4Q84_bin1'
    second = '4Q111_binq'
    firstVecs = []
    secondVecs = []

    for f in os.listdir(os.path.join(root, first)):
        fullName = os.path.join(root, first, f)
        vec = getVector(fullName, model, transform)
        firstVecs.append(vec)

    for f in os.listdir(os.path.join(root, second)):
        fullName = os.path.join(root, second, f)
        vec = getVector(fullName, model, transform)
        secondVecs.append(vec)

    X = np.array(firstVecs+secondVecs)
    X = X.reshape((X.shape[0], X.shape[2]))
    labels_true = [1 for i in firstVecs]
    labels_true += [2 for _ in secondVecs]
    kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
    print(kmeans.labels_)

    from sklearn import metrics
    #class_labelss = list(set(labels))
    #labelToNum = {class_labelss[i]: i for i in range(len(class_labelss))}
    #labels_true = [labelToNum[l] for l in class_labels]

    val = metrics.adjusted_mutual_info_score(labels_true, kmeans.labels_)
    print("k means metric: {}".format(val))
    # from sklearn.decomposition import PCA
    # import matplotlib.pyplot as plt
    # from mpl_toolkits.mplot3d import Axes3D
    # import seaborn as sns
    # import pandas as pd
    # pca = PCA(n_components=3)
    # pca_result = pca.fit_transform(features)
    # df = pd.DataFrame()
    # df['pca-one'] = pca_result[:, 0]
    # df['pca-two'] = pca_result[:, 1]
    # df['pca-three'] = pca_result[:, 2]
    # df['y'] = class_labels
    # print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))
    # plt.figure(figsize=(16, 10))
    # sns.scatterplot(
    #     x="pca-one", y="pca-two",
    #     hue="y",
    #     palette=sns.color_palette("hls", len(list(set(class_labels)))),
    #     data=df,
    #     legend="full",
    #     alpha=0.3
    # )
    # from sklearn.manifold import TSNE
    # tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    # tsne_results = tsne.fit_transform(features)
    # df = pd.DataFrame()
    # df['tsne-2d-one'] = tsne_results[:, 0]
    # df['tsne-2d-two'] = tsne_results[:, 1]
    # df['y'] = class_labels
    # plt.figure(figsize=(16, 10))
    # sns.scatterplot(
    #     x="tsne-2d-one", y="tsne-2d-two",
    #     hue="y",
    #     palette=sns.color_palette("hls", len(list(set(class_labels)))),
    #     data=df,
    #     legend="full",
    #     alpha=0.3
    # )







def main():
    TRAIN_ON_BINARY = True
    ADD_BINARY = False
    MNIST = False
    if not TRAIN_ON_BINARY:
        dataroot =  r'/home/olya/Documents/fragmentsData/LettersAlef'#r'/home/olya/Documents/Data/DSS/DSS_Joins'  #r"/home/olya/Documents/fragmentsData/perManu" #args.dataroot
    else:
        dataroot = r'/home/olya/Documents/fragmentsData/DSS_Joins_bw'
    lfw_dataroot = args.lfw
    dataset_csv = args.dataset_csv
    lfw_batch_size = args.lfw_batch_size
    lfw_validation_epoch_interval = args.lfw_validation_epoch_interval
    model_architecture = args.model
    epochs = args.epochs
    training_triplets_path = args.training_triplets_path
    num_triplets_train = args.num_triplets_train
    resume_path = args.resume_path
    batch_size = args.batch_size
    num_workers = args.num_workers
    embedding_dimension = args.embedding_dim
    pretrained = args.pretrained
    optimizer = args.optimizer
    learning_rate = args.lr
    margin = args.margin
    start_epoch = 0

    def myTransform(pilImages):
        import numpy as np
        import imgaug as ia
        import imgaug.augmenters as iaa
        from PIL import Image

        seq = iaa.Sequential([
            # iaa.LinearContrast((0.5, 2.0), per_channel=0.5),
            # iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
            # iaa.Multiply((0.5, 1.5), per_channel=0.2),
            #
             iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),

            # Try only grayscale
            #iaa.Grayscale(alpha=(0.0, 1.0)),

        ])
        images = np.array(pilImages)

        augmentedImages = seq(images=[images])

        images = [Image.fromarray(i) for i in augmentedImages]
        return images[0]



    def myResize(im):
        from PIL import Image
        desired_size = 40
        resized= []


        old_size = im.size  # old_size[0] is in (width, height) format

        ratio = float(desired_size) / max(old_size)
        new_size = tuple([int(x * ratio) for x in old_size])
        # use thumbnail() or resize() method to resize the input image

        # thumbnail is a in-place operation

        # im.thumbnail(new_size, Image.ANTIALIAS)

        im = im.resize(new_size, Image.ANTIALIAS)
        # create a new image and paste the resized on it

        new_im = Image.new("L", (desired_size, desired_size), color =255)
        new_im.paste(im, ((desired_size - new_size[0]) // 2,
                          (desired_size - new_size[1]) // 2))

        return new_im

    # Define image data pre-processing transforms
    #   ToTensor() normalizes pixel values between [0, 1]
    #   Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) normalizes pixel values between [-1, 1]
    data_transforms = transforms.Compose([
        #transforms.Resize(size=(32,32)),
        transforms.Lambda(myResize),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        #transforms.Lambda(myTransform),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5, 0.5] if ADD_BINARY else [0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5, 0.5] if ADD_BINARY else [0.5, 0.5, 0.5]
        )
    ])

    data_transforms_bin = transforms.Compose([
        #transforms.Resize(size=(40, 40)),
        transforms.Lambda(myResize),
        #transforms.RandomHorizontalFlip(),
        #transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5],
            std=[0.5]
        )
    ])



    if TRAIN_ON_BINARY:
        tripletsPath =  '/home/olya/Documents/facenetbased-dss/datasets/training_fragment_triplets_letters500000_bin.npy'#None #  "/home/olya/Documents/facenetbased-dss/datasets/training_fragment_triplets_500000_bin.npy"
        train_dataloader = torch.utils.data.DataLoader(
            dataset=TripletFragmentsDatasetBinaryLetters(
                root_dir=dataroot,
                num_triplets=num_triplets_train,
                training_triplets_path= tripletsPath,
                transform=data_transforms_bin
            ),
            batch_size=batch_size,
            num_workers=8,
            shuffle=True
        )
        # Size 160x160 RGB image
        lfw_transforms = transforms.Compose([
            transforms.Lambda(myResize),
            #transforms.Resize(size=(40, 40)), #think is we need resize
            # transforms.Lambda(myTransform),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5] ,
                std=[0.5]
            )
        ])
    else:
        tripletsPath =  "/home/olya/Documents/facenetbased-dss/datasets/training_fragment_triplets_1100000.npy" #if ADD_BINARY else "/home/olya/Documents/facenetbased-dss/datasets/training_fragment_triplets_nobin_1100000.npy"
        train_dataloader = torch.utils.data.DataLoader(
            dataset=TripletFragmentsDataset(
                root_dir=dataroot,
                num_triplets=num_triplets_train,
                training_triplets_path=tripletsPath,
                transform=data_transforms,
                addBinary= ADD_BINARY
            ),
            batch_size=batch_size,
            num_workers=0,
            shuffle=True
        )
        # Size 160x160 RGB image
        lfw_transforms = transforms.Compose([
            transforms.Resize(size=(40, 40)),
            # transforms.Lambda(myTransform),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5, .5] if ADD_BINARY else [0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5, .5] if ADD_BINARY else [0.5, 0.5, 0.5]
            )
        ])

    # lfw_dataloader = torch.utils.data.DataLoader(
    #     dataset=LFWDataset(
    #         dir=r'/home/olya/Documents/fragmentsData/DSS_Joins_Test',
    #         transform=lfw_transforms,
    #         addBinary=ADD_BINARY
    #     ),
    #     batch_size=lfw_batch_size,
    #     num_workers=num_workers,
    #     shuffle=False
    # )
    #
    # regular_dataloader = torch.utils.data.DataLoader(
    #     dataset=LettersDataset(
    #         dir=r'/home/olya/Documents/fragmentsData/perManu',
    #         transform=lfw_transforms
    #     ),
    #     batch_size=16,
    #     num_workers=num_workers,
    #     shuffle=False
    # )





    # Instantiate model
    if MNIST:
        from models.LeNet import Net
        model = Net()
    else:
        model = set_model_architecture(
            model_architecture=model_architecture,
            pretrained=pretrained,
            embedding_dimension=embedding_dimension
        )

    # Load model to GPU or multiple GPUs if available
    model, flag_train_multi_gpu = set_model_gpu_mode(model)

    # Set optimizer
    optimizer_model = set_optimizer(
        optimizer=optimizer,
        model=model,
        learning_rate=learning_rate
    )

    # Resume from a model checkpoint
    if resume_path:
        if os.path.isfile(resume_path):
            print("Loading checkpoint {} ...".format(resume_path))

            checkpoint = torch.load(resume_path)
            start_epoch = checkpoint['epoch']

            # In order to load state dict for optimizers correctly, model has to be loaded to gpu first
            if flag_train_multi_gpu:
                model.module.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint['model_state_dict'])

            optimizer_model.load_state_dict(checkpoint['optimizer_model_state_dict'])

            print("Checkpoint loaded: start epoch from checkpoint = {}\nRunning for {} epochs.\n".format(
                    start_epoch,
                    epochs - start_epoch
                )
            )
        else:
            print("WARNING: No checkpoint found at {}!\nTraining from scratch.".format(resume_path))



    freezeLayers = False
    if freezeLayers:
        for param in model.model.parameters():
           param.requires_grad = False
        input_features_fc_layer = model.model.fc.in_features
        #Output embedding
        model.model.fc = nn.Linear(input_features_fc_layer, 128)

    if MNIST:
        from dataloaders.mnist_test import EmnistDatasetTest

        test_dataloader = torch.utils.data.DataLoader(
            dataset=EmnistDatasetTest(
                data_transforms=lfw_transforms
            ),
            batch_size=16,
            num_workers=num_workers,
            shuffle=False
        )

    elif TRAIN_ON_BINARY:
        test_dataloader = torch.utils.data.DataLoader(
            dataset=FragmentsDatasetBinaryLetters(
                dir=r'/home/olya/Documents/fragmentsData/LettersAlefTest',
                transform=lfw_transforms,
                isTest=False
            ),
            batch_size=16,
            num_workers=num_workers,
            shuffle=False
        )
    else:
        test_dataloader = torch.utils.data.DataLoader(
            dataset=FragmentsDataset(
                dir=r'/home/olya/Documents/fragmentsData/DSS_Joins_Test',
                transform=lfw_transforms,
                isTest= True,
                addBinary=ADD_BINARY
            ),
            batch_size=16,
            num_workers=0,
            shuffle=False
        )
    #howMinLettersEffectEmbedding(model, test_dataloader)
    #calcAccuracy(model, test_dataloader)
    #calcStatus(test_dataloader)
    #testModel(model, test_dataloader)

    compare2Manus(model, data_transforms_bin)

    calcAccuracy(model, test_dataloader)

    # Start Training loop
    print("Training using triplet loss on {} triplets starting for {} epochs:\n".format(
            num_triplets_train,
            epochs - start_epoch
        )
    )

    start_epoch = start_epoch
    end_epoch = start_epoch + epochs
    #
    # best_distances = validate_lfw(
    #         model=model,
    #         lfw_dataloader=lfw_dataloader,
    #         model_architecture=model_architecture,
    #         epoch=0,
    #         epochs=epochs
    #     )

    # Start training model using Triplet Loss
    train_triplet(
        start_epoch=start_epoch,
        end_epoch=end_epoch,
        epochs=epochs,
        train_dataloader=train_dataloader,
        lfw_dataloader=None, #lfw_dataloader,
        lfw_validation_epoch_interval=lfw_validation_epoch_interval,
        model=model,
        model_architecture=model_architecture,
        optimizer_model=optimizer_model,
        embedding_dimension=embedding_dimension,
        batch_size=batch_size,
        margin=margin,
        flag_train_multi_gpu=flag_train_multi_gpu,
        regular_dataloader=None,
        test_loader = test_dataloader
    )



if __name__ == '__main__':
    main()

