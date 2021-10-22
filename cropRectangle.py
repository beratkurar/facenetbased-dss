import random
import scipy.io
from PIL import Image, ImageDraw
import pandas as pd

def choosePoint(poly):
    w , h =  poly.shape
    pw = random.randint(0, w-1)
    ph = random.randint(0,h-1)
    return (pw,ph)

def valid(poly, point):
    w,h, = point
    return poly[w][h] == 1

def chooseRandomPoints(poly, pnum):
    points = []
    while pnum > 0:
        point = choosePoint(poly)
        while not valid(poly, point):
            point = choosePoint(poly)
        pnum -=1
        points.append(point)

    return points


def hasLettersForBinary( center, side, image ):
    import numpy as np

    isBinary = False
    if not isBinary:
        return True

    if side<100:
        return False
    #binary image is 2 times larget than boundary

    w, h = center
    image = image.resize( ( image.size[0]//2, image.size[1]//2  ))
    im = image.crop((h - side, w - side, h + side, w + side))


    arr = np.array(im)
    arr[arr < 100] = 1
    arr[arr>100] = 0
    h,w= arr.shape
    black = arr.sum()/(h*w)

    if black >0.1 and black <0.35:
        return True
    return False




def isValidRectable(poly, center, sideLen):
    if sideLen ==0:
        return True
    totalw, totalh = poly.shape
    w, h = center
    if w-sideLen<0 or w+sideLen>=totalw or h-sideLen<0 or h+sideLen>=totalh:
        return False
    realSum=poly[w-sideLen:w+sideLen, h-sideLen: h+sideLen].sum()
    validSum= (sideLen*2)**2
    return realSum == validSum


def adjustPoint(poly, p):
    w,h=p
    w1 =w
    while w1>0 and poly[w1][h]==1 :
        w1-=1
    w2 = w
    while  w2 < poly.shape[0] and poly[w2][h] == 1 :
        w2 += 1

    h1 = h
    while h1 > 0 and poly[w][h1] == 1 :
        h1 -= 1
    h2 = h
    while h2 < poly.shape[1] and poly[w][h2] == 1 :
        h2 += 1

    #return ( w1+(w2-w1)//2, h), (w, h1+(h2-h1)//2 )
    return (w1 + (w2 - w1) // 2, h1 + (h2 - h1) // 2)


def findLargestRectangle(center, poly, largestSideLen):
    """

    :param p1:
    :param poly:
    :param largestSideLen:
    :return:
    """
    currside = largestSideLen

    while isValidRectable(poly, center, currside ):
        currside += 1
    return currside



def _getLagestRectangle(poly, points):
    largestSideLen = 0
    largestCenter = (0,0)
    for p in points:
        p1 =  adjustPoint(poly, p)
        newSideLen = findLargestRectangle(p1, poly, largestSideLen)
        if newSideLen > largestSideLen:
            largestSideLen = newSideLen
            largestCenter = p1
    return largestCenter, largestSideLen


def _sampleRectangles(poly, points):

    allRectangles = []
    for p1 in points:
        newSideLen = findLargestRectangle(p1, poly, 0)
        if newSideLen > 0 and isValidRectable(poly, p1, newSideLen - 1):
            if newSideLen > min( poly.shape)/30:
                allRectangles.append((p1, newSideLen))
    return allRectangles



def getMatFilePath(imgFilePath):
    import os

    imgName = imgFilePath.split("/")[-1]
    splited = imgName.split("-")
    plate = splited[0]


    if "test" in imgFilePath.lower():
        boundariesFolder = r"/home/olya/Documents/fragmentsData/DSS_Joins_Test_boundaries"
    else:
        boundariesFolder = r"/home/olya/Documents/fragmentsData/unknownBase/DSS_Fragments/boundaries"
    allImages = os.listdir(r"{}/{}".format(boundariesFolder, plate))

    maxLen = 0
    imageToGet = ""
    for n in allImages:
        currLen = len(os.path.commonprefix([n, imgName]))
        if currLen >= maxLen:
            imageToGet = n
            maxLen = currLen

    matfile = os.path.join(boundariesFolder, plate, imageToGet)
    return matfile



def filterValidRectangles(allRec, imgFilePath):
    from PIL import Image
    import numpy as np
    im = Image.open(imgFilePath)
    ret = []
    for rec in allRec:
        p, sideLen = rec
        if hasLettersForBinary(p, sideLen, im):
            ret.append(rec)
    return ret



class ImageRactangle:

    def __init__(self, isTest):
        self.imToRec = {}
        orentDFPath = "/home/olya/Documents/facenetbased-dss/misc/fragmentOrentTest.csv" if isTest else "/home/olya/Documents/facenetbased-dss/misc/fragmentOrentTrain.csv"
        orent = {}
        orent1 = pd.read_csv(orentDFPath).transpose().to_dict()
        for k,v in orent1.items():
            orent[v["Unnamed: 0"]] = v["orentation"]
        self.orentation =  orent

    def getLagestRectangle( self, imgFilePath, usecache =True):
        import os
        import numpy as np

        if usecache:
            if imgFilePath in self.imToRec:
                return self.imToRec[imgFilePath]
            recFile = imgFilePath[:-4]+".npy"
            if os.path.exists(recFile):
                d=np.load(recFile,allow_pickle='TRUE').item()
                return d['center'], d['side'], d['rand_sample']



        matFile = getMatFilePath(imgFilePath)
        mat = scipy.io.loadmat(matFile)
        boundary = mat['L']
        imageName= imgFilePath.split("/")[-1]
        # if imageName in self.orentation:
        #     orent = ""#self.orentation[imageName]
        #     if orent == 'c':
        #         print("right")
        #         boundary = np.array(Image.fromarray(boundary).rotate(angle=240))
        #     elif orent =="v":
        #         boundary = np.array(Image.fromarray(boundary ).rotate(angle=90))
        #         print("left")
        #     elif orent == "x":
        #         print("upside")
        #         boundary = np.array(Image.fromarray(boundary).rotate(angle=180))
        #     else:
        #         print("orent {} regular".format(orent))
        # else:
        #     raise Exception("unknown file")

        boundary[boundary == 2] = 1
        points = chooseRandomPoints(boundary, 30)
        center, side = _getLagestRectangle(boundary, points)
        allRec = _sampleRectangles(boundary, points)
        self.imToRec[imgFilePath] = (center, side, allRec)
        runBinary = False
        if runBinary:
            allRec = filterValidRectangles(allRec, imgFilePath)
        return center, side, allRec


def cacheRactangles(folder):
    import os
    import numpy as np
    isBinary = False
    imrec=ImageRactangle(isTest = False)
    i=0
    for dir in os.listdir(folder):

        if isBinary and  not dir.endswith("_bin2"):
            continue
        currDir  =os.path.join(folder, dir)
        for f in os.listdir(currDir):
            if f.endswith("npy"):
                continue
            try:
                p = os.path.join(currDir, f)
                center,side, allRec = imrec.getLagestRectangle(p, usecache=False)
                rect = {'center': center, 'side':side, 'rand_sample' : allRec}
                rectFile = f[:-4]+".npy"
                print(i, rectFile)
                i+=1
                rectFile = os.path.join(currDir, rectFile)
                if os.path.exists(rectFile):
                    os.remove(rectFile)
                np.save(rectFile, rect )
            except Exception as e:
                print("cannot save ractangle for {}".format(f))
                raise e


def main():

    #random.seed(10)

    rec = ImageRactangle(isTest=False)

    rec.getLagestRectangle(usecache=False, imgFilePath=r"/home/olya/Documents/fragmentsData/DSS_Joins_bw/3Q14_bin2/P745-Fg029-R-C01-R01-D27022014-T134539-LR924_012.jpg")

    print("")


    # mat = scipy.io.loadmat('/home/olya/Documents/fragmentsData/DSS_Joins_boundaries/P160/P160-Fg001-R-C01-R01-D30052012-T121100-LR445_ColorCalData_IAA_Both_CC110304_110702_frag_boundaries.mat')
    #
    # boundary = mat['L']
    # boundary[boundary == 2] = 1
    # points = chooseRandomPoints(boundary, 30)
    # center, side = _getLagestRectangle(boundary, points)
    # allRec = _sampleRectangles(boundary, points)
    # w,h = center
    #
    # im = Image.fromarray(boundary*255)
    # image = "/home/olya/Documents/fragmentsData/DSS_Joins_bw/4Q182_bin2/P160-Fg001-R-C01-R01-D30052012-T121210-LR924_012_F.jpg"
    # im  = Image.open(image)
    # draw = ImageDraw.Draw(im)

    #draw.rectangle( [ 400, 700 , 400+1, 700+1 ] )
    #p = adjustPoint(boundary, (700,400))

    #draw.rectangle([(p[1], p[0]), (p[1]+1, p[0]+1)])

    # for c,s in allRec:
    #     w,h = c
    #     draw.rectangle([h - s, w - s, h + s, w + s])
    #     draw.rectangle([(h,w), (h+1, w+1)])
    # im.show()



if __name__ == "__main__":
    #main()
    cacheRactangles("/home/olya/Documents/fragmentsData/unknownBase/DSS_Fragments/fragments_nojp1")